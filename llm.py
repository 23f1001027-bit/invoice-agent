"""Decision layer.

1. Python pulls the facts from the cover sheet and the candidate evidence paragraphs
   (every paragraph with bracketed references except the cover-sheet one).  No tokens.
2. ONE small model call per batch sees only those candidate paragraphs and picks the
   decisive one and the action (~150 input tokens per package instead of ~1,000).
3. If the model is unavailable (429, timeout), a keyword classifier decides instead,
   so a provider failure never turns into one blanket action.
"""
import json
import os
import re
import threading
import time

import httpx

LLM_URL = os.environ.get("LLM_URL", "https://aipipe.org/openai/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
TOKEN = os.environ.get("AIPIPE_TOKEN", "")
MAX_PARALLEL = int(os.environ.get("LLM_PARALLEL", "2"))
CLIENT = httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0))
GATE = threading.Semaphore(MAX_PARALLEL)          # global: protects the provider rate limit

REF_RE = re.compile(r"\[[A-Za-z0-9][^\[\]\"\n,]{0,80}\]")
ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
DECOY_RE = re.compile(r"^\s*(archive|training|example|historic|superseded|old draft)|non-operative|"
                      r"not (a|an) (case )?instruction|vocabulary, not|explicitly labelled as training", re.I)
COVER_RE = re.compile(r"supplier\s+(?P<v>[^;]+?)\s*;\s*invoice\s+(?P<n>[^;\s]+)\s*;\s*stated total\s+"
                      r"(?P<c>[A-Z]{3})\s*(?P<a>[0-9][0-9,]*(?:\.[0-9]+)?)", re.I)
ZERO_DEC = {"JPY", "KRW", "VND", "CLP", "ISK"}

SYSTEM = """You are an accounts-payable controller. Each invoice package below lists its candidate evidence paragraphs (A, B, C...). Exactly one paragraph is DECISIVE: it states the conclusion for THIS invoice. Archive notes, training appendices, examples and superseded drafts are never decisive, even if they contain action words.

Choose the action that the decisive paragraph determines:
- settle_invoice: records reconcile and the amount is WITHIN autonomous authority; nothing pending.
- request_approval: commercially valid / payable but OUTSIDE the operator's authority, so the approval workflow is needed.
- hold_invoice: payment must pause until a stated verification completes (callback, certificate, confirmation pending).
- reject_duplicate: this SAME invoice (same vendor and invoice number) was already paid. A payment for a DIFFERENT invoice is not a duplicate.
- open_exception: material records conflict (amount, currency, vendor, quantity, bank details disagree) and need an exception workflow.
Respect negation ("no hold is needed", "is not a duplicate", "does not exceed").

Reply with JSON only: {"decisions":[{"index":0,"paragraph":"A","action":"...","why":"one short sentence"}]}"""


# ------------------------------------------------------------------ extraction (no tokens)
def _texts(pkg):
    docs = pkg.get("documents") if isinstance(pkg, dict) else None
    if isinstance(docs, list):
        return [(str(d.get("name", "")), str(d.get("text", ""))) for d in docs if isinstance(d, dict)]
    out = []

    def walk(o, label):
        if isinstance(o, str):
            out.append((label, o))
        elif isinstance(o, dict):
            for k, v in o.items():
                if k not in ("packageId", "receivedAt"):
                    walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, label)
    walk(pkg, "")
    return out


def extract(pkg):
    facts, candidates, cover_para = None, [], None
    for name, text in _texts(pkg):
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            m = COVER_RE.search(para)
            if m and facts is None:
                cur = m["c"].upper()
                amt = float(m["a"].replace(",", ""))
                facts = {"vendorName": m["v"].strip(), "invoiceNumber": m["n"].strip().rstrip(".,"),
                         "amountMinor": int(round(amt * (1 if cur in ZERO_DEC else 100))), "currency": cur}
                cover_para = para
                continue
            refs = list(dict.fromkeys(REF_RE.findall(para)))
            if refs:
                candidates.append({"text": para, "refs": refs, "decoy": bool(DECOY_RE.search(para)),
                                   "doc": name})
    # most likely decisive first: three refs, not a decoy
    candidates.sort(key=lambda c: (c["decoy"], len(c["refs"]) != 3))
    return facts, cover_para, candidates


# ------------------------------------------------------------------ keyword fallback
def keyword_action(text):
    t = text.lower()
    neg = lambda w: re.search(r"\b(no|not|never|without|nor|rather than)\b[^.]{0,40}" + w, t)
    if re.search(r"conflict|mismatch|disagree|do not agree|does not agree|inconsistent|differs? from|"
                 r"discrepanc|exception workflow", t) and not neg(r"(conflict|mismatch|discrepanc)"):
        return "open_exception"
    if re.search(r"already (been )?paid|previously paid|paid in full|duplicate (payment|submission|of)|"
                 r"same invoice", t) and not neg(r"(paid|duplicate)"):
        return "reject_duplicate"
    if re.search(r"outside[^.]{0,40}authority|exceeds?[^.]{0,40}(authority|limit|threshold)|"
                 r"approval workflow|financial-approval|requires? (an? )?approv", t):
        return "request_approval"
    if re.search(r"\bhold\b|pause|pending|callback|until[^.]{0,60}(verif|confirm|receiv)|awaiting", t) \
            and not neg(r"(hold|pause|pending)"):
        return "hold_invoice"
    return "settle_invoice"


# ------------------------------------------------------------------ model call
def _post(user, deadline):
    body = {"model": LLM_MODEL, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    delay = 1.5
    while True:
        left = deadline - time.time()
        if left < 4:
            raise TimeoutError("no time left for the model")
        with GATE:
            r = CLIENT.post(LLM_URL, headers={"Authorization": f"Bearer {TOKEN}"}, json=body,
                            timeout=httpx.Timeout(min(25.0, left), connect=5.0))
        if r.status_code in (429, 500, 502, 503) and deadline - time.time() > delay + 6:
            print("LLM", r.status_code, "retrying in", delay, flush=True)
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        j = r.json()
        print("LLM usage", json.dumps(j.get("usage", {})), flush=True)
        text = j["choices"][0]["message"]["content"]
        return json.loads(re.search(r"\{.*\}", text, re.S).group(0)).get("decisions", [])


def llm(packages):
    """Returns raw decision dicts aligned with `packages` (core.normalize validates them)."""
    t0 = time.time()
    extracted = [extract(p) for p in packages]
    blocks = []
    for i, (facts, cover, cands) in enumerate(extracted):
        lines = [f"=== PACKAGE index={i} ==="]
        if facts:
            lines.append(f"Invoice: {facts['vendorName']} / {facts['invoiceNumber']} / "
                         f"{facts['currency']} {facts['amountMinor']} minor units")
        elif cover:
            lines.append("Cover: " + cover)
        for k, c in enumerate(cands[:6]):
            lines.append(f"{chr(65 + k)}: {c['text']}")
        blocks.append("\n".join(lines))

    model = {}
    try:
        for d in _post("\n\n".join(blocks), t0 + 32):
            if isinstance(d, dict) and isinstance(d.get("index"), int):
                model[d["index"]] = d
    except Exception as exc:
        print("model unavailable, using keyword rules:", repr(exc), flush=True)

    out = []
    for i, (facts, cover, cands) in enumerate(extracted):
        if not cands:
            out.append(None)
            continue
        d = model.get(i) or {}
        k = ord(str(d.get("paragraph", "A")).strip()[:1].upper() or "A") - 65
        chosen = cands[k] if 0 <= k < len(cands[:6]) and not cands[k]["decoy"] else cands[0]
        action = d.get("action") if d.get("action") in ACTIONS else keyword_action(chosen["text"])
        src = "model" if d.get("action") in ACTIONS else "rules"
        if facts is None:
            facts = {"vendorName": "", "invoiceNumber": "", "amountMinor": 0, "currency": ""}
            print("COVER NOT PARSED", json.dumps(cover or "")[:300], flush=True)
        print("PICK", i, src, action, "|", chosen["text"][:160], flush=True)
        out.append({"action": action, "decisive_paragraph": chosen["text"], "evidenceRefs": chosen["refs"][:3],
                    "facts": dict(facts), "rationale": str(d.get("why", "")),
                    "_quote": chosen["text"], "_src": src})
    return out

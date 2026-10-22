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
RULES_WIN = os.environ.get("RULES_WIN", "1") == "1"
MODEL_BUDGET = float(os.environ.get("MODEL_BUDGET", "15"))
MAX_PARALLEL = int(os.environ.get("LLM_PARALLEL", "2"))
CLIENT = httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0))
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
- settle_invoice: ONLY when the decisive paragraph affirmatively says records reconcile, the amount is WITHIN autonomous authority, and nothing is pending, conflicting or already paid. If in doubt between settle_invoice and another action, choose the other action.
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
    """Deterministic reading of the decisive paragraph. Returns an action, or None when unsure."""
    t = " ".join(text.lower().split())
    has = lambda p: re.search(p, t) is not None

    # 1. duplicate: an earlier payment of the SAME invoice (a different invoice is not a duplicate)
    different = has(r"different (signed )?(commercial )?invoice|not (a|the) (same|duplicate)|is not a duplicate|"
                    r"no (earlier|prior) (payment|posting)")
    if not different and has(r"(earlier|prior|previous|completed) (payment|posting|settled entry|remittance)|"
                             r"already (been )?(paid|settled)|paid-items ledger|same commercial key|"
                             r"earlier settled entry|duplicate (payment|submission)"):
        return "reject_duplicate"

    # 2. conflicting records -> exception workflow
    no_discrepancy = has(r"\bno\b[^.]{0,80}(discrepanc|conflict|mismatch|exception)[^.]{0,20}(remains|exists|found)?")
    if has(r"\bconflicts?\b|contradictory|mutually inconsistent|mutually incompatible|does not reconcile|"
           r"do not reconcile|cannot be reconciled|outside the permitted reconciliation tolerance|beyond tolerance|"
           r"\bmismatch|disagree|exception workflow") and not has(r"no (price|quantity)?[^.]{0,60}conflict"):
        return "open_exception"
    if has(r"discrepanc") and not no_discrepancy:
        return "open_exception"

    # 3. over delegated authority -> approval
    if has(r"outside[^.]{0,60}authority|exceeds?[^.]{0,60}(authority|delegat|limit|threshold|ceiling|autonomous)|"
           r"only up to|above[^.]{0,40}(authority|delegation|limit)|approval workflow|financial-approval|"
           r"requires? (an? )?(named |senior |financial )?approv|beyond[^.]{0,40}(authority|delegation)"):
        return "request_approval"

    # 4. pending verification -> hold
    if has(r"callback|beneficiary details|destination-account|bank account|bank-detail change|"
           r"has not (yet )?(been )?(completed|confirmed|verified)|not yet (been )?(verified|confirmed|received)|"
           r"neither confirmed nor rejected|pending (verification|confirmation|receipt)|awaiting|until[^.]{0,60}"
           r"(verif|confirm|complete)|\bhold\b|\bpause"):
        if not has(r"no (hold|pause|pending)|not (on )?hold|hold is not|without (a )?hold"):
            return "hold_invoice"

    # 5. clean and within authority -> settle
    if has(r"within[^.]{0,60}(authority|limit|delegation|threshold)|reconcile[ds]?[^.]{0,90}(without|with no) "
           r"(an )?exception|with no exception|agree within the stated rounding tolerance|same payable total|covers every billed unit|"
           r"(may|can|should) be (settled|paid|released)|cleared for (payment|settlement)|approved for (payment|settlement)|"
           r"release(d)? for payment|eligible for (payment|settlement)|no payment was made for this|"
           r"no (paid item|prior settlement|earlier posting|prior payment) (with|for)|settle it\b"):
        return "settle_invoice"
    return None


# ------------------------------------------------------------------ model call
_breaker = {"until": 0.0}                         # skip the model for a while after a config error


def _post(user, deadline):
    if time.time() < _breaker["until"]:
        raise RuntimeError("model disabled after a recent configuration error")
    body = {"model": LLM_MODEL, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    delay = 1.0
    while True:
        left = deadline - time.time()
        if left < 3 or not GATE.acquire(timeout=max(0.1, left - 3)):
            raise TimeoutError("no time left for the model")
        try:
            left = deadline - time.time()
            r = CLIENT.post(LLM_URL, headers={"Authorization": f"Bearer {TOKEN}"}, json=body,
                            timeout=httpx.Timeout(max(1.0, left), connect=min(5.0, max(1.0, left))))
        finally:
            GATE.release()
        if r.status_code in (429, 500, 502, 503) and deadline - time.time() > delay + 4:
            print("LLM", r.status_code, "retrying in", delay, flush=True)
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code in (400, 401, 403, 404):
            _breaker["until"] = time.time() + 600
        r.raise_for_status()
        j = r.json()
        print("LLM usage", json.dumps(j.get("usage", {})), flush=True)
        text = j["choices"][0]["message"]["content"]
        return json.loads(re.search(r"\{.*\}", text, re.S).group(0)).get("decisions", [])


def _block(i, facts, cover, cands):
    lines = [f"=== PACKAGE index={i} ==="]
    if facts:
        lines.append(f"Invoice: {facts['vendorName']} / {facts['invoiceNumber']} / "
                     f"{facts['currency']} {facts['amountMinor']} minor units")
    elif cover:
        lines.append("Cover: " + cover)
    for k, c in enumerate(cands[:6]):
        lines.append(f"{chr(65 + k)}: {c['text']}")
    return "\n".join(lines)


def llm(packages):
    """Returns raw decision dicts aligned with `packages` (core.normalize validates them).

    Packages whose decisive paragraph the rules read with confidence need no model call;
    only the remaining ones go to the model, in one call, within a hard time budget.
    """
    t0 = time.time()
    extracted = [extract(p) for p in packages]
    first_rule = {}
    for i, (facts, cover, cands) in enumerate(extracted):
        clean = [c for c in cands if not c["decoy"]]
        if len(clean) == 1 or (clean and len(clean[0]["refs"]) == 3 and all(len(c["refs"]) != 3 for c in clean[1:])):
            first_rule[i] = keyword_action(clean[0]["text"])
    ask = [i for i, (_, _, cands) in enumerate(extracted) if cands and not first_rule.get(i)]

    model = {}
    if ask:
        try:
            user = "\n\n".join(_block(n, *extracted[i]) for n, i in enumerate(ask))
            for d in _post(user, t0 + MODEL_BUDGET):
                if isinstance(d, dict) and isinstance(d.get("index"), int) and 0 <= d["index"] < len(ask):
                    model[ask[d["index"]]] = d
        except httpx.HTTPStatusError as exc:
            print("model unavailable, using keyword rules:", exc.response.status_code, exc.response.text[:400], flush=True)
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
        rule = keyword_action(chosen["text"])
        m_act = d.get("action") if d.get("action") in ACTIONS else None
        if rule and m_act and rule != m_act:
            print("DISAGREE model", m_act, "rules", rule, flush=True)
        if rule and (RULES_WIN or not m_act):
            action, src = rule, ("model+rules" if m_act == rule else "rules")
        elif m_act:
            action, src = m_act, "model"
        else:
            action, src = "hold_invoice", "rules-default"
        if action == "settle_invoice" and rule != "settle_invoice":
            # settling the wrong invoice is the costliest mistake: only settle when the text clearly allows it
            print("SETTLE GUARD", src, "->", rule or "hold_invoice", flush=True)
            action, src = rule or "hold_invoice", src + "+guard"
        if facts is None:
            facts = {"vendorName": "", "invoiceNumber": "", "amountMinor": 0, "currency": ""}
            print("COVER NOT PARSED", json.dumps(cover or "")[:300], flush=True)
        print("PICK", i, src, action, "|", " ".join(chosen["text"].split())[:420], flush=True)
        out.append({"action": action, "decisive_paragraph": chosen["text"], "evidenceRefs": chosen["refs"][:3],
                    "facts": dict(facts), "rationale": str(d.get("why", "")),
                    "_quote": chosen["text"], "_src": src})
    return out

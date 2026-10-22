"""Model layer: one batched call per chunk of packages. Any OpenAI-compatible API works.

Token savers:
- packages are sent as plain text (not JSON-escaped), with an id on every paragraph that
  carries bracketed references, so the model answers with a short paragraph id instead of
  copying the paragraph back;
- the model writes a short analysis only; the rationale is assembled in Python.
"""
import json
import os
import re
import time

import httpx

LLM_URL = os.environ.get("LLM_URL", "https://aipipe.org/openai/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5-mini")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "gpt-4.1-mini")
REASONING = os.environ.get("REASONING_EFFORT", "low")
TOKEN = os.environ.get("AIPIPE_TOKEN", "")
CLIENT = httpx.Client(timeout=httpx.Timeout(40.0, connect=5.0))

REF_RE = re.compile(r"\[[A-Za-z0-9][^\[\]\"\n,]{0,80}\]")

SYSTEM = """You are an accounts-payable controller. For EACH invoice package choose exactly ONE action:

- settle_invoice: valid, records reconcile, amount within autonomous authority.
- request_approval: commercially valid, but amount/terms are OUTSIDE delegated authority, so an approver must sign off.
- hold_invoice: payment must PAUSE until a stated verification completes (bank-detail callback, missing certificate, pending confirmation...).
- reject_duplicate: the SAME commercial invoice (same vendor + same invoice number) was ALREADY PAID. A payment for a DIFFERENT invoice number is NOT a duplicate.
- open_exception: material records conflict (amount, currency, vendor, quantity disagree between documents) and need an exception workflow.

Packages contain several documents. Paragraphs that carry bracketed references are labelled like <P12>.
The cover/intake sheet only summarises the invoice (supplier, invoice number, total); its reference is never decisive.
Ignore noise: office/queue/footer/timestamp text, training examples, archive or historic cases, superseded drafts or old policy revisions, and sentences that merely mention action words. Respect negation ("no hold is required", "not a duplicate"), thresholds and dates.

For each package:
1. Find the ONE labelled paragraph that states the conclusion for THIS invoice under the CURRENT policy revision (normally it carries exactly three bracketed references).
2. "analysis": max 45 words with the controlling facts (e.g. amount vs limit, which invoice was paid, which verification is pending, which records conflict).
3. "action": the action that paragraph determines.
4. "facts": from the CONTROLLING invoice (cover sheet values unless a signed record explicitly corrects them). amountMinor = integer in minor units (x100 for INR/USD/EUR/GBP/AED etc., x1 for JPY/KRW). "amountText" = the total exactly as printed.

Reply with JSON only:
{"decisions":[{"index":0,"paragraph":"P12","analysis":"...","action":"...","facts":{"vendorName":"...","invoiceNumber":"...","amountText":"...","amountMinor":0,"currency":"INR"}}]}
Return exactly one decision per package."""

_logged = {"n": 0}


def _paragraphs(s):
    """Split a document string into paragraphs; blocks with >3 refs are split per line."""
    out = []
    for block in re.split(r"\n\s*\n", s):
        if len(REF_RE.findall(block)) > 3 and "\n" in block:
            out.extend(l for l in block.split("\n") if l.strip())
        elif block.strip():
            out.append(block)
    return out


def render(pkg, pmap):
    """Render a package as labelled plain text. pmap collects paragraph id -> text."""
    lines = []

    def walk(obj, label):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "packageId":
                    continue
                walk(v, f"{label}.{k}" if label else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{label}[{i}]")
        elif isinstance(obj, str):
            if "\n" not in obj and len(obj) < 120 and not REF_RE.search(obj):
                lines.append(f"{label}: {obj}")
                return
            lines.append(f"## {label}")
            for para in _paragraphs(obj):
                if REF_RE.search(para):
                    pid = f"P{len(pmap)}"
                    pmap[pid] = para
                    lines.append(f"<{pid}> {para.strip()}")
                else:
                    lines.append(para.strip())
        elif obj is not None:
            lines.append(f"{label}: {obj}")

    walk(pkg, "")
    return "\n".join(lines)


def _amount_from_text(text, currency):
    t = re.sub(r"[^0-9.]", "", str(text or ""))
    if not t or t.count(".") > 1:
        return None
    zero_dec = currency.upper() in {"JPY", "KRW"}
    if "." in t:
        return int(round(float(t) * (1 if zero_dec else 100)))
    # no decimals printed: only trust it when it looks like a formatted major amount
    if re.search(r"[,₹$€£¥]|[A-Z]{3}", str(text)):
        return int(t) * (1 if zero_dec else 100)
    return None


def _post(model, user):
    body = {"model": model, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    if model.startswith(("gpt-5", "o3", "o4")):
        body["reasoning_effort"] = REASONING
    else:
        body["temperature"] = 0
    r = CLIENT.post(LLM_URL, headers={"Authorization": f"Bearer {TOKEN}"}, json=body)
    r.raise_for_status()
    j = r.json()
    print("LLM usage", model, json.dumps(j.get("usage", {})), flush=True)
    text = j["choices"][0]["message"]["content"]
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0)).get("decisions", [])


def llm(packages):
    """packages: list of package dicts. Returns a list of raw decision dicts aligned with `packages`."""
    if _logged["n"] < 1 and packages:           # one sample in the logs helps tuning
        _logged["n"] += 1
        print("SAMPLE PACKAGE:", json.dumps(packages[0], ensure_ascii=False)[:6000], flush=True)
    pmaps, chunks = [], []
    for i, p in enumerate(packages):
        pm = {}
        chunks.append(f"=== PACKAGE index={i} ===\n{render(p, pm)}")
        pmaps.append(pm)
    user = "\n\n".join(chunks)

    t0 = time.time()
    try:
        decisions = _post(LLM_MODEL, user)
    except Exception as exc:
        print("primary model failed:", repr(exc), flush=True)
        if not FALLBACK_MODEL or time.time() - t0 > 25:
            raise
        decisions = _post(FALLBACK_MODEL, user)

    aligned = [None] * len(packages)
    for d in decisions:
        if not (isinstance(d, dict) and isinstance(d.get("index"), int) and 0 <= d["index"] < len(packages)):
            continue
        i = d["index"]
        pid = str(d.get("paragraph", "")).strip("<> ")
        para = pmaps[i].get(pid, "")
        refs = list(dict.fromkeys(REF_RE.findall(para)))
        f = d.get("facts") or {}
        cur = str(f.get("currency", "")).upper()
        amt = _amount_from_text(f.get("amountText"), cur)
        if amt is not None and amt != f.get("amountMinor"):
            print("AMOUNT FIX", f.get("amountText"), f.get("amountMinor"), "->", amt, flush=True)
            f["amountMinor"] = amt
        aligned[i] = {"action": d.get("action"), "decisive_paragraph": para, "evidenceRefs": refs[:3],
                      "facts": f, "rationale": str(d.get("analysis", ""))}
    return aligned

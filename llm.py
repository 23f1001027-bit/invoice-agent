"""Model layer: one batched call per chunk of packages. Any OpenAI-compatible API works."""
import json
import os
import re

import httpx

LLM_URL = os.environ.get("LLM_URL", "https://aipipe.org/openai/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
TOKEN = os.environ.get("AIPIPE_TOKEN", "")
CLIENT = httpx.Client(timeout=httpx.Timeout(38.0, connect=5.0))

SYSTEM = """You are an accounts-payable controller. For EACH invoice package choose exactly ONE action:

- settle_invoice: the invoice is valid, its records reconcile, and the amount is within autonomous approval authority.
- request_approval: the invoice is commercially valid, but the amount/terms are OUTSIDE delegated authority, so a named approver must sign off.
- hold_invoice: payment must PAUSE until a stated verification completes (for example a bank-detail callback, missing certificate or pending confirmation).
- reject_duplicate: the SAME commercial invoice (same vendor and same invoice number/amount) was ALREADY PAID. A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict (amounts, currency, vendor or quantities disagree between documents) and need an exception workflow.

How to read the documents:
- Documents mix useful facts with old examples, archive material, training decoys, negated statements (e.g. "no hold is required", "this is not a duplicate") and irrelevant action words. Ignore all of those.
- Find the ONE paragraph that actually determines the action for THIS invoice under the current policy. Base the decision on it, and respect negation.
- evidenceRefs: copy EXACTLY the three decisive bracketed references (the text in square brackets, e.g. [XYZ-123]) that appear in that deciding paragraph. Never use the cover-sheet reference, archive examples or training decoys. Copy them character for character, including the brackets.
- facts: vendorName and invoiceNumber exactly as written for this invoice; currency as the 3-letter code; amountMinor is the invoice amount as an INTEGER in the currency's minor unit (multiply by 100 for 2-decimal currencies such as INR, USD, EUR, GBP; JPY and KRW have no minor unit). Remove thousands separators first and double-check the arithmetic.
- rationale: one or two sentences (at least 60 characters) that name the chosen action and quote at least two of the evidence references.

Reply with JSON only, no markdown:
{"decisions":[{"index":0,"action":"...","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"..."},"evidenceRefs":["[...]","[...]","[...]"],"rationale":"..."}]}
Return one decision per package, using the package's index."""

_logged = {"n": 0}


def llm(packages):
    """packages: list of package dicts. Returns a list of raw decision dicts aligned with `packages`."""
    if _logged["n"] < 1 and packages:           # one sample in the logs helps tuning
        _logged["n"] += 1
        print("SAMPLE PACKAGE:", json.dumps(packages[0], ensure_ascii=False)[:3500], flush=True)
    user = json.dumps([{"index": i, "package": p} for i, p in enumerate(packages)], ensure_ascii=False)
    r = CLIENT.post(
        LLM_URL,
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={
            "model": LLM_MODEL,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        },
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    decisions = json.loads(re.search(r"\{.*\}", text, re.S).group(0)).get("decisions", [])
    aligned = [None] * len(packages)
    for d in decisions:
        if isinstance(d, dict) and isinstance(d.get("index"), int) and 0 <= d["index"] < len(packages):
            aligned[d["index"]] = d
    return aligned

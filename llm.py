"""Model layer: one batched call per chunk of packages. Any OpenAI-compatible API works."""
import json
import os
import re

import httpx

LLM_URL = os.environ.get("LLM_URL", "https://aipipe.org/openai/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
TOKEN = os.environ.get("AIPIPE_TOKEN", "")
CLIENT = httpx.Client(timeout=httpx.Timeout(38.0, connect=5.0))

SYSTEM = """You are an accounts-payable controller reviewing invoice packages. For EACH package choose exactly ONE action:

- settle_invoice: valid, records reconcile, and the amount is within autonomous approval authority.
- request_approval: commercially valid, but the amount or terms are OUTSIDE delegated authority, so a named approver must sign off.
- hold_invoice: payment must PAUSE until a stated verification completes (e.g. bank-detail callback, missing certificate, pending confirmation).
- reject_duplicate: the SAME commercial invoice (same vendor, same invoice number/amount) was ALREADY PAID. A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict (amounts, currency, vendor or quantities disagree between documents) and need an exception workflow.

Method (follow it for every package):
1. Identify the CONTROLLING case: the actual invoice under review under the current policy revision. Old examples, archive material, training decoys, superseded policy text and cover-sheet summaries are NOT controlling, even when they mention action words such as "settle", "hold" or "duplicate".
2. Find the ONE paragraph that decides the action for the controlling case. Respect negation ("no hold is required", "this is not a duplicate"), conditions, thresholds and dates.
3. Copy that paragraph VERBATIM into "decisive_paragraph".
4. In "analysis" (max 60 words) state the controlling facts that fix the action (amount vs limit, matching/mismatching records, payment history, pending verification).
5. Take vendorName, invoiceNumber, amountMinor and currency from the CONTROLLING invoice only (never from an archived or example invoice). amountMinor is an INTEGER in the currency's minor unit (x100 for INR, USD, EUR, GBP, AED; JPY and KRW have none). Remove thousands separators first and recheck the arithmetic.
6. evidenceRefs: exactly the three bracketed references (text in square brackets, e.g. [XYZ-123]) that appear INSIDE the decisive paragraph, copied character for character including brackets. Never use the cover-sheet reference, archive examples or decoys.
7. rationale: 2 or 3 sentences, 150 to 450 characters. Name the chosen action, cite all three evidence references verbatim, and explain how those facts lead to that action (for example: the amount exceeds the stated limit, the invoice number was already paid, a verification is still pending).

Reply with JSON only, no markdown:
{"decisions":[{"index":0,"decisive_paragraph":"...","analysis":"...","action":"...","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"..."},"evidenceRefs":["[...]","[...]","[...]"],"rationale":"..."}]}
Return exactly one decision per package, using the package's index."""

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

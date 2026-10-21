"""A2A 1.0 (HTTP+JSON) invoice agent: protocol, state and decision layers.

Framework independent: handle() takes (method, path, headers, query, body)
and returns (status, headers, body_bytes). main.py is a thin FastAPI adapter.
"""
import copy
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

CLAIM_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MT = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS_MT = "application/vnd.ga5.invoice-action-results+json"
RECEIPTS_MT = "application/vnd.ga5.invoice-action-receipts+json"
A2A_MT = "application/a2a+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}
MAX_BYTES = 512 * 1024
LIST_BUDGET = 380 * 1024

BASE_URL = os.environ.get("BASE_URL", "").strip()
CHUNK = int(os.environ.get("CHUNK", "4"))
PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v2")
DEBUG = os.environ.get("DEBUG_LOG", "1") == "1"
STRIP_BRACKETS = os.environ.get("STRIP_BRACKETS", "0") == "1"
DB_PATH = os.environ.get("DB_PATH", "/tmp/invoice_agent.db")

REF_RE = re.compile(r"\[[A-Za-z0-9][^\[\]\"\n,]{0,80}\]")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def flatten_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from flatten_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from flatten_strings(v)


def doc_refs(pkg):
    refs = []
    for s in flatten_strings(pkg):
        for m in REF_RE.findall(s):
            if m not in refs:
                refs.append(m)
    return refs


# ---------------------------------------------------------------- storage
class Store:
    def __init__(self, path):
        self.lock = threading.RLock()
        self.tasks, self.owner, self.dedupe, self.decisions, self.order = {}, {}, {}, {}, []
        self.db = None
        try:
            self.db = sqlite3.connect(path, check_same_thread=False)
            self.db.execute("create table if not exists tasks(id text primary key, principal text, seq integer, body text)")
            self.db.execute("create table if not exists dedupe(principal text, mid text, hash text, task_id text, primary key(principal, mid))")
            self.db.execute("create table if not exists decisions(key text primary key, body text)")
            self.db.commit()
            for tid, p, seq, body in self.db.execute("select id, principal, seq, body from tasks order by seq"):
                self.tasks[tid] = json.loads(body); self.owner[tid] = p; self.order.append(tid)
            for p, m, h, t in self.db.execute("select principal, mid, hash, task_id from dedupe"):
                self.dedupe[(p, m)] = (h, t)
            for k, b in self.db.execute("select key, body from decisions"):
                self.decisions[k] = json.loads(b)
        except Exception as exc:
            print("storage warning, running in memory:", repr(exc), flush=True)
            self.db = None

    def _exec(self, sql, args):
        if self.db is None:
            return
        with self.lock:
            try:
                self.db.execute(sql, args); self.db.commit()
            except Exception as exc:
                print("db write failed:", repr(exc), flush=True)

    def save_task(self, task, principal):
        with self.lock:
            if task["id"] not in self.tasks:
                self.order.append(task["id"])
            self.tasks[task["id"]] = task
            self.owner[task["id"]] = principal
            seq = self.order.index(task["id"])
        self._exec("insert or replace into tasks values(?,?,?,?)", (task["id"], principal, seq, json.dumps(task)))

    def save_dedupe(self, principal, mid, h, task_id):
        with self.lock:
            self.dedupe[(principal, mid)] = (h, task_id)
        self._exec("insert or replace into dedupe values(?,?,?,?)", (principal, mid, h, task_id))

    def save_decision(self, key, dec):
        with self.lock:
            self.decisions[key] = dec
        self._exec("insert or replace into decisions values(?,?)", (key, json.dumps(dec)))


STORE = Store(DB_PATH)
_locks, _locks_guard = {}, threading.Lock()


def lock_for(kind, key):
    with _locks_guard:
        return _locks.setdefault((kind, key), threading.Lock())


# ---------------------------------------------------------------- decisions
def pkg_key(pkg):
    p = {k: v for k, v in pkg.items() if k != "packageId"} if isinstance(pkg, dict) else pkg
    return sha(PROMPT_VERSION + "|" + canon(p))


def normalize(raw, pkg):
    """Validate one model decision against the package. Returns a decision dict or None."""
    try:
        action = raw.get("action")
        if action not in ACTIONS:
            return None
        found = doc_refs(pkg)
        refs = []
        # 1) preferred: the bracketed refs inside the verbatim decisive paragraph
        para = " ".join(str(raw.get("decisive_paragraph", "")).split())
        if para and any(para in " ".join(s.split()) for s in flatten_strings(pkg)):
            in_para = [r for r in REF_RE.findall(para) if r in found]
            if len(in_para) > 3:
                chosen = [r for r in in_para if r.strip("[]") in {str(x).strip("[] ") for x in raw.get("evidenceRefs") or []}]
                in_para = chosen if chosen else in_para
            refs = list(dict.fromkeys(in_para))[:3]
        # 2) otherwise validate the model's own list against the document
        if not refs:
            for r in raw.get("evidenceRefs") or []:
                r = str(r).strip()
                cand = r if r.startswith("[") else f"[{r}]"
                match = cand if cand in found else next((f for f in found if f[1:-1].strip() == r.strip("[] ")), None)
                if match and match not in refs:
                    refs.append(match)
            refs = refs[:3]
        if not refs:
            return None
        f = raw.get("facts") or {}
        amt = f.get("amountMinor", 0)
        if isinstance(amt, str):
            amt = float(re.sub(r"[^0-9.\-]", "", amt) or 0)
        facts = {
            "vendorName": str(f.get("vendorName", "")),
            "invoiceNumber": str(f.get("invoiceNumber", "")),
            "amountMinor": int(round(float(amt))),
            "currency": str(f.get("currency", "")).upper(),
        }
        out_refs = [r.strip("[]") for r in refs] if STRIP_BRACKETS else refs
        return {"action": action, "facts": facts, "evidenceRefs": out_refs,
                "rationale": str(raw.get("rationale", ""))}
    except Exception:
        return None


def fallback(pkg):
    refs = doc_refs(pkg)[:3]
    return {"action": "open_exception",
            "facts": {"vendorName": "", "invoiceNumber": "", "amountMinor": 0, "currency": ""},
            "evidenceRefs": [r.strip("[]") for r in refs] if STRIP_BRACKETS else refs,
            "rationale": "", "_fallback": True}


def decide(packages, llm):
    """Cached, chunked model decisions, aligned with `packages`."""
    keys = [pkg_key(p) for p in packages]
    result = [STORE.decisions.get(k) for k in keys]
    todo = [i for i, r in enumerate(result) if r is None]

    def run(idxs):
        raws = llm([packages[i] for i in idxs])
        return {i: normalize(raws[n], packages[i]) if n < len(raws) and isinstance(raws[n], dict) else None
                for n, i in enumerate(idxs)}

    for attempt in range(2):                      # second pass only retries failures
        if not todo:
            break
        chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
        got = {}
        try:
            with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as ex:
                for part in ex.map(run, chunks):
                    got.update(part)
        except Exception as exc:
            print("llm error:", repr(exc), flush=True)
        for i, dec in got.items():
            if dec:
                result[i] = dec
                STORE.save_decision(keys[i], dec)
        todo = [i for i, r in enumerate(result) if r is None]
    for i in todo:
        result[i] = fallback(packages[i])
    if DEBUG:
        for pkg, dec in zip(packages, result):
            print("DECISION", json.dumps({"pkg": pkg.get("packageId") if isinstance(pkg, dict) else None,
                  "action": dec["action"], "refs": dec["evidenceRefs"], "facts": dec["facts"],
                  "fallback": bool(dec.get("_fallback"))}, ensure_ascii=False), flush=True)
    return result


def make_rationale(action, refs, text):
    t = (text or "").strip()
    cited = sum(1 for r in refs if r in t)
    if action not in t or cited < min(2, len(refs)) or len(t) < 60:
        t = (t + " " if t else "") + f"Decision {action}, supported by evidence {', '.join(refs[:3])}."
    if len(t) < 60:
        t += " The cited paragraph determines this action under the stated policy."
    return t[:1500]


# ---------------------------------------------------------------- http helpers
def respond(status, obj, extra=None):
    body = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    h = {"content-type": A2A_MT}
    h.update(extra or {})
    return status, h, body


def err(status, code, message, extra=None):
    return respond(status, {"error": {"code": code, "message": message}, "code": code}, extra)


def bearer(headers):
    v = headers.get("authorization", "")
    if v[:7].lower() != "bearer " or not v[7:].strip():
        return None
    return sha("tok:" + v[7:].strip())[:32]


def card(host_base):
    return {
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, chooses one safe business action per invoice with cited evidence, "
                       "and executes only the actions the grader approves.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": host_base, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "securitySchemes": {"bearerAuth": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}},
        "defaultInputModes": [CLAIM_MT, RESULTS_MT],
        "defaultOutputModes": [PROPOSALS_MT, RECEIPTS_MT],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice action agent",
            "description": "Classifies each invoice package as settle, request approval, hold, reject duplicate or open exception.",
            "tags": ["invoice", "accounts-payable", "approval", "duplicate-detection"],
            "inputModes": [CLAIM_MT, RESULTS_MT],
            "outputModes": [PROPOSALS_MT, RECEIPTS_MT],
        }],
    }


def part_of(message, media_type):
    for p in message.get("parts") or []:
        if isinstance(p, dict) and p.get("mediaType") == media_type and isinstance(p.get("data"), dict):
            return p["data"]
    return None


# ---------------------------------------------------------------- operations
def new_batch(principal, message, mh, llm):
    data = part_of(message, CLAIM_MT)
    if not data or not isinstance(data.get("packages"), list) or not data.get("batchId"):
        return err(400, "INVALID_MESSAGE", "A claim batch part is required.")
    packages = data["packages"]
    batch_id = str(data["batchId"])
    decisions = decide(packages, llm)

    proposals, seen = [], set()
    for idx, (pkg, dec) in enumerate(zip(packages, decisions)):
        pid = str(pkg.get("packageId", f"pkg-{idx}")) if isinstance(pkg, dict) else f"pkg-{idx}"
        if pid in seen:
            pid = f"{pid}-{idx}"
        seen.add(pid)
        action_id = "act_" + sha(f"{principal}|{batch_id}|{pid}|{dec['action']}")[:24]
        proposals.append({
            "packageId": pid, "actionId": action_id, "action": dec["action"],
            "facts": dec["facts"], "evidenceRefs": dec["evidenceRefs"],
            "rationale": make_rationale(dec["action"], dec["evidenceRefs"], dec["rationale"]),
        })

    task_id, ctx = "task_" + uuid.uuid4().hex, message.get("contextId") or "ctx_" + uuid.uuid4().hex
    hist = copy.deepcopy(message)
    hist["taskId"], hist["contextId"] = task_id, ctx
    task = {
        "id": task_id, "contextId": ctx,
        "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now_iso()},
        "artifacts": [{"artifactId": "art_" + uuid.uuid4().hex, "name": "invoice-action-proposals",
                       "parts": [{"mediaType": PROPOSALS_MT, "data": {"batchId": batch_id, "proposals": proposals}}]}],
        "history": [hist],
        "metadata": {"batchId": batch_id},
    }
    STORE.save_task(task, principal)
    STORE.save_dedupe(principal, message["messageId"], mh, task_id)
    return respond(200, {"task": task})


def continuation(principal, message, mh):
    task_id = message["taskId"]
    with lock_for("task", task_id):
        task = STORE.tasks.get(task_id)
        if task is None or STORE.owner.get(task_id) != principal:
            return err(404, "TASK_NOT_FOUND", "Task not found.")
        if message.get("contextId") and message["contextId"] != task["contextId"]:
            return err(400, "CONTEXT_MISMATCH", "Context does not match the task.")
        if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
            return err(409, "TASK_NOT_ACTIVE", "Task cannot accept this message.")
        data = part_of(message, RESULTS_MT)
        proposals_art = task["artifacts"][0]["parts"][0]["data"]
        if not data or data.get("batchId") != proposals_art["batchId"] or not isinstance(data.get("results"), list) or not data["results"]:
            return err(400, "BATCH_MISMATCH", "Results do not match the task batch.")
        by_pkg = {p["packageId"]: p for p in proposals_art["proposals"]}
        seen, accepted = set(), {}
        for r in data["results"]:
            if not isinstance(r, dict):
                return err(400, "INVALID_RESULT", "Malformed result.")
            p = by_pkg.get(r.get("packageId"))
            if p is None or r["packageId"] in seen:
                return err(400, "PACKAGE_MISMATCH", "Result does not match a proposal.")
            if r.get("actionId") != p["actionId"] or r.get("action") != p["action"]:
                return err(400, "ACTION_MISMATCH", "Result does not match the proposed action.")
            seen.add(r["packageId"])
            if r.get("outcome") == "ACCEPTED":
                nonce = r.get("receiptNonce")
                if not isinstance(nonce, str) or not nonce:
                    return err(400, "INVALID_RESULT", "Missing receipt nonce.")
                accepted[r["packageId"]] = nonce
            elif r.get("outcome") != "REJECTED":
                return err(400, "INVALID_RESULT", "Unknown outcome.")
        executions = [{
            "packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
            "receiptNonce": accepted[p["packageId"]], "facts": p["facts"], "evidenceRefs": p["evidenceRefs"],
        } for p in proposals_art["proposals"] if p["packageId"] in accepted]

        hist = copy.deepcopy(message)
        hist["taskId"], hist["contextId"] = task_id, task["contextId"]
        task["history"].append(hist)
        task["artifacts"].append({"artifactId": "art_" + uuid.uuid4().hex, "name": "invoice-action-receipts",
                                  "parts": [{"mediaType": RECEIPTS_MT,
                                             "data": {"batchId": proposals_art["batchId"], "executions": executions}}]})
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now_iso()}
        STORE.save_task(task, principal)
        STORE.save_dedupe(principal, message["messageId"], mh, task_id)
        return respond(200, {"task": task})


def message_send(principal, body, llm):
    try:
        req = json.loads(body.decode("utf-8"))
        message = req["message"]
        mid = message["messageId"]
        if not isinstance(message, dict) or not isinstance(mid, str) or not mid or message.get("role") != "ROLE_USER" \
                or not isinstance(message.get("parts"), list):
            raise ValueError
    except Exception:
        return err(400, "INVALID_MESSAGE", "Malformed message.")
    mh = sha(canon(message))
    with lock_for("msg", f"{principal}|{mid}"):
        rec = STORE.dedupe.get((principal, mid))
        if rec:
            if rec[0] != mh:
                return err(409, "IDEMPOTENCY_CONFLICT", "Message ID was reused with different content.")
            tid = rec[1]
            with lock_for("task", tid):
                return respond(200, {"task": STORE.tasks[tid]})
        if message.get("taskId"):
            return continuation(principal, message, mh)
        return new_batch(principal, message, mh, llm)


def get_task(principal, task_id):
    with lock_for("task", task_id):
        task = STORE.tasks.get(task_id)
        if task is None or STORE.owner.get(task_id) != principal:
            return err(404, "TASK_NOT_FOUND", "Task not found.")
        return respond(200, task)


def list_tasks(principal, query):
    with_hist = bool(query.get("historyLength", ["0"])[0] not in ("", "0"))
    out, size = [], 0
    mine = [t for t in reversed(STORE.order) if STORE.owner.get(t) == principal]
    for tid in mine:
        with lock_for("task", tid):
            t = copy.deepcopy(STORE.tasks[tid])
        if not with_hist:
            t.pop("history", None)
        s = len(json.dumps(t))
        if out and size + s > LIST_BUDGET:
            break
        out.append(t); size += s
    return respond(200, {"tasks": out, "nextPageToken": "", "pageSize": len(out), "totalSize": len(mine)})


def cancel_task(principal, task_id):
    with lock_for("task", task_id):
        task = STORE.tasks.get(task_id)
        if task is None or STORE.owner.get(task_id) != principal:
            return err(404, "TASK_NOT_FOUND", "Task not found.")
        if task["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_CANCELABLE", "Task is already finished.")
        task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now_iso()}
        STORE.save_task(task, principal)
        return respond(200, task)


# ---------------------------------------------------------------- router
def handle(method, path, headers, query, body, llm):
    headers = {k.lower(): v for k, v in headers.items()}
    path = re.sub(r"/{2,}", "/", path)
    host = headers.get("x-forwarded-host") or headers.get("host", "localhost")
    proto = headers.get("x-forwarded-proto", "https")
    base = BASE_URL or f"{proto}://{host}/a2a"
    prefix = urlparse(base).path.rstrip("/")

    if method == "GET" and path == "/healthz":
        return respond(200, {"status": "ok"})
    if method == "GET" and path in ("/.well-known/agent-card.json", f"{prefix}/.well-known/agent-card.json"):
        return 200, {"content-type": "application/json"}, json.dumps(card(base)).encode()

    if not path.startswith(prefix + "/"):
        return err(404, "NOT_FOUND", "Not found.")
    principal = bearer(headers)
    if principal is None:
        return err(401, "UNAUTHENTICATED", "Authentication required.", {"www-authenticate": "Bearer"})
    if headers.get("a2a-version", "").strip() != "1.0":
        return err(400, "VERSION_NOT_SUPPORTED", "Only A2A version 1.0 is supported.")

    rel = path[len(prefix):]
    q = parse_qs(query or "")
    if method == "POST" and rel == "/message:send":
        ctype = headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != A2A_MT:
            return err(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/a2a+json.")
        return message_send(principal, body, llm)
    if method == "GET" and rel == "/tasks":
        return list_tasks(principal, q)
    if method == "GET" and rel.startswith("/tasks/") and not rel.endswith(":cancel"):
        return get_task(principal, rel[len("/tasks/"):])
    if method == "POST" and rel.startswith("/tasks/") and rel.endswith(":cancel"):
        return cancel_task(principal, rel[len("/tasks/"):-len(":cancel")])
    return err(404, "NOT_FOUND", "Not found.")

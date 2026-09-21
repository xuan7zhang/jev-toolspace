"""Minimal client for TypeSafe's System One API, verified against https://docs.typesafe.ai/api and one live call (2026-09-21):
    POST https://api.typesafe.ai/v1/systemone   Authorization: Bearer <key>
    body   {"model": "jev-latest", "state": <str|object|array>, "questions": {qid: {"type": "noul", "instructions": ...}}}
    reply  {"model": "jev-1.13.0", "answers": {qid: {"type": "noul", "noul": p}}, "usage": {"input_tokens", "output_tokens"}}
A noul question returns an independent probability in [0, 1]; every question is answered against the same state in one
parallel pass, so one request with one noul per tool gives per-tool relevance judgments that are not a softmax over tools.
Limits (docs.typesafe.ai/models): 64k tokens per request, 32k for state + longest question, 1,200 requests/min, 250k tok/s.
Price: $0.042 per 1M input tokens, output free.  Errors: 401 key, 422 validation, 429 rate limit, 529 overload.
Key from $JEV_API_KEY (or $TYPESAFE_API_KEY).  JEV_MOCK=1 returns deterministic fake answers and never calls the network.
"""
import hashlib, json, os, random, threading, time, urllib.error, urllib.request
URL = "https://api.typesafe.ai/v1/systemone"; PRICE_PER_M_INPUT = 0.042

class JevClient:
    def __init__(self, model="jev-latest", cache=None, rpm=600, retries=8, timeout=60):
        self.model, self.rpm, self.retries, self.timeout = model, rpm, retries, timeout
        self.mock = os.environ.get("JEV_MOCK") == "1"
        self.key = os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
        if not self.mock and not self.key: raise RuntimeError("set JEV_API_KEY (source ~/.jev_api_env) or JEV_MOCK=1")
        self.cache_path = cache; self.cache = {}; self.lock = threading.Lock(); self.last = 0.0
        if cache and os.path.exists(cache):
            for l in open(cache):
                try: r = json.loads(l); self.cache[r["key"]] = r
                except Exception: pass

    @staticmethod
    def _key(model, state, questions):
        return hashlib.sha256(json.dumps([model, state, questions], sort_keys=True).encode()).hexdigest()

    def _throttle(self):
        with self.lock:
            wait = self.last + 60.0 / self.rpm - time.time()
            if wait > 0: time.sleep(wait)
            self.last = time.time()

    def ask(self, state, questions, tag=None):
        """-> dict(answers={qid: p}, model, usage, latency_s, cached, key).  Raises after `retries` failed attempts."""
        k = self._key(self.model, state, questions)
        if k in self.cache: return dict(self.cache[k], cached=True)
        if self.mock:
            rng = random.Random(k); ans = {q: round(rng.random(), 3) for q in questions}
            rec = dict(key=k, answers=ans, model="MOCK", usage={"input_tokens": 0, "output_tokens": 0}, latency_s=0.0, tag=tag, mock=True)
        else:
            body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode(); err = None
            for attempt in range(self.retries):
                self._throttle(); t0 = time.time()
                try:
                    req = urllib.request.Request(URL, data=body, method="POST", headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                    out = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read()); lat = time.time() - t0
                    ans = {q: a.get("noul") for q, a in out["answers"].items()}
                    rec = dict(key=k, answers=ans, raw=out["answers"], model=out.get("model"), usage=out.get("usage", {}), latency_s=round(lat, 4), tag=tag, mock=False); break
                except urllib.error.HTTPError as e:
                    err = f"HTTP {e.code}: {e.read()[:300]!r}"
                    if e.code in (400, 401, 403, 422): raise RuntimeError(err)
                except Exception as e: err = repr(e)
                time.sleep(min(60, 2 ** attempt + random.random()))
            else: raise RuntimeError(f"Jev call failed after {self.retries} attempts: {err}")
        with self.lock:
            self.cache[k] = rec
            if self.cache_path:
                with open(self.cache_path, "a") as f: f.write(json.dumps(rec) + "\n")
        return dict(rec, cached=False)

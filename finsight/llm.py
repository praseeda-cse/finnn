"""Client for any OpenAI-compatible chat server, so the model can be swapped without code changes:
Ollama, vLLM or llama.cpp's llama-server running an open model on a free Kaggle/Colab GPU, or a
free hosted endpoint. Requests run in parallel; every reply is appended to a JSONL file as soon as
it arrives, so a run resumes where it stopped after a disconnect."""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

THINK = re.compile(r"<think>.*?</think>", re.S)


class ChatClient:
    def __init__(self, base_url, model, api_key=None, workers=4, timeout=900, retries=8):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.workers, self.timeout, self.retries = model, workers, timeout, retries
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def chat(self, messages, temperature=0.0, max_tokens=512, seed=None):
        """One reply. Reasoning models' <think> sections are removed; rate limits and server errors
        are retried with back-off."""
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            payload["seed"] = seed
        for attempt in range(self.retries):
            try:
                r = requests.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(float(r.headers.get("retry-after") or min(60, 2 ** attempt)))
                    continue
                r.raise_for_status()
                text = r.json()["choices"][0]["message"].get("content") or ""
                return THINK.sub("", text).strip()
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"no reply from {self.url} after {self.retries} tries")

    def run(self, jobs, out_path, n=1, temperature=0.0, max_tokens=512):
        """jobs: {uid: messages}. n=1: rows {"uid", "raw"}; n>1: rows {"uid", "samples": [...]}, the
        samples drawn with seeds 0..n-1. Questions already in out_path are skipped. Returns {uid: row}."""
        done = {}
        if os.path.exists(out_path):
            for line in open(out_path, encoding="utf-8"):
                row = json.loads(line)
                done[row["uid"]] = row
        todo = [uid for uid in jobs if uid not in done]
        lock = threading.Lock()

        def one(uid):
            if n == 1:
                return {"uid": uid, "raw": self.chat(jobs[uid], temperature, max_tokens)}
            return {"uid": uid, "samples": [self.chat(jobs[uid], temperature, max_tokens, seed=i) for i in range(n)]}

        with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(self.workers) as pool:
            futures = [pool.submit(one, uid) for uid in todo]
            for fut in as_completed(futures):
                row = fut.result()
                with lock:
                    done[row["uid"]] = row
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"\r{len(done)}/{len(jobs)} done", end="")
        print()
        return done

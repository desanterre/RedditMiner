
# reddit_miner/services/reddit_sentinel/reddit_sentinel_optimized.py

import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from colorama import init, Fore, Style
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict

from gpt4all import GPT4All

init(autoreset=True)

# --------------------------
# Worker-side globals
# --------------------------
_WORKER_MODEL = None

def _init_worker_model(model_path: str):
    """
    Called once per worker process. Loads a dedicated GPT4All model per process.
    """
    global _WORKER_MODEL
    _WORKER_MODEL = GPT4All(model_path, allow_download=False)

def _build_batch_prompt(pairs: List[Tuple[int, str]]) -> str:
    """
    Build a strict JSON-output prompt for a batch of comments.
    """
    items = []
    for idx, body in pairs:
        body = "" if pd.isna(body) else str(body)
        items.append({"index": idx, "comment": body})

    prompt = (
        "You are a strict content moderation classifier.\n"
        "For each item, classify the 'comment' as exactly one of: EXTREMIST, TOXIC, SAFE.\n"
        "Return ONLY a compact JSON object with the following schema:\n"
        "{\"results\": [{\"index\": <int>, \"label\": \"EXTREMIST|TOXIC|SAFE\"}, ...]}\n"
        "Do NOT add explanations, prefixes, or any text outside JSON.\n\n"
        f"Items:\n{json.dumps(items, ensure_ascii=False)}\n"
    )
    return prompt

def _parse_json_results(raw: str) -> Dict[int, str]:
    """
    Parse model output into {index: label}. Includes a best-effort fallback.
    """
    raw = str(raw).strip()

    # Try strict JSON
    try:
        obj = json.loads(raw)
        results = obj.get("results", [])
        out = {}
        for r in results:
            idx = int(r["index"])
            label = str(r["label"]).upper().strip()
            if label not in ("EXTREMIST", "TOXIC", "SAFE"):
                label = "SAFE"
            out[idx] = label
        return out
    except Exception:
        pass

    # Fallback: regex extraction
    import re
    out = {}
    pattern = re.compile(r'"index"\s*:\s*(\d+)\s*,\s*"label"\s*:\s*"(EXTREMIST|TOXIC|SAFE)"', re.IGNORECASE)
    for m in pattern.finditer(raw):
        idx = int(m.group(1))
        label = m.group(2).upper()
        out[idx] = label
    return out

def _classify_batch_worker(args: Tuple[str, List[Tuple[int, str]], int, float]) -> Dict[int, str]:
    """
    Worker classifying a single batch.
    args:
      - model_path: str
      - pairs: List[(index, body)]
      - max_tokens: int
      - temperature: float
    """
    global _WORKER_MODEL
    model_path, pairs, max_tokens, temperature = args

    if _WORKER_MODEL is None:
        _init_worker_model(model_path)

    prompt = _build_batch_prompt(pairs)

    try:
        raw = _WORKER_MODEL.generate(
            prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_k=40,
            top_p=0.95,
            repeat_penalty=1.1,
        )
        return _parse_json_results(raw)
    except Exception:
        return {idx: "SAFE" for idx, _ in pairs}


class RedditSentinel:
    def __init__(
        self,
        model_path: str = None,
        database_url: str = None,
        batch_size: int = 32,
        max_workers: int = 2,
        temperature: float = 0.0,
        prefilter_min_len: int = 10,
    ):
        self.database_url = database_url or os.environ.get(
            "DATABASE_URL",
            "postgresql://reddit_user:reddit_pass@localhost:5432/reddit_db"
        )
        self.model_path = model_path or "models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"

        self.batch_size = max(1, int(batch_size))
        self.max_workers = max(1, int(max_workers))
        self.temperature = float(temperature)
        self.prefilter_min_len = int(prefilter_min_len)

        self.df_comments = pd.DataFrame()

        # Logging helpers
        self.log_info = lambda msg: print(Fore.CYAN + "[INFO] " + Style.BRIGHT + str(msg))
        self.log_success = lambda msg: print(Fore.GREEN + "[OK] " + Style.BRIGHT + str(msg))
        self.log_warn = lambda msg: print(Fore.YELLOW + "[WARN] " + Style.BRIGHT + str(msg))
        self.log_error = lambda msg: print(Fore.RED + "[ERROR] " + Style.BRIGHT + str(msg))

    # --------------------------
    # Compatibility: load_model()
    # --------------------------
    def load_model(self):
        """
        Compatibility method to satisfy legacy CLI calls.
        In the optimized version, models are loaded per worker process.
        This method only verifies the model file exists and logs accordingly.
        """
        if not os.path.exists(self.model_path):
            self.log_error(f"Model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        self.log_info("Model path verified. Models will be loaded per worker process.")
        self.log_success("Compatibility: load_model() completed (no-op).")

    # --------------------------
    # Load comments from DB
    # --------------------------
    def load_comments_from_db(self):
        self.log_info("Loading comments from DB...")
        engine = create_engine(self.database_url)
        with engine.connect() as conn:
            query = text("SELECT author, subreddit, body, score, date_utc, permalink FROM comments;")
            self.df_comments = pd.read_sql(query, conn)
        self.log_success(f"Loaded {len(self.df_comments)} comments")
        if self.df_comments.empty:
            self.log_warn("No comments found in DB. Exiting.")
            return

        # Minimal normalization
        if "body" not in self.df_comments.columns:
            self.log_error("Column 'body' missing in comments table.")
            self.df_comments["body"] = ""
        self.df_comments["body"] = self.df_comments["body"].astype(str).fillna("")

    # --------------------------
    # Classify comments (batched + multiprocess)
    # --------------------------
    def classify_comments(self):
        if self.df_comments.empty:
            self.log_warn("No comments loaded. Cannot classify.")
            return

        self.log_info("Preparing classification...")
        self.df_comments["ai_classification"] = ""

        # Pre-filter: mark short comments as SAFE directly
        needs_llm_mask = self.df_comments["body"].str.len() >= self.prefilter_min_len
        self.df_comments.loc[~needs_llm_mask, "ai_classification"] = "SAFE"

        indices_to_classify = self.df_comments.index[needs_llm_mask].tolist()
        if not indices_to_classify:
            self.log_success("Nothing to classify; all comments marked SAFE by pre-filter.")
            return

        pairs = [(int(idx), self.df_comments.at[idx, "body"]) for idx in indices_to_classify]

        batches: List[List[Tuple[int, str]]] = [
            pairs[i:i + self.batch_size] for i in range(0, len(pairs), self.batch_size)
        ]
        self.log_info(f"Total to classify via LLM: {len(pairs)} | batches: {len(batches)} | batch_size={self.batch_size}")

        # Token budget estimation per batch (compact JSON)
        base_tokens_per_item = 14
        def tokens_for_batch(n): return max(64, int(n * base_tokens_per_item + 64))

        extremist_count = 0

        self.log_info(f"Spawning {self.max_workers} worker process(es) with individual GPT4All models...")
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_init_worker_model,
            initargs=(self.model_path,)
        ) as executor:
            futures = []
            for batch in batches:
                max_toks = tokens_for_batch(len(batch))
                futures.append(
                    executor.submit(
                        _classify_batch_worker,
                        (self.model_path, batch, max_toks, self.temperature)
                    )
                )

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Classifying (batches)", ncols=100):
                try:
                    mapping = fut.result()  # {index: label}
                except Exception as e:
                    self.log_error(f"Batch classification error: {e}")
                    mapping = {}

                for idx, label in mapping.items():
                    if label not in ("EXTREMIST", "TOXIC", "SAFE"):
                        label = "SAFE"
                    self.df_comments.at[idx, "ai_classification"] = label
                    if label == "EXTREMIST":
                        extremist_count += 1

        self.log_success(f"Classification done. Found {extremist_count} extremist comments.")

    # --------------------------
    # Export flagged comments to Excel
    # --------------------------
    def export_flagged_comments(self, output="flagged_comments.xlsx"):
        extremist_comments = self.df_comments[self.df_comments["ai_classification"] == "EXTREMIST"]
        self.log_info(f"Exporting extremist comments to {output}...")
        extremist_comments.to_excel(output, index=False, engine="openpyxl")
        self.log_success(f"Export completed: {len(extremist_comments)} comments saved to {output}")


if __name__ == "__main__":
    sentinel = RedditSentinel(
        model_path="models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf",
        database_url=os.environ.get(
            "DATABASE_URL",
            "postgresql://reddit_user:reddit_pass@localhost:5432/reddit_db"
        ),
        batch_size=32,
        max_workers=2,
        temperature=0.0,
        prefilter_min_len=10
    )

    sentinel.load_comments_from_db()
    sentinel.load_model()  # compatibility no-op
    sentinel.classify_comments()
    sentinel.export_flagged_comments("flagged_comments.xlsx")

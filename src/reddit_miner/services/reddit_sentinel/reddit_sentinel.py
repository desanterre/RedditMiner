
# reddit_miner/services/reddit_sentinel/reddit_sentinel.py
# Complete sentinel with: complexity-aware human-moderator prompt, signals extraction,
# confidence gating, prompt budgeting, batched classification, and incremental DB persistence.

import os
import json
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError
from colorama import init, Fore, Style
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from gpt4all import GPT4All

init(autoreset=True)

# --------------------------
# Prompt budgeting constants
# --------------------------
# If your runtime reports ~2048 tokens, use a margin to avoid overflow.
MAX_CONTEXT_TOKENS = 2000
TOKENS_PER_CHAR = 1.0 / 4.0           # heuristic: ~4 chars ≈ 1 token
MAX_PROMPT_CHARS = int(MAX_CONTEXT_TOKENS / TOKENS_PER_CHAR * 0.85)  # ~85% of window
PER_ITEM_CHAR_LIMIT = 800             # cap each comment to 800 chars

# --------------------------
# Label confidence thresholds (SAFE by default)
# --------------------------
EXTREMIST_THRESHOLD = 0.85
TOXIC_THRESHOLD     = 0.70

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


def _truncate_body(text: str, limit: int = PER_ITEM_CHAR_LIMIT) -> str:
    """
    Truncate a comment to fit per-item character limits.
    """
    if not text:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit]


def _build_batch_prompt(pairs: List[Tuple[int, str]]) -> str:
    """
    Complexity-aware, human-moderator-style prompt.
    - Truncates each comment upstream (PER_ITEM_CHAR_LIMIT).
    - Encourages analysis-first (signals) then decision-second.
    - Uses short rule codes to reduce tokens while preserving traceability.
    Output schema:
      {"results":[
        {"index":<int>,
         "label":"EXTREMIST|TOXIC|SAFE",
         "confidence":0.0-1.0,
         "explanation":"<=22 words",
         "reasons":["R?","D?","E?","T?","C?","Q?"],
         "signals":{
           "stance":"endorse|condemn|report|quote|analyze|satire|unclear",
           "violence_call":true|false,
           "endorsement":true|false,
           "recruitment":true|false,
           "target_present":true|false,
           "target_type":"person|group|protected|org|none",
           "threat":true|false,
           "slur":true|false,
           "dehumanization":true|false,
           "dogwhistle":true|false,
           "quoted":true|false,
           "negation":true|false,
           "reporting":true|false,
           "hypothetical":true|false,
           "satire":true|false,
           "ambiguity":true|false
         }}
      ]}
    """
    items = []
    for idx, body in pairs:
        body = "" if pd.isna(body) else str(body)
        body = _truncate_body(body, PER_ITEM_CHAR_LIMIT)
        items.append({"index": idx, "comment": body})

    rubric = (
        "You are an experienced human-like moderator. Perform analysis before decision.\n"
        "Return ONLY JSON exactly matching:\n"
        "{\"results\":[{\"index\":<int>,\"label\":\"EXTREMIST|TOXIC|SAFE\",\"confidence\":0.0-1.0,"
        "\"explanation\":\"<=22 words\",\"reasons\":[\"R?\",\"D?\",\"E?\",\"T?\",\"C?\",\"Q?\"],"
        "\"signals\":{\"stance\":\"endorse|condemn|report|quote|analyze|satire|unclear\",\"violence_call\":true|false,"
        "\"endorsement\":true|false,\"recruitment\":true|false,\"target_present\":true|false,"
        "\"target_type\":\"person|group|protected|org|none\",\"threat\":true|false,\"slur\":true|false,"
        "\"dehumanization\":true|false,\"dogwhistle\":true|false,\"quoted\":true|false,\"negation\":true|false,"
        "\"reporting\":true|false,\"hypothetical\":true|false,\"satire\":true|false,\"ambiguity\":true|false}}]}\n\n"
        "Definitions (AUTHOR-centric):\n"
        "R1 EXTREMIST: Author endorses/praises/advocates/recruits for violent extremism OR calls for violence against persons/groups OR praises recognized extremist orgs.\n"
        "R2 TOXIC: Author uses targeted insults/slurs/demeaning language/threats/harassment aimed at a person/group. Ambient profanity without a target is SAFE.\n"
        "R3 SAFE: Reporting/analysis/criticism/condemnation/quotation/satire/hypotheticals/ambiguous intent without endorsement or targeted abuse.\n"
        "Context rules:\n"
        "C1 Quoted text without explicit author endorsement -> SAFE.\n"
        "C2 Reporting/linking/news-style without approval -> SAFE.\n"
        "C3 Negation/condemnation cancels endorsement -> SAFE.\n"
        "C4 Satire/irony: If ambiguous -> SAFE.\n"
        "C5 Hypotheticals without advocacy -> SAFE.\n"
        "C6 Ambiguity -> SAFE, lower confidence.\n"
        "Decision steps:\n"
        "D1 Violence intent: Does the AUTHOR call for or endorse violence? If yes -> EXTREMIST (high conf). Else -> D2.\n"
        "D2 Targeted abuse: Direct insults/slurs/threats/harassment at a target? If yes -> TOXIC (moderate-high conf). Else -> SAFE.\n"
        "D3 Target required for TOXIC; author endorsement required for EXTREMIST.\n"
        "D4 Multilingual: Apply same rubric across languages.\n"
        "Tie-breakers:\n"
        "T1 If stance=quote/report/analyze and no endorsement -> SAFE.\n"
        "T2 If negation or condemnation present -> SAFE.\n"
        "T3 If uncertainty or mixed cues -> choose SAFE.\n"
        "Output constraints:\n"
        "- Use labels EXACTLY: EXTREMIST, TOXIC, SAFE.\n"
        "- confidence in [0.0,1.0].\n"
        "- explanation neutral, <=22 words, no policy text, no chain-of-thought.\n"
        "- reasons is a short code list (e.g., [\"R1\",\"D1\",\"C2\",\"T3\"]).\n"
        "- signals must be booleans/enum exactly as specified.\n\n"
        "Items:\n"
    )

    prompt = rubric + json.dumps(items, ensure_ascii=False)
    return prompt


def _parse_json_results(raw: str) -> Dict[int, Dict[str, Any]]:
    """
    Parse model output into:
      {index: {"label": str, "confidence": float, "explanation": str,
               "reasons": List[str], "signals": Dict[str, Any]}}
    Best-effort fallback if JSON is imperfect.
    """
    raw = str(raw).strip()

    # Strict JSON first
    try:
        obj = json.loads(raw)
        results = obj.get("results", [])
        out: Dict[int, Dict[str, Any]] = {}
        for r in results:
            idx = int(r["index"])
            label = str(r.get("label", "SAFE")).upper().strip()
            conf = float(r.get("confidence", 0.0))
            expl = str(r.get("explanation", "")).strip()
            reasons = r.get("reasons", [])
            signals = r.get("signals", {})

            if label not in ("EXTREMIST", "TOXIC", "SAFE"):
                label, conf = "SAFE", 0.0
            conf = max(0.0, min(1.0, conf))
            if len(expl) > 160:  # keep short
                expl = expl[:160]
            if not isinstance(reasons, list):
                reasons = []
            if not isinstance(signals, dict):
                signals = {}

            out[idx] = {
                "label": label,
                "confidence": conf,
                "explanation": expl,
                "reasons": reasons,
                "signals": signals
            }
        return out
    except Exception:
        pass

    # Minimal fallback (regex) if needed
    import re
    out: Dict[int, Dict[str, Any]] = {}
    pat = re.compile(r'"index"\s*:\s*(\d+)\s*,\s*"label"\s*:\s*"(EXTREMIST|TOXIC|SAFE)"', re.IGNORECASE)
    for m in pat.finditer(raw):
        idx = int(m.group(1))
        label = m.group(2).upper()
        out[idx] = {"label": label, "confidence": 0.0, "explanation": "", "reasons": [], "signals": {}}
    return out


def _rebatch_to_prompt_budget(pairs: List[Tuple[int, str]]) -> List[List[Tuple[int, str]]]:
    """
    Split pairs into batches that respect MAX_PROMPT_CHARS budget to avoid context overflow.
    """
    batches: List[List[Tuple[int, str]]] = []
    current: List[Tuple[int, str]] = []
    # Larger header estimate due to complex rubric and schema definition
    HEADER_CHARS = 1600  # estimate for rubric + instructions + "Items:\n"
    approx_chars = HEADER_CHARS

    def item_cost(idx: int, body: str) -> int:
        # Rough cost for JSONified item: {"index": <num>, "comment": "<text>"}
        # body already truncated.
        return 48 + len(body)

    for idx, body in pairs:
        body_trunc = _truncate_body(body, PER_ITEM_CHAR_LIMIT)
        cost = item_cost(idx, body_trunc)
        if current and (approx_chars + cost) > MAX_PROMPT_CHARS:
            batches.append(current)
            current = []
            approx_chars = HEADER_CHARS
        current.append((idx, body_trunc))
        approx_chars += cost

    if current:
        batches.append(current)

    return batches


def _classify_batch_worker(args: Tuple[str, List[Tuple[int, str]], int, float]) -> Dict[int, Dict[str, Any]]:
    """
    Worker classifying a single batch.
    args:
      - model_path: str
      - pairs: List[(index, body)]
      - max_tokens: int
      - temperature: float
    Returns:
      {index: {"label": "EXTREMIST|TOXIC|SAFE", "confidence": float, "explanation": "...",
               "reasons": [...], "signals": {...}}}
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
        # On error, mark everything SAFE with low confidence
        return {idx: {"label": "SAFE", "confidence": 0.0, "explanation": "", "reasons": [], "signals": {}} for idx, _ in pairs}


def _finalize_label(label_raw: str, conf: float, signals: Dict[str, Any]) -> str:
    """
    Apply SAFE-by-default gating with signal checks.
    """
    stance = signals.get("stance", "unclear")
    endorsement = bool(signals.get("endorsement", False))
    violence_call = bool(signals.get("violence_call", False))
    recruitment = bool(signals.get("recruitment", False))
    target_present = bool(signals.get("target_present", False))
    quoted = bool(signals.get("quoted", False))
    reporting = bool(signals.get("reporting", False))
    negation = bool(signals.get("negation", False))
    ambiguity = bool(signals.get("ambiguity", False))
    slur = bool(signals.get("slur", False))
    threat = bool(signals.get("threat", False))
    dehum = bool(signals.get("dehumanization", False))

    # Hard SAFE conditions
    if negation or reporting or (quoted and not endorsement):
        return "SAFE"
    if ambiguity and conf < 0.80:
        return "SAFE"
    if stance in ("report", "quote", "analyze") and not endorsement:
        return "SAFE"

    # Extremist requires strong signals
    if (label_raw == "EXTREMIST" and conf >= EXTREMIST_THRESHOLD) and (
        violence_call or recruitment or endorsement
    ):
        return "EXTREMIST"

    # Toxic requires target and abuse signals
    if (label_raw == "TOXIC" and conf >= TOXIC_THRESHOLD) and (
        target_present and (slur or threat or dehum)
    ):
        return "TOXIC"

    # Otherwise SAFE
    return "SAFE"


class RedditSentinel:
    def __init__(
        self,
        model_path: Optional[str] = None,
        database_url: Optional[str] = None,
        batch_size: int = 16,             # safer default when adding explanations
        max_workers: int = 1,             # start with 1 on laptops; increase only if RAM allows
        temperature: float = 0.0,
        prefilter_min_len: int = 10,      # short comments → SAFE directly
        warmup_in_main: bool = True,      # quick warmup classification in main process
        persist_warmup: bool = True,      # persist warmup results to DB
        force_sequential: bool = False,   # run without multiprocessing if your machine struggles
        limit: Optional[int] = None       # optional LIMIT when loading from DB
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
        self.warmup_in_main = bool(warmup_in_main)
        self.persist_warmup = bool(persist_warmup)
        self.force_sequential = bool(force_sequential)
        self.limit = limit

        self.df_comments = pd.DataFrame()
        self.key_column: Optional[str] = None  # 'id' preferred; fallback to 'permalink'

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
    # Helpers
    # --------------------------
    def _model_name(self) -> str:
        return os.path.basename(self.model_path)

    def _choose_key_column(self, engine) -> str:
        """
        Prefer 'id' if present, else fallback to 'permalink'.
        Raises if neither exists.
        """
        insp = inspect(engine)
        cols = [c["name"] for c in insp.get_columns("comments")]
        if "id" in cols:
            return "id"
        if "permalink" in cols:
            return "permalink"
        raise RuntimeError("Table 'comments' must have either 'id' or 'permalink' to update rows.")

    # --------------------------
    # Load comments from DB (only unmoderated)
    # --------------------------
    def load_comments_from_db(self):
        self.log_info("Loading unmoderated comments from DB...")
        engine = create_engine(self.database_url)

        # Determine key column dynamically
        self.key_column = self._choose_key_column(engine)

        # Build SELECT statement dynamically to include key column
        base_sql = f"""
            SELECT {self.key_column} AS key_col,
                   author, subreddit, body, score, date_utc, permalink
            FROM comments
            WHERE ai_label IS NULL
        """
        if self.limit:
            base_sql += " LIMIT :limit"
            query = text(base_sql).bindparams(limit=self.limit)
        else:
            query = text(base_sql)

        with engine.connect() as conn:
            self.df_comments = pd.read_sql(query, conn)

        self.log_success(f"Loaded {len(self.df_comments)} unmoderated comments")
        if self.df_comments.empty:
            self.log_warn("Nothing to classify; exiting.")
            return

        if "body" not in self.df_comments.columns:
            self.log_error("Column 'body' missing in comments table.")
            self.df_comments["body"] = ""

        # Normalize types
        self.df_comments["body"] = self.df_comments["body"].astype(str).fillna("")
        if "key_col" not in self.df_comments.columns:
            raise RuntimeError("Missing 'key_col' in loaded data.")

    # --------------------------
    # DB persistence
    # --------------------------
    def _persist_batch_results(self, conn, batch_rows: List[Dict[str, Any]]) -> int:
        """
        Persist a batch classification to DB using executemany.
        Returns number of rows attempted to persist.
        batch_rows elements:
          {
            "key_value": <id or permalink>,
            "ai_label": <str>,
            "ai_explanation": <str>,
            "ai_model": <str>,
          }
        """
        assert self.key_column is not None, "key_column must be set before persisting"

        sql = text(f"""
            UPDATE comments
            SET
                ai_label = :ai_label,
                ai_explanation = :ai_explanation,
                ai_model = :ai_model,
                ai_moderated_at = NOW()
            WHERE {self.key_column} = :key_value
        """)

        try:
            conn.execute(sql, batch_rows)  # executemany with list of dicts
            return len(batch_rows)
        except OperationalError as e:
            self.log_warn(f"DB operational error on batch; retrying once: {e}")
            conn.execute(sql, batch_rows)
            return len(batch_rows)

    def persist_prefilter_safe(self) -> int:
        """
        Persist pre-filtered SAFE rows (short comments) to DB so they won't be fetched next time.
        Returns number of rows persisted.
        """
        if self.df_comments.empty:
            return 0
        engine = create_engine(self.database_url)
        model_name = self._model_name()
        mask = (self.df_comments["ai_classification"] == "SAFE") & (self.df_comments["body"].str.len() < self.prefilter_min_len)
        if not mask.any():
            return 0

        rows = []
        for _, row in self.df_comments.loc[mask, :].iterrows():
            rows.append({
                "key_value": row["key_col"],
                "ai_label": "SAFE",
                "ai_explanation": "Short/neutral comment pre-filtered.",
                "ai_model": model_name
            })

        with engine.connect() as conn:
            trans = conn.begin()
            try:
                count = self._persist_batch_results(conn, rows)
                trans.commit()
                return count
            except Exception as e:
                trans.rollback()
                self.log_error(f"Prefilter SAFE commit failed, rolled back: {e}")
                return 0

    # --------------------------
    # Warmup (optional) — persist results too
    # --------------------------
    def _main_process_warmup(self, sample_pairs: List[Tuple[int, str]]):
        """
        Load a temporary model in the main process and classify a tiny sample to confirm
        generation works; optionally persist immediately.
        """
        self.log_info("Warmup: loading model in main process for a tiny batch...")
        try:
            model = GPT4All(self.model_path, allow_download=False)
            prompt = _build_batch_prompt(sample_pairs)
            raw = model.generate(prompt, max_tokens=128, temp=0.0, top_k=40, top_p=0.95, repeat_penalty=1.1)
            mapping = _parse_json_results(raw)

            to_update = []
            model_name = self._model_name()
            for idx, result in (mapping or {}).items():
                # Threshold-gated final label using signals
                final_label = _finalize_label(
                    result.get("label", "SAFE"),
                    float(result.get("confidence", 0.0)),
                    result.get("signals", {})
                )
                expl = result.get("explanation", "")

                self.df_comments.at[idx, "ai_classification"] = final_label
                key_value = self.df_comments.at[idx, "key_col"]
                to_update.append({
                    "key_value": key_value,
                    "ai_label": final_label,
                    "ai_explanation": expl,
                    "ai_model": model_name
                })

            if self.persist_warmup and to_update:
                engine = create_engine(self.database_url)
                with engine.connect() as conn:
                    trans = conn.begin()
                    try:
                        count = self._persist_batch_results(conn, to_update)
                        trans.commit()
                        self.log_success(f"Warmup persisted {count} row(s).")
                    except Exception as e:
                        trans.rollback()
                        self.log_error(f"Warmup DB commit failed, rolled back: {e}")

            self.log_success("Warmup succeeded (main process).")
        except Exception as e:
            self.log_warn(f"Warmup failed or timed out: {e}")
        finally:
            try:
                del model
            except Exception:
                pass

    # --------------------------
    # Sequential classification (no multiprocessing)
    # --------------------------
    def _classify_batched_sequential(self, batches: List[List[Tuple[int, str]]], tokens_for_batch) -> int:
        """
        Sequential classification path: no multiprocessing.
        Persists results to DB per batch with commit per batch.
        """
        extremist_count = 0
        model_name = self._model_name()
        engine = create_engine(self.database_url)

        self.log_info("Running sequential classification (no multiprocessing)...")
        with engine.connect() as conn:
            for i, batch in enumerate(tqdm(batches, total=len(batches), desc="Classifying (sequential)", ncols=100), start=1):
                try:
                    temp_model = GPT4All(self.model_path, allow_download=False)
                    prompt = _build_batch_prompt(batch)
                    raw = temp_model.generate(
                        prompt,
                        max_tokens=tokens_for_batch(len(batch)),
                        temp=self.temperature,
                        top_k=40,
                        top_p=0.95,
                        repeat_penalty=1.1,
                    )
                    mapping = _parse_json_results(raw)
                except Exception as e:
                    self.log_error(f"Sequential batch error: {e}")
                    mapping = {idx: {"label": "SAFE", "confidence": 0.0, "explanation": "", "reasons": [], "signals": {}} for idx, _ in batch}
                finally:
                    try:
                        del temp_model
                    except Exception:
                        pass

                to_update = []
                for idx, result in (mapping or {}).items():
                    final_label = _finalize_label(
                        result.get("label", "SAFE"),
                        float(result.get("confidence", 0.0)),
                        result.get("signals", {})
                    )
                    expl = result.get("explanation", "")

                    self.df_comments.at[idx, "ai_classification"] = final_label
                    if final_label == "EXTREMIST":
                        extremist_count += 1

                    key_value = self.df_comments.at[idx, "key_col"]
                    to_update.append({
                        "key_value": key_value,
                        "ai_label": final_label,
                        "ai_explanation": expl,
                        "ai_model": model_name
                    })

                # Commit per batch
                if to_update:
                    trans = conn.begin()
                    try:
                        count = self._persist_batch_results(conn, to_update)
                        trans.commit()
                        self.log_success(f"[Batch {i}] Persisted {count} row(s).")
                    except Exception as e:
                        trans.rollback()
                        self.log_error(f"[Batch {i}] DB commit failed, rolled back: {e}")

        return extremist_count

    # --------------------------
    # Classify comments (batched + multiprocessing by default)
    # --------------------------
    def classify_comments(self, persist_prefilter_safe_rows: bool = True):
        if self.df_comments.empty:
            self.log_warn("No comments loaded. Cannot classify.")
            return

        self.log_info("Preparing classification...")
        self.df_comments["ai_classification"] = ""

        # Pre-filter short comments → SAFE directly
        needs_llm_mask = self.df_comments["body"].str.len() >= self.prefilter_min_len
        self.df_comments.loc[~needs_llm_mask, "ai_classification"] = "SAFE"

        # Optionally persist pre-filtered SAFE rows immediately
        if persist_prefilter_safe_rows:
            count = self.persist_prefilter_safe()
            if count:
                self.log_success(f"Persisted {count} pre-filtered SAFE comment(s) to DB.")

        indices_to_classify = self.df_comments.index[needs_llm_mask].tolist()
        if not indices_to_classify:
            self.log_success("Nothing to classify; all comments marked SAFE by pre-filter.")
            return

        # Build pairs from indices
        pairs = [(int(idx), self.df_comments.at[idx, "body"]) for idx in indices_to_classify]

        # First, apply nominal batch size for predictable progress
        initial_batches: List[List[Tuple[int, str]]] = [
            pairs[i:i + self.batch_size] for i in range(0, len(pairs), self.batch_size)
        ]
        # Flatten and re-batch to respect prompt budget (avoid context overflow)
        flattened = [p for batch in initial_batches for p in batch]
        batches = _rebatch_to_prompt_budget(flattened)

        self.log_info(
            f"Total to classify via LLM: {len(pairs)} | prompt-budgeted batches: {len(batches)} | "
            f"per-item cap={PER_ITEM_CHAR_LIMIT} chars"
        )

        # Conservative token budget per batch (label + short explanation + confidence + signals)
        def tokens_for_batch(n: int) -> int:
            # Compact JSON results + label/explanation/conf per item
            return min(256, max(128, 32 + n * 10))

        # Warmup on a tiny batch in main process (e.g., first 2 items), persist too
        if self.warmup_in_main and len(batches) > 0:
            warmup_pairs = batches[0][: min(2, len(batches[0]))]
            self._main_process_warmup(warmup_pairs)

        # Sequential mode (safe on low-RAM machines)
        if self.force_sequential:
            extremist_count = self._classify_batched_sequential(batches, tokens_for_batch)
            self.log_success(f"Classification done (sequential). Found {extremist_count} extremist comment(s).")
            return

        # Multiprocessing path
        self.log_info(f"Spawning {self.max_workers} worker process(es) with individual GPT4All models...")
        extremist_count = 0
        model_name = self._model_name()
        engine = create_engine(self.database_url)

        def make_args(batch: List[Tuple[int, str]]) -> Tuple[str, List[Tuple[int, str]], int, float]:
            max_toks = tokens_for_batch(len(batch))
            return (self.model_path, batch, max_toks, self.temperature)

        with engine.connect() as conn:
            try:
                with ProcessPoolExecutor(
                    max_workers=self.max_workers,
                    initializer=_init_worker_model,
                    initargs=(self.model_path,)
                ) as executor:
                    # Stream tasks progressively to keep memory pressure low
                    for i, mapping in enumerate(
                        tqdm(
                            executor.map(_classify_batch_worker, (make_args(b) for b in batches)),
                            total=len(batches), desc="Classifying (batches)", ncols=100
                        ),
                        start=1
                    ):
                        to_update = []
                        for idx, result in (mapping or {}).items():
                            final_label = _finalize_label(
                                result.get("label", "SAFE"),
                                float(result.get("confidence", 0.0)),
                                result.get("signals", {})
                            )
                            expl = result.get("explanation", "")

                            # Update DataFrame
                            self.df_comments.at[idx, "ai_classification"] = final_label
                            if final_label == "EXTREMIST":
                                extremist_count += 1

                            # Prepare DB update row
                            key_value = self.df_comments.at[idx, "key_col"]
                            to_update.append({
                                "key_value": key_value,
                                "ai_label": final_label,
                                "ai_explanation": expl,
                                "ai_model": model_name
                            })

                        # Persist this batch and commit
                        if to_update:
                            trans = conn.begin()
                            try:
                                count = self._persist_batch_results(conn, to_update)
                                trans.commit()
                                self.log_success(f"[Batch {i}] Persisted {count} row(s).")
                            except Exception as e:
                                trans.rollback()
                                self.log_error(f"[Batch {i}] DB commit failed, rolled back: {e}")
            except Exception as e:
                self.log_error(f"Multiprocessing path failed: {e}")
                self.log_warn("Falling back to sequential classification...")
                extremist_count = self._classify_batched_sequential(batches, tokens_for_batch)

        self.log_success(f"Classification done. Found {extremist_count} extremist comment(s).")

    # --------------------------
    # Export flagged comments to Excel (optional)
    # --------------------------
    def export_flagged_comments(self, output: str = "flagged_comments.xlsx"):
        extremist_comments = self.df_comments[self.df_comments["ai_classification"] == "EXTREMIST"]
        self.log_info(f"Exporting extremist comments to {output}...")
        extremist_comments.to_excel(output, index=False, engine="openpyxl")
        self.log_success(f"Export completed: {len(extremist_comments)} comment(s) saved to {output}")


# --------------------------
# Script entrypoint (example run)
# --------------------------
if __name__ == "__main__":
    sentinel = RedditSentinel(
        model_path="models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf",
        database_url=os.environ.get(
            "DATABASE_URL",
            "postgresql://reddit_user:reddit_pass@localhost:5432/reddit_db"
        ),
        batch_size=16,              # start safe; bump to 32 if stable
        max_workers=1,              # increase to 2+ only if RAM allows (smaller model recommended)
        temperature=0.0,
        prefilter_min_len=10,
        warmup_in_main=True,
        persist_warmup=True,        # persist the first tiny batch immediately
        force_sequential=False,     # set True if your machine struggles with multiprocessing
        limit=None                  # set e.g. 5000 for testing
    )

    # Pipeline
    sentinel.load_comments_from_db()     # Only ai_label IS NULL
    sentinel.load_model()                # compatibility no-op
    sentinel.classify_comments(persist_prefilter_safe_rows=True)  # write short SAFE rows first
    # Optional Excel export of extremist-only subset
    sentinel.export_flagged_comments("flagged_comments.xlsx")

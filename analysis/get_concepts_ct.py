#!/usr/bin/env python3
"""
Port of CLEAR's examples/concepts/get_concepts.py to chest-CT reports.

Differences vs the original:
- Talks to a vLLM **OpenAI-compatible HTTP server** (the way we run vLLM in
  containers on this GH200 node), not the in-process ``vllm.LLM`` class.
- Two few-shot examples are CT-flavored instead of CXR-flavored.
- Iterates the CT-RATE / MERLIN training-report CSVs (one row per volume),
  with a per-text dedup so the same report is only sent to the LLM once even
  when it is shared across multiple volumes (e.g. ``train_1_a_1`` /
  ``train_1_a_2`` share the same Findings_EN).
- Writes incrementally as JSON-Lines so we can resume on restart.

Output schema (one record per line, matches CLEAR's ``mimic_concepts.json``
list-of-dicts shape after ``[json.loads(l) for l in open(out)]``):

    {"id": "<text-hash or report-id>", "model_output": "<raw LLM JSON>"}

Cleaning/dedup of the produced concepts is intentionally out of scope --
CLEAR did not do it in this script either; it lives downstream in
``concept_importance.py`` / embedding-time.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from openai import AsyncOpenAI
except ImportError as e:
    sys.exit("`pip install openai` first (the vLLM portal speaks the OpenAI API).")


# ---------------------------------------------------------------------------
# Prompt: same shape as CLEAR's, two CT-flavored few-shot examples.
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Please respond in valid JSON only."
)

# The two few-shot example reports below are authored synthetic composites in
# CT-RATE style, not real patient reports. Verified 2026-08-24: neither report
# nor its distinctive clauses appears in the CT-RATE or Merlin report corpora.
EX1_REPORT = (
    "Trachea and main bronchi are patent. Calcific plaques in the aortic arch. "
    "Heart size is normal and there is no pericardial effusion. No mediastinal "
    "lymphadenopathy. In the lung windows, peribronchial reticulonodular "
    "densities and minimal consolidations are present in the bilateral lower "
    "lobes; subsegmental atelectasis in the right middle lobe. The visualized "
    "left kidney is atrophic. Anterior osteophytes are seen in the thoracic "
    "vertebrae."
)
EX1_OBS = [
    "calcific plaques in the aortic arch",
    "normal heart size",
    "no pericardial effusion",
    "no mediastinal lymphadenopathy",
    "peribronchial reticulonodular densities in bilateral lower lobes",
    "minimal consolidation in bilateral lower lobes",
    "subsegmental atelectasis in the right middle lobe",
    "atrophic left kidney",
    "anterior osteophytes in the thoracic vertebrae",
]

EX2_REPORT = (
    "Spiculated solid nodule measuring 18 mm in the right upper lobe, "
    "suspicious for primary lung neoplasm. Right paratracheal and subcarinal "
    "lymphadenopathy with the largest node measuring 14 mm in short axis. "
    "Centrilobular emphysema in the upper lobes. Mild bronchiectasis in the "
    "right middle lobe. Mosaic attenuation pattern in the lower lobes. Small "
    "right pleural effusion. Mild cardiomegaly. Coronary artery calcifications."
)
EX2_OBS = [
    "spiculated solid nodule in the right upper lobe measuring 18 mm",
    "suspicious for primary lung neoplasm",
    "right paratracheal lymphadenopathy",
    "subcarinal lymphadenopathy",
    "largest mediastinal node measures 14 mm in short axis",
    "centrilobular emphysema in the upper lobes",
    "mild bronchiectasis in the right middle lobe",
    "mosaic attenuation pattern in the lower lobes",
    "small right pleural effusion",
    "mild cardiomegaly",
    "coronary artery calcifications",
]


def _format_example(report: str, obs: list[str]) -> str:
    return (
        f"Question: What are the descriptive observations in the report? {report}\n"
        f"Respond with a JSON object with this structure:\n"
        f'{{"observations": {json.dumps(obs)}}}'
    )


def build_user_prompt(report: str) -> str:
    return (
        _format_example(EX1_REPORT, EX1_OBS) + "\n"
        + _format_example(EX2_REPORT, EX2_OBS) + "\n"
        + f"Question: What are the descriptive observations in the report? {report}\n"
        f"Respond with a JSON object with this structure:"
    )


def normalize(s: str) -> str:
    return " ".join(str(s).split()) if s and not pd.isna(s) else ""


def text_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Concurrent driver
# ---------------------------------------------------------------------------
async def run_one(client, model, prompt, sem):
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2048,
            # Qwen3-style reasoning models default to thinking-on; disable for
            # this structured extraction task (faster, deterministic content).
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content


async def main_async(args):
    df = pd.read_csv(args.reports_csv)
    if args.text_col not in df.columns:
        sys.exit(f"--text_col {args.text_col!r} not in {list(df.columns)}")
    df["__text"] = df[args.text_col].map(normalize)
    df = df[df["__text"].str.len() > 0].copy()

    # Per-text dedup: many CT-RATE volumes share the same report.
    df["__id"] = df["__text"].map(text_id)
    df = df.drop_duplicates("__id").reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"reports to process: {len(df)} (after dedup, after --limit)")

    # Resume by skipping ids already in the output JSONL.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    continue
        print(f"already done: {len(done_ids)} (resuming, will append)")

    todo = df[~df["__id"].isin(done_ids)].reset_index(drop=True)
    if len(todo) == 0:
        print("nothing to do.")
        return

    client = AsyncOpenAI(base_url=args.vllm_url, api_key="EMPTY")
    sem = asyncio.Semaphore(args.concurrency)

    async def worker(rid, txt):
        try:
            out = await run_one(client, args.model, build_user_prompt(txt), sem)
        except Exception as e:
            return rid, None, str(e)
        return rid, out, None

    tasks = [
        asyncio.create_task(worker(row["__id"], row["__text"]))
        for _, row in todo.iterrows()
    ]
    with out_path.open("a") as f, tqdm(total=len(tasks)) as pbar:
        for fut in asyncio.as_completed(tasks):
            rid, out, err = await fut
            if err is not None:
                pbar.write(f"[fail {rid}] {err[:200]}")
                pbar.update(1)
                continue
            f.write(json.dumps({"id": rid, "model_output": out}) + "\n")
            f.flush()
            pbar.update(1)
    print(f"done. wrote {out_path}")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reports_csv", required=True,
        help="e.g. /path/to/ACT/model/data/ct_rate/train_reports.csv",
    )
    ap.add_argument(
        "--text_col", default="Findings_EN",
        help="report column to feed the LLM (CT-RATE: Findings_EN; "
             "MERLIN reports use 'report')",
    )
    ap.add_argument(
        "--vllm_url", default=os.environ.get("VLLM_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL of the vLLM portal",
    )
    ap.add_argument(
        "--model", required=True,
        help="model name as served by vLLM (e.g. Qwen/Qwen3-30B-A3B-Instruct-2507)",
    )
    ap.add_argument("--out", required=True, help="JSONL output path")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, only process the first N reports (for piloting)")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse()))

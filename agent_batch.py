"""
Customer Feedback AI Agent -- Steady Batch Mode
--------------------------------------------------
Processes the full dataset (or however many rows you set) one tweet at
a time, in order, with no artificial delays or concurrency -- just a
steady run from start to finish. Uses your local Ollama model, so it's
free.

Saves progress every CHECKPOINT_EVERY tweets, so if it crashes partway
through (or you stop it with Ctrl+C), you don't lose everything --
re-running will pick up where it left off.

Setup (one-time): same as agent_local.py
  1. Install Ollama, run: ollama pull qwen2.5:7b-instruct
  2. pip install ollama pandas tqdm
"""

import re
import json
import time
from pathlib import Path
import pandas as pd
import ollama
from tqdm import tqdm

CSV_PATH = "amazon_tweets_10k.csv"
TEXT_COLUMN = "text"
OUTPUT_PATH = "agent_results_full.csv"
CHECKPOINT_EVERY = 25             # save progress every N tweets

# Set this to None to process every row in the CSV, or a number to cap it
NUM_TWEETS_TO_PROCESS = 100

MODEL = "qwen2.5:7b-instruct"

INTENT_OPTIONS = ["question", "complaint", "request", "praise"]
PRODUCT_OPTIONS = ["order", "delivery", "payment", "account", "device", "app", "other"]
PROBLEM_OPTIONS = ["delay", "damaged", "wrong_item", "refund", "access", "billing", "none", "other"]

ALWAYS_ESCALATE_PATTERNS = [
    r"\blawyer\b", r"\blegal\b", r"\bsue\b", r"\bfraud\b",
    r"\bunsafe\b", r"\binjur(y|ed)\b", r"\bdiscriminat", r"\bscam\b",
]


def call_local_model(prompt: str, max_tokens: int = 200) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens, "temperature": 0.2},
    )
    return response["message"]["content"].strip()


def classify_tweet(text: str) -> dict:
    prompt = f"""Classify this customer tweet. Respond with ONLY a JSON object, no other text, no markdown code fences.

Tweet: "{text}"

Return JSON with exactly these fields:
- "intent": one of {INTENT_OPTIONS}
- "product": one of {PRODUCT_OPTIONS}
- "problem": one of {PROBLEM_OPTIONS}
- "confidence": a number from 0.0 to 1.0

JSON:"""
    raw = call_local_model(prompt, max_tokens=200)
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": "unknown", "product": "unknown", "problem": "unknown", "confidence": 0.0}


def contains_escalation_keyword(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in ALWAYS_ESCALATE_PATTERNS)


def llm_escalation_check(text: str, classification: dict) -> bool:
    prompt = f"""A customer support tweet was classified as:
Intent: {classification['intent']}, Product: {classification['product']}, Problem: {classification['problem']}

Tweet: "{text}"

Should this be escalated to a human agent instead of getting an automated reply?
Respond with ONLY the single word "yes" or "no", nothing else."""
    raw = call_local_model(prompt, max_tokens=10)
    return raw.strip().lower().startswith("y")


def should_escalate(text: str, classification: dict, confidence_threshold: float = 0.6) -> dict:
    if contains_escalation_keyword(text):
        return {"escalate": True, "reason": "keyword_match"}
    if classification.get("confidence", 1.0) < confidence_threshold:
        if llm_escalation_check(text, classification):
            return {"escalate": True, "reason": "llm_judged_needs_human"}
        return {"escalate": False, "reason": "llm_judged_ok_low_confidence"}
    return {"escalate": False, "reason": "auto_handled"}


def draft_reply(text: str, classification: dict) -> str:
    prompt = f"""You are a helpful, empathetic Amazon customer support agent.
Keep the reply concise (under 280 characters), acknowledge the issue, and give a clear next step.

Customer tweet: "{text}"
Classified as: intent={classification['intent']}, product={classification['product']}, problem={classification['problem']}

Write only the reply text, nothing else."""
    return call_local_model(prompt, max_tokens=150)


def process_one(text: str) -> dict:
    classification = classify_tweet(text)
    decision = should_escalate(text, classification)
    reply = None if decision["escalate"] else draft_reply(text, classification)
    return {
        "tweet": text,
        "intent": classification.get("intent"),
        "product": classification.get("product"),
        "problem": classification.get("problem"),
        "confidence": classification.get("confidence"),
        "escalate": decision["escalate"],
        "escalation_reason": decision["reason"],
        "drafted_reply": reply,
    }


def load_existing_results() -> pd.DataFrame:
    if Path(OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH)
    return pd.DataFrame()


def main():
    df = pd.read_csv(CSV_PATH).dropna(subset=[TEXT_COLUMN])
    if NUM_TWEETS_TO_PROCESS is not None:
        df = df.head(NUM_TWEETS_TO_PROCESS)

    existing = load_existing_results()
    already_done = set(existing["tweet"]) if not existing.empty else set()

    remaining = df[~df[TEXT_COLUMN].isin(already_done)]

    print(f"Total tweets: {len(df)}  |  Already processed: {len(already_done)}  |  Remaining: {len(remaining)}")
    if len(remaining) == 0:
        print("Nothing left to process -- all done.")
        return

    results = existing.to_dict("records") if not existing.empty else []
    start_time = time.time()

    for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining), desc="Processing tweets")):
        text = str(row[TEXT_COLUMN])
        result = process_one(text)
        results.append(result)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    # Final save
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    elapsed = time.time() - start_time
    print(f"\nDone. Processed {len(remaining)} new tweets in {elapsed:.1f}s "
          f"({elapsed/max(len(remaining),1):.2f}s/tweet average).")
    print(f"Total results saved: {len(results)} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

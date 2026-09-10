"""
Customer Feedback AI Agent -- Free/Local Version
---------------------------------------------------
Same as agent.py, but uses a locally-run open-source model via Ollama
instead of the paid Anthropic API. Runs entirely on your own GPU,
completely free, no internet required after the model is downloaded.

Setup (one-time):
  1. Install Ollama from https://ollama.com
  2. Run: ollama pull qwen2.5:7b-instruct
  3. Run: pip install ollama pandas

No API key, no .env file needed for this version.
"""

import re
import json
import pandas as pd
import ollama

CSV_PATH = "amazon_tweets_10k.csv"
TEXT_COLUMN = "text"
NUM_TWEETS_TO_PROCESS = 10

MODEL = "qwen2.5:7b-instruct"

INTENT_OPTIONS = ["question", "complaint", "request", "praise"]
PRODUCT_OPTIONS = ["order", "delivery", "payment", "account", "device", "app", "other"]
PROBLEM_OPTIONS = ["delay", "damaged", "wrong_item", "refund", "access", "billing", "none", "other"]

ALWAYS_ESCALATE_PATTERNS = [
    r"\blawyer\b", r"\blegal\b", r"\bsue\b", r"\bfraud\b",
    r"\bunsafe\b", r"\binjur(y|ed)\b", r"\bdiscriminat", r"\bscam\b",
]


def call_local_model(prompt: str, max_tokens: int = 200) -> str:
    """Sends a prompt to the locally-running Ollama model and returns its text response."""
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

    # Local models are sometimes chattier than Claude -- try to pull out
    # just the JSON object if there's extra text around it
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [warning] Could not parse classification, got: {raw}")
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


def main():
    print(f"Using local model: {MODEL} (make sure Ollama is running and the model is pulled)\n")

    df = pd.read_csv(CSV_PATH).dropna(subset=[TEXT_COLUMN]).head(NUM_TWEETS_TO_PROCESS)
    print(f"Processing {len(df)} tweets...\n")

    results = []
    for i, row in df.iterrows():
        text = str(row[TEXT_COLUMN])
        print("=" * 70)
        print(f"Tweet: {text}")

        classification = classify_tweet(text)
        print(f"Classification: {classification}")

        decision = should_escalate(text, classification)
        print(f"Escalation decision: {decision}")

        if decision["escalate"]:
            reply = None
            print("-> Routed to human review, no auto-reply drafted.")
        else:
            reply = draft_reply(text, classification)
            print(f"Drafted reply: {reply}")

        results.append({
            "tweet": text,
            "intent": classification.get("intent"),
            "product": classification.get("product"),
            "problem": classification.get("problem"),
            "confidence": classification.get("confidence"),
            "escalate": decision["escalate"],
            "escalation_reason": decision["reason"],
            "drafted_reply": reply,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv("agent_results_local.csv", index=False)
    print(f"\nSaved results for {len(out_df)} tweets to agent_results_local.csv")


if __name__ == "__main__":
    main()

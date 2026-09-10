# Amazon Customer Feedback AI Agent

An AI agent that reads customer support tweets directed at Amazon
(@AmazonHelp), classifies each one, decides whether it needs human
escalation, and drafts a suggested reply — using a free, locally-run
open-source LLM (no paid API required).

## What it does

For each tweet, the agent:
1. **Classifies** it on three axes: intent (question/complaint/request/praise),
   product (order/delivery/payment/account/device/app), and problem type
   (delay/damaged/wrong_item/refund/access/billing).
2. **Decides on escalation** — keyword rules catch clear-cut cases (legal
   threats, safety issues, fraud); anything the classifier is unsure about
   gets a second opinion from the LLM before deciding.
3. **Drafts a reply** for anything not escalated.

## Setup

1. Install [Ollama](https://ollama.com) (runs open-source LLMs locally, free).
2. Pull the model used here:
   ```
   ollama pull qwen2.5:7b-instruct
   ```
3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data

- `filter_amazon_tweets.py` — filters the full Kaggle
  ["Customer Support on Twitter"](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
  dataset (`twcs.csv`, ~2.8M rows, not included here due to size) down to
  ~10,000 real customer tweets directed at AmazonHelp.
- `amazon_tweets_10k.csv` — the filtered output, included in this repo so
  you can run the agent immediately without re-downloading the full dataset.

## Running the agent

**Quick test on a handful of tweets:**
```
python agent_local.py
```
Processes the first 10 tweets (edit `NUM_TWEETS_TO_PROCESS` at the top of
the file to change this) and prints each classification, escalation
decision, and drafted reply live. Saves to `agent_results_local.csv`.

**Full steady run through more/all of the dataset:**
```
python agent_batch.py
```
Processes tweets in order with a progress bar, checkpointing every 25
tweets so you can safely stop and resume. Saves to `agent_results_full.csv`.
Edit `NUM_TWEETS_TO_PROCESS` at the top (set to `None` for the whole file,
or a number to cap it — a subsample is recommended for reproducibility).

**Note on `agent.py`:** an earlier version using the paid Anthropic Claude
API instead of a local model. Kept for reference; requires an
`ANTHROPIC_API_KEY` in a `.env` file to run. `agent_local.py` and
`agent_batch.py` are the actively-used, free versions.

## Status / results

_Fill in after running: classifier accuracy against a hand-labeled sample,
how many tweets were auto-handled vs. escalated, and a couple of example
tweet → classification → drafted reply outputs._

## What I learned

_A few sentences on what was hardest and what's next._

## License

MIT

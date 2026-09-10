# Report

## Problem framing

What does "good" mean for Amazon customer support specifically? What did you
choose NOT to build, and why?

## Results vs. baselines

Compare your system against:
- A trivial baseline (e.g. always predict the most common category, never escalate)
- A simple baseline (e.g. keyword matching, or TF-IDF + logistic regression
  without any LLM involvement)

Report real numbers here once you've run `src/evaluate.py` against your
golden set.

## Failure analysis

Your top 5 failure modes, each with a real example tweet and your hypothesis
for why it failed.

## What's misleading about my headline number?

Mandatory honest section — e.g. class imbalance inflating accuracy, golden
set sampling bias, confidence scores that don't reflect true calibration, etc.

## What I'd do next with one more week


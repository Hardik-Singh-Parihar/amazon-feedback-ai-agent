# Decision Log

Non-obvious decisions made while building this, and why.

- Chose AmazonHelp as the single brand because the dataset has enough volume and natural sub-categories (orders, delivery, payments, devices) to make both product and problem classification meaningful within one brand.
- Chose to define intent/product/problem categories by reading real tweets first (open coding) rather than importing Banking77's generic intent set, because Banking77 is banking-specific and didn't fit Amazon's issue types.
- Chose Qwen2.5 7B (local, via Ollama) over a paid API to keep the whole pipeline free and reproducible without requiring reviewers to have their own API key.

# Intelligent Candidate Discovery & Ranking — PoC

## Quick Start

```bash
pip install orjson   # optional but speeds up JSON parsing ~2.4x
python fast_rank.py --candidates candidates.jsonl --out submission.csv
```

Runtime: **~31s on CPU (no GPU, no network)** for 100K candidates.

---

## Architecture

A multi-signal, rules-free ranking engine that goes beyond keyword matching.

### Scoring Dimensions

| Dimension | Weight | What it captures |
|-----------|--------|-----------------|
| Role Legitimacy | 20% | Is this actually an AI/ML role? (title + career trajectory) |
| Technical Skills | 30% | Right stack: embeddings, vector DBs, ranking/IR, Python |
| Career Quality | 20% | Product company exp, YoE sweet spot, production deployment |
| Semantic Fit | 15% | Career narrative alignment to JD via word-set intersection |
| Behavioral Signals | 10% | Availability, response rate, interview reliability |
| Location/Logistics | 5% | India-preferred, notice period, relocation intent |

### Red-Flag Multipliers (applied after weighted sum)

| Flag | Multiplier | Rationale |
|------|-----------|-----------|
| Non-technical role (HR, Marketing, Accountant…) | 0.05× | Career domain mismatch |
| Entire career at consulting giants (TCS/Infosys/Wipro…) | 0.35× | JD explicitly requires product-company background |
| Outside India + unwilling to relocate | 0.25× | No visa sponsorship; India-based role |
| Only recent LLM/LangChain exp, no pre-LLM AI | 0.50× | JD: "people who understood retrieval before it was fashionable" |
| Honeypot (impossible timeline) | 0.0× | Impossible profile → auto-exclude |

---

## Key Design Decisions

### Why semantic fit uses set intersection, not embeddings
The compute constraint (5min CPU, no network) rules out running an embedding model at inference time. Word/bigram set intersection against JD-derived vocabulary captures the right signals with negligible compute.

### Why role legitimacy outweighs raw skill score
A candidate with 20 AI keywords listed as skills but whose career shows "Marketing Manager → Operations Manager → HR Manager" is a trap. Role legitimacy checks title AND full career trajectory, downweighting keyword stuffers by 20×.

### Why behavioral signals are a multiplier (not just additive)
A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is not actually hirable. Behavioral signals compress the otherwise-high scores of inactive candidates, surfacing who is genuinely available.

### Honeypot detection
Checks for timeline impossibilities:
- Experience claimed > (graduation year → today)
- career `duration_months` sum >> stated `years_of_experience`
- Job start dates in the future or before 1970

---

## Top-10 Output (sample run on 100K pool)

| Rank | Candidate | Title | YoE | Companies | Flags |
|------|-----------|-------|-----|-----------|-------|
| 1 | CAND_0018499 | Senior ML Engineer | 7.2y | Zomato, Google, Flipkart | — |
| 2 | CAND_0086022 | Senior Applied Scientist | 5.3y | Sarvam AI, Uber | — |
| 3 | CAND_0078002 | ML Engineer | 6.3y | Meta, CRED | — |
| 4 | CAND_0039383 | Applied ML Engineer | 7.1y | Meesho, Swiggy, Paytm | — |
| 5 | CAND_0050454 | AI Engineer | 6.8y | Rephrase.ai, Uber, Adobe | — |
| 6 | CAND_0091909 | ML Engineer | 6.9y | Rephrase.ai, Meta | — |
| 7 | CAND_0064326 | Search Engineer | 7.6y | Sarvam AI, Freshworks, Apple | — |
| 8 | CAND_0009691 | Applied ML Engineer | 6.2y | LinkedIn, Amazon | — |
| 9 | CAND_0006557 | NLP Engineer | 7.9y | Paytm, Apple | — |
| 10 | CAND_0008425 | Senior NLP Engineer | 7.8y | Ola, Zomato, Amazon | — |

---

## Files

| File | Purpose |
|------|---------|
| `fast_rank.py` | Main ranker — run this to produce submission |
| `rank.py` | Full-featured version (more readable, slightly slower) |
| `team_redrob_ai.csv` | Submission output (top-100 ranked candidates) |
| `submission_metadata_template.yaml` | Metadata template |

---

## Constraints Compliance

| Constraint | Limit | Actual |
|------------|-------|--------|
| Wall-clock | ≤ 5 min | ~31s |
| Memory | ≤ 16 GB | <500 MB |
| Compute | CPU only | ✅ |
| Network | Off | ✅ (no API calls) |
| GPU | None | ✅ |

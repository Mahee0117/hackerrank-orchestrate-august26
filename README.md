# 📬 Message Notification Router

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Ollama](https://img.shields.io/badge/LLM-Gemma4-green)
![Hackathon](https://img.shields.io/badge/HackerRank-Orchestrate-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

An AI-powered routing engine for WhatsApp that decides, **per user and per message**, whether to `notify`, `digest`, or `mute` — reasoning over text, images, and voice notes using a locally-run multimodal LLM.

Built as a submission for the HackerRank Orchestrate 24-hour Hackathon, demonstrating an AI-powered personalized WhatsApp notification routing system.

---

## Features

- AI-powered, personalized notification routing (`notify` / `digest` / `mute`)
- Multimodal reasoning — text, image posters/screenshots, and voice notes
- Scam and spam detection using rule-based safety signals + LLM judgment
- Historical context reasoning — routes messages using each user's past behavior
- Evidence-backed decisions — surfaces relevant historical message IDs, not just a verdict
- Fully local inference via Ollama + Gemma4 — no API keys, no per-token cost
- Crash-safe: resumes automatically after interruption, writes output incrementally
- Built-in output verification and evaluation workflow

---

## Tech Stack

- **Python** — pipeline and orchestration
- **Pandas** — dataset loading and joins
- **Ollama + Gemma4** — local multimodal LLM (text + vision)
- **Whisper** (optional) — voice note transcription
- **CSV** — input/output data format

---

## The Problem

WhatsApp is noisy. A single stream can mix family chats, society notices, school updates, co-worker messages, business promos, image posters, voice notes, and scams. Treating every message the same causes two failures: important messages get buried, and low-value or risky messages interrupt the user anyway.

This system makes a **personalized** routing decision for every incoming message:

| Action | Meaning |
|---|---|
| `notify` | Important enough to interrupt the user now |
| `digest` | Safe but low priority — show later |
| `mute` | Repetitive, unwanted, low-value, suspicious, scam-like, or unsafe |

The same message can (and should) get different actions for different users — a sale poster is signal for one user and noise for another; a payment reminder is legitimate from a trusted admin but risky from an unknown sender.

---

## Architecture

```
messages.csv
     │
     ▼
  loader          Loads all dataset CSVs (users, groups, businesses,
                   history, events, images, voice notes)
     │
     ▼
  media_processor  Image → Gemma4 Vision   |   Voice → Whisper / fallback
     │
     ▼
  context_builder  Joins user behavior, group/business metadata, and
                    ranked history into a structured context + evidence
                    IDs + safety signals
     │
     ▼
  prompts          Builds a compact, structured prompt for the LLM
     │
     ▼
  ollama_client     Calls Gemma4 locally via Ollama, strict JSON output
     │
     ▼
  writer            Validates against schema, clamps confidence, writes row
     │
     ▼
  dataset/output.csv
```

Each message flows through the pipeline independently, and every completed prediction is written to `output.csv` immediately — so an interrupted run loses at most one row and can always be resumed.

![alt text](../Screenshots/System_Architecture.png)

---

## Why Gemma4 (via Ollama)?

- **Runs entirely locally** — no API keys, no per-token cost, no rate limits during a 24-hour hackathon
- **Multimodal out of the box** — the same model handles text reasoning and image understanding, avoiding a separate vision pipeline
- **Right-sized for the task** — this is a bounded classification problem (3 actions × 11 message types), not open-ended generation, so a local model is sufficient when paired with strong context and a constrained JSON contract
- **Deterministic-leaning** — low temperature and a strict JSON-only system prompt keep outputs consistent and parseable

---

## How Routing Decisions Are Made

Before a decision is generated, the system builds a full picture of the message *and* the receiving user:

- **User behavior** — quiet hours, recent opens, replies, dismissals, and reports
- **Conversation context** — group role/mute state and activity, or business identity, verification, and account history
- **Historical pattern matching** — how this user has reacted to similar past messages
- **Safety signals** — rule-based risk flags (e.g. high forward count, unverified sender, prior reports) computed independently of the LLM
- **Media content** — extracted meaning from any attached image or voice note

This context, along with a rule-based confidence baseline, is handed to Gemma4, which returns the final `action`, `message_type`, `reason`, and `confidence` as strict JSON. Historical messages that most closely match the current one (same sender/group/business and prior user reaction) are surfaced as `evidence_message_ids`, or marked `none` if nothing relevant exists.

**Images** are described by Gemma4 Vision and fed into the same reasoning pipeline as text. **Voice notes** are transcribed with Whisper when available, falling back to a structured placeholder otherwise — so the pipeline never breaks on a missing optional dependency.

---

![alt text](../Screenshots/AI_Decision.png)

## Reliability Features

- **Crash-safe incremental writes** — each row is saved the moment it's produced, not batched at the end
- **Resume support** — reruns automatically skip any message already in `output.csv`
- **Duplicate guard** — prevents the same message from being written twice
- **Schema validation on write** — invalid or missing fields fall back to safe defaults instead of corrupting the file
- **Graceful LLM failure handling** — unparseable model output still produces a safe fallback row instead of crashing the run
- **Post-run verification** — automatically checks schema, row count, ID coverage, and duplicates after every run
- **Optional concurrency** — overlaps context preparation with LLM inference for faster throughput

---

## Repository Layout

```
.
├── README.md
├── problem_statement.md          # Full challenge spec
├── AGENTS.md                     # Rules for AI coding tools + transcript logging
├── code/
│   ├── main.py                   # Entry point / pipeline orchestration
│   ├── loader.py                 # Dataset loading
│   ├── media_processor.py        # Image (vision) + voice (ASR) handling
│   ├── context_builder.py        # Context assembly, evidence ranking, safety signals
│   ├── prompts.py                # Prompt construction
│   ├── ollama_client.py          # Gemma4 inference via Ollama
│   ├── writer.py                 # Validation + CSV output
│   ├── requirements.txt
│   └── evaluation/
│       └── main.py               # Scoring against sample_messages.csv
└── dataset/
    ├── messages.csv               # Messages to route
    ├── output.csv                 # Predictions (generated)
    ├── sample_messages.csv        # Solved examples for format reference
    ├── users.csv
    ├── groups.csv
    ├── group_members.csv
    ├── business_accounts.csv
    ├── user_business_history.csv
    ├── message_history.csv
    ├── message_events.csv
    ├── images.csv
    ├── voice_notes.csv
    ├── daily_notification_summary.csv
    └── media/
        ├── images/
        └── audio/
```

---

## Setup

**Prerequisites**

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- The `gemma4` model pulled:
  ```bash
  ollama pull gemma4
  ```
- (Optional, for real voice transcription) `openai-whisper` installed — otherwise voice notes use a structured fallback description

**Install dependencies**

```bash
cd code
pip install -r requirements.txt
```

---

## Usage

Run the full pipeline from the `code/` directory:

```bash
python3 main.py
```

**Flags**

| Flag | Description |
|---|---|
| `--limit N` | Process only the first `N` messages (useful for quick testing) |
| `--fresh` | Delete any existing `output.csv` and reprocess everything from scratch |
| `--workers N` | Run `N` concurrent workers for text/voice messages (default `1`). Image messages always run sequentially. |

Example — quick smoke test:

```bash
python3 main.py --limit 5 --fresh
```

Example — full run with concurrency:

```bash
python3 main.py --workers 3
```

If interrupted, simply rerun the same command — already-completed messages are automatically skipped.

---

## Output Format

`dataset/output.csv` contains one row per message in `dataset/messages.csv`:

| Column | Description |
|---|---|
| `message_id` | Incoming message ID |
| `action` | `notify`, `digest`, or `mute` |
| `message_type` | `personal`, `urgent`, `event`, `payment`, `business_update`, `promotion`, `greeting`, `forward`, `spam`, `scam`, `unknown` |
| `reason` | Short human-readable explanation |
| `confidence` | Float in `[0, 1]` |
| `evidence_message_ids` | Semicolon-separated historical message IDs used as evidence, or `none` |

---

## Results

- ✔ Processed all 110 messages in `dataset/messages.csv`
- ✔ Generated 110 valid predictions with zero missing or extra IDs
- ✔ Zero duplicate rows in final output (verified programmatically)
- ✔ Resume-from-interruption tested and working
- ✔ Image (poster/screenshot) understanding via Gemma4 Vision
- ✔ Voice note handling with automatic fallback when Whisper is unavailable
- ✔ 100% strict JSON validation on LLM output before writing to disk

---

## Design Trade-offs

- **Local inference over an API model** — removes cost/rate-limit risk for a 24-hour build, at the cost of slower per-message latency (vision calls in particular)
- **Rule-based safety signals feeding the LLM**, rather than trusting model judgment alone — reduces the chance a plausible-sounding scam gets a high-confidence `notify`
- **Voice transcription has a graceful fallback** when Whisper isn't installed, trading transcription fidelity for pipeline robustness
- **Strict JSON-only prompting** trades some model expressiveness for output reliability, since the pipeline writes directly to a graded CSV
- **Incremental writes over batched writes** prioritize crash-safety and resumability over raw throughput

---

## Current Limitations

- Voice transcription quality depends on Whisper being installed locally; without it, voice notes fall back to a structured placeholder rather than true transcription
- Local inference is slower than a cloud API, especially for image (vision) messages
- Routing quality is ultimately bounded by Gemma4's reasoning quality on edge cases the rule-based safety signals don't catch
- Evidence retrieval uses metadata + heuristic scoring rather than semantic similarity, so it can miss less obvious historical matches

---

## Future Improvements

- Vector database for semantic (embedding-based) historical message retrieval
- Fine-tuned or distilled classifier for faster, cheaper routing on clear-cut cases
- Streaming inference for lower perceived latency
- Learning from real user feedback (opens/dismissals) to recalibrate confidence over time
- Mobile/edge deployment of the routing pipeline

---

## Evaluation

Predictions are scored against hidden ground-truth labels on:

- Correctness of `action`
- Correctness of `message_type`
- Usefulness and consistency of `reason`
- Whether `evidence_message_ids` point to genuinely relevant historical messages
- Confidence calibration

A local evaluation workflow is included at `code/evaluation/main.py`, scoring predictions against the solved rows in `dataset/sample_messages.csv` so approach quality can be checked before submission:

```bash
python3 code/evaluation/main.py
```

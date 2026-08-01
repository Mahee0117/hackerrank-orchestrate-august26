# 📬 Message Notification Router

An AI-powered routing engine for WhatsApp that decides, **per user and per message**, whether to `notify`, `digest`, or `mute` — reasoning over text, images, and voice notes using a locally-run multimodal LLM.

Built for the **HackerRank Orchestrate** 24-hour hackathon.

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
┌─────────────────┐
│  loader.py       │  Loads all dataset CSVs (users, groups, businesses,
└────────┬─────────┘  history, events, images, voice notes)
         ▼
┌─────────────────┐
│ media_processor  │  Image  → Gemma4 Vision (Ollama multimodal)
│      .py         │  Voice  → Whisper if available, else structured fallback
└────────┬─────────┘
         ▼
┌─────────────────┐
│ context_builder  │  Joins user behavior, group/business metadata,
│      .py         │  and ranked message_history/message_events into a
└────────┬─────────┘  single structured context + evidence IDs + safety flags
         ▼
┌─────────────────┐
│   prompts.py     │  Builds a compact, structured prompt (message +
└────────┬─────────┘  context + safety signals + confidence hint)
         ▼
┌─────────────────┐
│ ollama_client.py │  Calls Gemma4 locally via Ollama, forces strict JSON
└────────┬─────────┘  output, extracts and parses the JSON payload
         ▼
┌─────────────────┐
│   writer.py      │  Validates fields against the allowed schema,
└────────┬─────────┘  clamps confidence to [0, 1], appends the row
         ▼
   dataset/output.csv
```

Each message flows through the pipeline independently, and every completed prediction is written to `output.csv` immediately — so an interrupted run loses at most one row and can always be resumed.

---

## Why Gemma4 (via Ollama)?

- **Runs entirely locally** — no API keys, no per-token cost, no rate limits during a 24-hour hackathon.
- **Multimodal out of the box** — the same model handles text reasoning and image understanding (posters, screenshots), avoiding a separate vision pipeline.
- **Good enough reasoning for structured classification** — the task is a bounded decision (3 actions × 11 message types), not open-ended generation, so a mid-size local model is sufficient when paired with strong context and a constrained JSON output contract.
- **Deterministic-leaning** — run with `temperature=0.2` and a strict "JSON only, no markdown, no explanations" system prompt to keep outputs parseable and consistent.

---

## How Routing Decisions Are Made

`context_builder.py` assembles everything the LLM needs to reason about **this message, for this user**, before a single token is generated:

1. **User behavior** — quiet hours, recent opens/replies/dismissals/reports (`users.csv`).
2. **Conversation context** — group type, role, mute state, activity (`groups.csv`, `group_members.csv`); or business identity, verification, domain, account age, and report count (`business_accounts.csv`, `user_business_history.csv`).
3. **Historical pattern matching** — the user's past messages and how they reacted to them (`message_history.csv`, `message_events.csv`), scored and ranked for relevance to the current message.
4. **Safety signals** — rule-based flags (e.g. high `forwarded_count`, unverified business, no prior relationship, prior reports) computed before the LLM call, so risk isn't purely dependent on model judgment.
5. **Media content** — extracted text/description from images or voice notes (see below).

All of this is compressed into a single structured context block plus a **confidence hint** (a rule-based baseline in `[0.30, 0.99]`), and handed to Gemma4 along with the message itself. The model returns strict JSON: `action`, `message_type`, `reason`, `confidence` — which `writer.py` then validates and clamps against the allowed schema before it's ever written to disk.

### How `evidence_message_ids` are selected

For each message, `context_builder.py` scores candidate rows from `message_history.csv` against the current message (same sender / group / business, similar type, and the user's recorded reaction in `message_events.csv`), and surfaces the most relevant historical message IDs as evidence. If nothing relevant exists, it's marked `none` rather than forced.

### How images and voice notes are handled

- **Images** (`media_processor.py`): sent to Gemma4 Vision (the same model, used multimodally) to produce a plain-text description of the poster/screenshot content, which is then injected into the same context/prompt pipeline as any text message.
- **Voice notes**: a `SpeechTranscriber` strategy pattern — `WhisperTranscriber` is used if `openai-whisper` is installed locally; otherwise a `FallbackTranscriber` returns a structured placeholder so the pipeline degrades gracefully instead of crashing.
- Image messages are deliberately processed **last** in the run (`main.py` reorders text/voice before image), since vision calls are the slowest step — this maximizes how much output is safely on disk if a run is interrupted.

---

## Reliability Features

The pipeline was built assuming a local LLM run can be slow, interrupted, or occasionally malformed — so several safeguards are built in:

- **Crash-safe incremental writes** — each row is appended to `output.csv` the moment it's produced, not batched at the end.
- **Resume support** — reruns skip any `message_id` already present in `output.csv`.
- **Duplicate guard** — a `done_ids` set is checked immediately before every write.
- **Schema validation on write** — `action` and `message_type` are checked against the allowed value sets; invalid or missing values fall back to safe defaults (`digest` / `unknown`) rather than corrupting the file.
- **LLM fallback row** — if the model output can't be parsed as JSON, the message still gets a safe `digest` / `unknown` row with a logged error, instead of failing the whole run.
- **Built-in post-run verification** (`verify_output`) — checks column schema, row count, line count, full ID coverage, duplicates, and blank fields, and prints a pass/fail report after every run.
- **Optional concurrency** (`--workers N`) — overlaps context/prompt preparation with LLM inference I/O for text/voice messages; image (vision) messages always run sequentially since they're the slowest step.

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
- (Optional, for real voice transcription) `openai-whisper` installed — otherwise voice notes use a structured fallback description.

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

At the end of every run, the script verifies `output.csv` for schema correctness, full coverage, and duplicate-free rows, and prints a pass/fail report.

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

## Design Trade-offs

- **Local inference over an API model** — removes cost/rate-limit risk for a 24-hour build, at the cost of slower per-message latency (vision calls in particular).
- **Rule-based safety signals + confidence hint feeding the LLM**, rather than trusting the model's judgment alone — reduces the chance that a plausible-sounding scam message gets a high-confidence `notify`.
- **Voice transcription has a graceful fallback** when Whisper isn't installed, so the pipeline never blocks on an optional dependency — at the cost of lower-fidelity understanding of voice note content in that mode.
- **Strict JSON-only prompting** trades a small amount of model expressiveness for output reliability, which matters more given the pipeline writes directly to a graded CSV.
- **Incremental writes over batched writes** prioritize crash-safety and resumability over raw throughput.

---

## Evaluation

Predictions are scored against hidden ground-truth labels on:

- Correctness of `action`
- Correctness of `message_type`
- Usefulness and consistency of `reason`
- Whether `evidence_message_ids` point to genuinely relevant historical messages
- Confidence calibration

A local evaluation script is included at `code/evaluation/main.py` for checking predictions against `dataset/sample_messages.csv` before submission.

---

## Submission Checklist

- [ ] `output.csv` has exactly one row per `message_id` in `messages.csv`, no duplicates, no missing/extra IDs
- [ ] Columns match the required schema and order exactly
- [ ] `code.zip` excludes `__pycache__/`, `.git/`, `.DS_Store`, and virtual environments
- [ ] No hardcoded secrets — API keys (if any) are read from environment variables
- [ ] `log.txt` chat transcript included per `AGENTS.md`

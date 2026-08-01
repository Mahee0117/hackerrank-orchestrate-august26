"""
prompts.py
----------
Builds the LLM prompt for a single incoming message.

Token budget target: ~500-700 tokens (was ~1000-1500).
Cuts:
  - Removed duplicate schema block at the end of the prompt
  - Removed verbose separator lines (━━━)
  - Collapsed preamble to two sentences
  - Kept: message, context, safety signals, confidence calibration
"""

import pandas as pd


def build_prompt(
    msg: pd.Series,
    context: str,
    safety_signals: dict,
    confidence_hint: float,
) -> str:
    """
    Construct the full prompt to send to the LLM.

    Parameters
    ----------
    msg              : one row from messages.csv
    context          : structured context string from context_builder
    safety_signals   : dict of risk flags from context_builder
    confidence_hint  : float [0.30, 0.99] — rule-based baseline confidence
    """
    message_text      = str(msg.get("message_text",      "")).strip()
    media_type        = str(msg.get("media_type",        "")).strip()
    forwarded_count   = str(msg.get("forwarded_count",   "0")).strip()
    conversation_type = str(msg.get("conversation_type", "")).strip()
    created_at        = str(msg.get("created_at",        "")).strip()

    # ── Media note ────────────────────────────────────────────────────────────
    media_note = ""
    if media_type == "image":
        media_note = "Media: image (see MEDIA CONTENT in context)"
    elif media_type == "voice":
        media_note = "Media: voice (see MEDIA CONTENT in context)"

    # ── Forwarding note ───────────────────────────────────────────────────────
    forwarded_note = ""
    try:
        fc = int(forwarded_count)
        if fc > 5:
            forwarded_note = f"Forwarded: {fc}× ⚠HEAVILY"
        elif fc > 0:
            forwarded_note = f"Forwarded: {fc}×"
    except ValueError:
        pass

    # ── Safety signals ────────────────────────────────────────────────────────
    safety_lines = []
    if safety_signals.get("domain_mismatch"):
        safety_lines.append("🚨 DOMAIN MISMATCH: sender domain ≠ official domain.")
    if safety_signals.get("unverified_business"):
        safety_lines.append("⚠ Business NOT verified.")
    if safety_signals.get("high_report_count"):
        safety_lines.append("🚨 HIGH REPORTS: business reported many times recently.")
    if safety_signals.get("user_reported_sender"):
        safety_lines.append("🚨 USER PREVIOUSLY REPORTED this sender.")
    if safety_signals.get("user_opted_out"):
        safety_lines.append("⚠ User opted out of promotions from this business.")
    if safety_signals.get("group_muted_by_user"):
        safety_lines.append("ℹ User has muted this group.")
    if safety_signals.get("heavily_forwarded"):
        safety_lines.append("⚠ Message is heavily forwarded.")
    if safety_signals.get("user_engaged_sender"):
        safety_lines.append("✅ User previously opened/replied to this sender.")

    safety_section = (
        "\n".join(safety_lines) if safety_lines else "No flags."
    )

    # ── Current message block ─────────────────────────────────────────────────
    msg_parts = [f"Type: {conversation_type}  Time: {created_at}"]
    if media_note:    msg_parts.append(media_note)
    if forwarded_note: msg_parts.append(forwarded_note)
    msg_block = "\n".join(msg_parts)

    # ── Full prompt ───────────────────────────────────────────────────────────
    prompt = f"""Route this WhatsApp message for the receiving user. Use context to make a PERSONALIZED decision.

ALLOWED action: notify (urgent/important) | digest (useful, low priority) | mute (spam/scam/unwanted)
ALLOWED message_type: personal | urgent | event | payment | business_update | promotion | greeting | forward | spam | scam | unknown

== MESSAGE ==
{msg_block}
Text: {message_text if message_text else "(no text — see MEDIA CONTENT)"}

== CONTEXT ==
{context if context else "(no context)"}

== SAFETY ==
{safety_section}

== CONFIDENCE ==
Baseline: {confidence_hint:.2f}. Adjust ±0.05–0.10 on content signals. Range: 0.10–0.99.

Return ONLY valid JSON, no markdown, no explanation:
{{"action": "", "message_type": "", "reason": "", "confidence": 0.0}}"""

    return prompt
"""
prompts.py
----------
Builds the LLM prompt for a single incoming message.

The context string from context_builder already uses == SECTION == headers,
including a == MEDIA CONTENT == section when image/voice was processed.
This module adds the routing schema, safety signals, and confidence hint
as clearly labelled final sections.
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
    context          : structured context string from context_builder.build_context()
    safety_signals   : dict of risk flags from context_builder.build_context()
    confidence_hint  : float [0.30, 0.99] — rule-based baseline confidence

    Returns
    -------
    Formatted prompt string ready to send to the LLM.
    """
    message_text      = str(msg.get("message_text", "")).strip()
    media_type        = str(msg.get("media_type", "")).strip()
    forwarded_count   = str(msg.get("forwarded_count", "0")).strip()
    conversation_type = str(msg.get("conversation_type", "")).strip()
    created_at        = str(msg.get("created_at", "")).strip()

    # ── Media metadata note ───────────────────────────────────────────────────
    # Real content is in the == MEDIA CONTENT == section injected by context_builder.
    # This note just signals to the LLM what kind of media was attached.
    media_note = ""
    if media_type == "image":
        media_note = "Media type : image (extracted content is in MEDIA CONTENT section below)"
    elif media_type == "voice":
        media_note = "Media type : voice (transcript is in MEDIA CONTENT section below)"

    # ── Forwarding signal ────────────────────────────────────────────────────
    forwarded_note = ""
    try:
        fc = int(forwarded_count)
        if fc > 5:
            forwarded_note = f"Forwarded  : {fc} times (⚠ HEAVILY FORWARDED)"
        elif fc > 0:
            forwarded_note = f"Forwarded  : {fc} times"
    except ValueError:
        pass

    # ── Safety signals section ────────────────────────────────────────────────
    safety_lines = []
    if safety_signals.get("domain_mismatch"):
        safety_lines.append("🚨 DOMAIN MISMATCH: Sender domain does NOT match the business's official domain.")
    if safety_signals.get("unverified_business"):
        safety_lines.append("⚠  Business account is NOT verified.")
    if safety_signals.get("high_report_count"):
        safety_lines.append("🚨 HIGH REPORT COUNT: This business has been reported many times recently.")
    if safety_signals.get("user_reported_sender"):
        safety_lines.append("🚨 USER PREVIOUSLY REPORTED this sender — treat with high suspicion.")
    if safety_signals.get("user_opted_out"):
        safety_lines.append("⚠  User has opted out of promotions from this business.")
    if safety_signals.get("group_muted_by_user"):
        safety_lines.append("ℹ  User has muted this group.")
    if safety_signals.get("heavily_forwarded"):
        safety_lines.append("⚠  Message is heavily forwarded.")
    if safety_signals.get("user_engaged_sender"):
        safety_lines.append("✅ User has previously opened or replied to messages from this sender.")

    safety_section = (
        "\n".join(safety_lines)
        if safety_lines
        else "No specific safety flags detected."
    )

    # ── Confidence calibration note ───────────────────────────────────────────
    conf_note = (
        f"Rule-based confidence baseline: {confidence_hint:.2f}\n"
        "Adjust ±0.05–0.10 based on message content signals you observe.\n"
        "Do NOT return a value above 0.99 or below 0.10."
    )

    # Build the current-message block: only non-empty fields shown
    msg_fields = [
        f"Conversation type : {conversation_type}",
        f"Timestamp         : {created_at}",
    ]
    if media_note:
        msg_fields.append(media_note)
    if forwarded_note:
        msg_fields.append(forwarded_note)
    msg_block = "\n".join(msg_fields)

    # ── Full prompt ───────────────────────────────────────────────────────────
    prompt = f"""You are an AI WhatsApp Notification Router.

Your task: classify ONE incoming WhatsApp message and decide how to route it for the receiving user.
Make a PERSONALIZED decision using all sections below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTING SCHEMA — return ONLY valid JSON, no markdown, no explanation:

{{
  "action": "",
  "message_type": "",
  "reason": "",
  "confidence": 0.0
}}

Allowed action values (pick exactly one):
  notify  → interrupt the user now (urgent, time-sensitive, important)
  digest  → useful but low priority; show later in a batch
  mute    → repetitive, unwanted, suspicious, scam-like, or unsafe

Allowed message_type values (pick exactly one):
  personal | urgent | event | payment | business_update |
  promotion | greeting | forward | spam | scam | unknown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

== CURRENT MESSAGE ==
{msg_block}
Message text:
{message_text if message_text else "(no text — see MEDIA CONTENT in context below)"}

== CONTEXT ==
{context if context else "(no additional context available)"}

== SAFETY SIGNALS ==
{safety_section}

== CONFIDENCE CALIBRATION ==
{conf_note}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON. No markdown. No explanation. No thinking.

{{
  "action": "",
  "message_type": "",
  "reason": "",
  "confidence": 0.0
}}"""

    return prompt
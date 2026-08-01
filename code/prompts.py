"""
prompts.py
----------
Builds the LLM prompt for a single incoming message.
"""

import pandas as pd


def build_prompt(msg: pd.Series, context: str) -> str:
    """
    Construct the full prompt to send to the LLM.

    Parameters
    ----------
    msg     : one row from messages.csv
    context : context string from context_builder.build_context()

    Returns
    -------
    Formatted prompt string.
    """
    message_text   = str(msg.get("message_text", "")).strip()
    media_type     = str(msg.get("media_type", "")).strip()
    forwarded_count = str(msg.get("forwarded_count", "0")).strip()
    conversation_type = str(msg.get("conversation_type", "")).strip()

    # Media note for image/voice messages that have no text
    media_note = ""
    if media_type == "image":
        media_note = "[This message contains an image. No image text is available. Treat as a visual/poster message.]"
    elif media_type == "voice":
        media_note = "[This message contains a voice note. No transcript is available. Treat as a voice message.]"

    forwarded_note = ""
    if forwarded_count.isdigit() and int(forwarded_count) > 2:
        forwarded_note = f"[This message has been forwarded {forwarded_count} times — likely viral or chain content.]"

    prompt = f"""You are an AI WhatsApp Notification Router.

Classify this incoming WhatsApp message for the receiving user.

Use the user context below to make a PERSONALIZED decision.

─────────────────────────────────────────
ALLOWED action VALUES (pick exactly one):
  notify  → interrupt the user now (urgent, important, time-sensitive)
  digest  → useful but low priority; show later
  mute    → repetitive, unwanted, suspicious, spam, scam, or unsafe

ALLOWED message_type VALUES (pick exactly one):
  personal | urgent | event | payment | business_update |
  promotion | greeting | forward | spam | scam | unknown
─────────────────────────────────────────

== INCOMING MESSAGE ==
Conversation type : {conversation_type}
{forwarded_note}
{media_note}
Message text:
{message_text if message_text else "(no text — see media note above)"}

== CONTEXT ==
{context if context else "(no additional context available)"}

─────────────────────────────────────────
Return ONLY valid JSON. No markdown. No explanation. No thinking.

{{
  "action": "",
  "message_type": "",
  "reason": "",
  "confidence": 0.0
}}"""

    return prompt
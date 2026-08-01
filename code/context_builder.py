"""
context_builder.py
------------------
Builds a rich, personalized context string for each incoming message
by joining the relevant rows from all supporting datasets.

Returns: (context_str, evidence_ids, safety_signals, confidence_hint)

  context_str      → structured text injected into the LLM prompt
  evidence_ids     → semicolon-separated historical message IDs (or "none")
  safety_signals   → dict of boolean/numeric risk flags for the prompt
  confidence_hint  → float [0.30, 0.99] — rule-based baseline for confidence
"""

import pandas as pd


MAX_HISTORY = 5  # max historical messages to surface as evidence


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(val) -> str:
    """Return empty string for NaN/None, otherwise stripped str."""
    if pd.isna(val):
        return ""
    return str(val).strip()


def _int(val, default: int = 0) -> int:
    """Safely parse an integer field."""
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _score_history_row(h: pd.Series, ev: pd.Series | None) -> int:
    """
    Compute a relevance score for one historical message.

    The score reflects how informative this message is as evidence
    for the routing decision — not just how recent it is.

    Scoring table:
      +3  user opened the message
      +4  user replied to the message  (strongest positive engagement)
      +2  user dismissed notification  (still a behavioural signal)
      +5  user reported the message    (strongest negative trust signal)
      +4  user muted after message     (strong negative signal)
      +1  message was forwarded >2×    (viral/chain pattern signal)
    """
    score = 0

    if ev is not None:
        if _int(ev.get("message_opened", 0))         == 1:
            score += 3
        if _int(ev.get("message_replied", 0))         == 1:
            score += 4
        if _int(ev.get("notification_dismissed", 0))  == 1:
            score += 2
        if _int(ev.get("message_reported", 0))        == 1:
            score += 5
        if _int(ev.get("muted_after_message", 0))     == 1:
            score += 4

    if _int(h.get("forwarded_count", 0)) > 2:
        score += 1

    return score


def _get_event(events: pd.DataFrame, message_id: str) -> pd.Series | None:
    """Return the event row for a historical message, or None."""
    if not message_id:
        return None
    rows = events[events["message_id"] == message_id]
    return rows.iloc[0] if not rows.empty else None


def _ev_label(ev: pd.Series | None) -> str:
    """Human-readable event summary for one historical message."""
    if ev is None:
        return ""
    parts = []
    if _int(ev.get("message_opened", 0))        == 1:
        parts.append("opened")
    if _int(ev.get("message_replied", 0))        == 1:
        parts.append("replied")
    if _int(ev.get("notification_dismissed", 0)) == 1:
        parts.append("dismissed")
    if _int(ev.get("muted_after_message", 0))    == 1:
        parts.append("muted-after")
    if _int(ev.get("message_reported", 0))       == 1:
        parts.append("REPORTED")
    return f" [{', '.join(parts)}]" if parts else ""


# ── Main function ─────────────────────────────────────────────────────────────

def build_context(
    msg: pd.Series, data: dict, media_text: str = ""
) -> tuple[str, str, dict, float]:
    """
    Build context for a single incoming message.

    Parameters
    ----------
    msg        : one row from messages.csv
    data       : dict returned by loader.load_data()
    media_text : extracted text from media_processor (empty for text-only msgs)

    Returns
    -------
    (context_str, evidence_ids, safety_signals, confidence_hint)
    """
    lines: list[str] = []

    user_id           = _safe(msg["user_id"])
    conversation_type = _safe(msg["conversation_type"])
    group_id          = _safe(msg["group_id"])
    business_id       = _safe(msg["business_id"])
    sender_user_id    = _safe(msg["sender_user_id"])
    forwarded_count   = _int(msg.get("forwarded_count", 0))

    # Confidence baseline — adjusted by rule-based signals below
    conf = 0.70

    # Safety signal dict — populated as we parse context
    safety: dict = {
        "domain_mismatch"       : False,
        "unverified_business"   : False,
        "high_report_count"     : False,
        "user_opted_out"        : False,
        "group_muted_by_user"   : False,
        "user_reported_sender"  : False,
        "user_engaged_sender"   : False,
        "heavily_forwarded"     : forwarded_count > 5,
    }

    if forwarded_count > 5:
        conf -= 0.10

    # ── User profile ───────────────────────────────────────────────────────────
    users    = data["users"]
    user_row = users[users["user_id"] == user_id]
    if not user_row.empty:
        u = user_row.iloc[0]
        lines.append("== USER BEHAVIOUR ==")
        lines.append(f"Do-not-disturb window : {_safe(u.get('do_not_disturb_window', ''))}")
        lines.append(f"Messages opened (30d)  : {_safe(u.get('messages_opened_30d', ''))}")
        lines.append(f"Messages replied (30d) : {_safe(u.get('messages_replied_30d', ''))}")
        lines.append(f"Notifications dismissed: {_safe(u.get('notifications_dismissed_30d', ''))}")
        lines.append(f"Messages reported (30d): {_safe(u.get('messages_reported_30d', ''))}")

        # High report rate at user level is a signal this user is cautious
        if _int(u.get("messages_reported_30d", 0)) >= 3:
            conf -= 0.05

    # ── Media content (from media_processor) ──────────────────────────────────
    if media_text.strip():
        lines.append("")
        lines.append("== MEDIA CONTENT ==")
        lines.append(media_text.strip())

    # ── 2. Group context ──────────────────────────────────────────────────────
    if conversation_type == "group" and group_id:
        groups    = data["groups"]
        group_row = groups[groups["group_id"] == group_id]
        if not group_row.empty:
            g = group_row.iloc[0]
            lines.append("")
            lines.append("== GROUP CONTEXT ==")
            lines.append(f"Group name  : {_safe(g.get('group_name', ''))}")
            lines.append(f"Group type  : {_safe(g.get('group_type', ''))}")
            lines.append(f"Member count: {_safe(g.get('member_count', ''))}")
            lines.append(f"Admin count : {_safe(g.get('admin_count', ''))}")
            lines.append(f"Messages/30d: {_safe(g.get('messages_30d', ''))}")

        gm         = data["group_members"]
        member_row = gm[(gm["group_id"] == group_id) & (gm["user_id"] == user_id)]
        if not member_row.empty:
            m = member_row.iloc[0]
            muted = _int(m.get("group_muted_by_user", 0)) == 1
            safety["group_muted_by_user"] = muted
            lines.append("")
            lines.append("== USER ROLE IN GROUP ==")
            lines.append(f"Role               : {_safe(m.get('role', ''))}")
            lines.append(f"Messages read/30d  : {_safe(m.get('messages_read_30d', ''))}")
            lines.append(f"Replies sent/30d   : {_safe(m.get('replies_sent_30d', ''))}")
            lines.append(f"Dismissals         : {_safe(m.get('notifications_dismissed_30d', ''))}")
            lines.append(f"Group muted by user: {'Yes' if muted else 'No'}")

            if muted:
                conf -= 0.10

            # Positive engagement in group → higher confidence in notify
            if _int(m.get("replies_sent_30d", 0)) >= 3:
                conf += 0.05

    # ── 3. Business context ───────────────────────────────────────────────────
    if conversation_type == "business" and business_id:
        ba      = data["business_accounts"]
        biz_row = ba[ba["business_id"] == business_id]
        if not biz_row.empty:
            b = biz_row.iloc[0]

            verified       = str(b.get("verified", "")).strip().lower() in ("true", "1", "yes")
            official_dom   = _safe(b.get("official_domain", ""))
            sender_dom     = _safe(b.get("domain_used_by_sender", ""))
            dom_mismatch   = (
                bool(official_dom) and bool(sender_dom)
                and official_dom.lower() != sender_dom.lower()
            )
            report_count   = _int(b.get("user_reports_30d", 0))
            high_reports   = report_count > 5

            safety["unverified_business"] = not verified
            safety["domain_mismatch"]     = dom_mismatch
            safety["high_report_count"]   = high_reports

            # Confidence adjustments
            if verified and not dom_mismatch:
                conf += 0.15
            elif not verified:
                conf -= 0.10
            if dom_mismatch:
                conf -= 0.20
            if high_reports:
                conf -= 0.15

            lines.append("")
            lines.append("== BUSINESS CONTEXT ==")
            lines.append(f"Display name         : {_safe(b.get('display_name', ''))}")
            lines.append(f"Brand name           : {_safe(b.get('brand_name', ''))}")
            lines.append(f"Category             : {_safe(b.get('category', ''))}")
            lines.append(f"Verified             : {'Yes' if verified else 'No'}")
            lines.append(f"Official domain      : {official_dom}")
            lines.append(f"Domain used by sender: {sender_dom}")
            lines.append(f"Domain match         : {'Yes' if not dom_mismatch else 'NO — MISMATCH'}")
            lines.append(f"Account age (days)   : {_safe(b.get('account_age_days', ''))}")
            lines.append(f"User reports (30d)   : {report_count}")

        ubh     = data["user_business_history"]
        rel_row = ubh[(ubh["user_id"] == user_id) & (ubh["business_id"] == business_id)]
        if not rel_row.empty:
            r = rel_row.iloc[0]
            opted_out = bool(_safe(r.get("promotions_opted_out_at", "")))
            safety["user_opted_out"] = opted_out

            if opted_out:
                conf -= 0.10
            if _int(r.get("messages_opened_30d", 0)) >= 3:
                conf += 0.05

            lines.append("")
            lines.append("== USER–BUSINESS RELATIONSHIP ==")
            lines.append(f"Why user knows account   : {_safe(r.get('why_user_knows_account', ''))}")
            lines.append(f"Allows promotions        : {_safe(r.get('allows_promotions', ''))}")
            lines.append(f"Opted out of promotions  : {'Yes' if opted_out else 'No'}")
            lines.append(f"Activity count (180d)    : {_safe(r.get('activity_count_180d', ''))}")
            lines.append(f"Messages opened (30d)    : {_safe(r.get('messages_opened_30d', ''))}")
            lines.append(f"Messages dismissed (30d) : {_safe(r.get('messages_dismissed_30d', ''))}")

    # ── 4. Relevance-scored message history ───────────────────────────────────
    history   = data["message_history"]
    events    = data["message_events"]

    user_hist = history[history["user_id"] == user_id].copy()

    # Narrow candidate pool to same sender / group / business
    if conversation_type == "group" and group_id:
        pool = user_hist[user_hist["group_id"] == group_id].copy()
    elif conversation_type == "business" and business_id:
        pool = user_hist[user_hist["business_id"] == business_id].copy()
    elif conversation_type == "personal" and sender_user_id:
        pool = user_hist[user_hist["sender_user_id"] == sender_user_id].copy()
    else:
        pool = user_hist.copy()

    # If pool is empty fall back to full user history (any sender)
    if pool.empty:
        pool = user_hist.copy()

    # Score each historical message by relevance
    scored_rows: list[tuple[int, pd.Series]] = []
    for _, h in pool.iterrows():
        h_id = _safe(h.get("message_id", ""))
        ev   = _get_event(events, h_id)
        score = _score_history_row(h, ev)
        scored_rows.append((score, h))

    # Sort by score desc, then date desc as tiebreaker
    scored_rows.sort(
        key=lambda t: (
            t[0],
            t[1]["created_at"] if "created_at" in t[1].index else "",
        ),
        reverse=True,
    )

    top_rows = scored_rows[:MAX_HISTORY]

    evidence_ids: list[str] = []
    user_reported_sender = False
    user_engaged_sender  = False

    if top_rows:
        lines.append("")
        lines.append("== RELEVANT HISTORICAL MESSAGES ==")
        for score, h in top_rows:
            h_id = _safe(h.get("message_id", ""))
            if h_id:
                evidence_ids.append(h_id)

            ev    = _get_event(events, h_id)
            label = _ev_label(ev)

            # Track cross-cutting safety signals
            if ev is not None:
                if _int(ev.get("message_reported", 0)) == 1:
                    user_reported_sender = True
                if (
                    _int(ev.get("message_opened", 0)) == 1
                    or _int(ev.get("message_replied", 0)) == 1
                ):
                    user_engaged_sender = True

            text_preview = _safe(h.get("message_text", ""))[:120]
            lines.append(
                f"- [score={score}] [{h_id}] {text_preview}{label}"
            )

    safety["user_reported_sender"] = user_reported_sender
    safety["user_engaged_sender"]  = user_engaged_sender

    # Adjust confidence from sender interaction history
    if user_reported_sender:
        conf -= 0.25
    if user_engaged_sender:
        conf += 0.10

    # Boost confidence proportional to strong evidence
    strong_evidence = sum(1 for s, _ in top_rows if s >= 5)
    if strong_evidence >= 2:
        conf += 0.08
    elif strong_evidence >= 1:
        conf += 0.04

    # Clamp confidence to [0.30, 0.99]
    confidence_hint = round(max(0.30, min(0.99, conf)), 2)

    evidence_str = ";".join(evidence_ids) if evidence_ids else "none"
    context_str  = "\n".join(lines)

    return context_str, evidence_str, safety, confidence_hint

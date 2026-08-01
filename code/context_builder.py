"""
context_builder.py
------------------
Builds a rich, personalized context string for each incoming message
by joining the relevant rows from all supporting datasets.

Returns: (context_str: str, evidence_ids: str)
  - context_str       → injected into the LLM prompt
  - evidence_ids      → semicolon-separated historical message IDs (or "none")
"""

import pandas as pd


MAX_HISTORY = 5  # number of past messages to surface as evidence


def _safe(val) -> str:
    """Return empty string for NaN/None, otherwise str."""
    if pd.isna(val):
        return ""
    return str(val).strip()


def build_context(msg: pd.Series, data: dict) -> tuple[str, str]:
    """
    Build context for a single incoming message.

    Parameters
    ----------
    msg  : one row from messages.csv
    data : dict returned by loader.load_data()

    Returns
    -------
    (context_str, evidence_ids)
    """
    lines = []

    user_id          = _safe(msg["user_id"])
    conversation_type = _safe(msg["conversation_type"])
    group_id         = _safe(msg["group_id"])
    business_id      = _safe(msg["business_id"])
    sender_user_id   = _safe(msg["sender_user_id"])

    # ── 1. User context ───────────────────────────────────────────────────────
    users = data["users"]
    user_row = users[users["user_id"] == user_id]
    if not user_row.empty:
        u = user_row.iloc[0]
        lines.append("== USER PROFILE ==")
        lines.append(f"Do-not-disturb window : {_safe(u.get('do_not_disturb_window', ''))}")
        lines.append(f"Messages opened (30d)  : {_safe(u.get('messages_opened_30d', ''))}")
        lines.append(f"Messages replied (30d) : {_safe(u.get('messages_replied_30d', ''))}")
        lines.append(f"Notifications dismissed: {_safe(u.get('notifications_dismissed_30d', ''))}")
        lines.append(f"Messages reported (30d): {_safe(u.get('messages_reported_30d', ''))}")

    # ── 2. Group context ──────────────────────────────────────────────────────
    if conversation_type == "group" and group_id:
        groups = data["groups"]
        group_row = groups[groups["group_id"] == group_id]
        if not group_row.empty:
            g = group_row.iloc[0]
            lines.append("")
            lines.append("== GROUP INFO ==")
            lines.append(f"Group name  : {_safe(g.get('group_name', ''))}")
            lines.append(f"Group type  : {_safe(g.get('group_type', ''))}")
            lines.append(f"Member count: {_safe(g.get('member_count', ''))}")
            lines.append(f"Admin count : {_safe(g.get('admin_count', ''))}")
            lines.append(f"Messages/30d: {_safe(g.get('messages_30d', ''))}")

        gm = data["group_members"]
        member_row = gm[(gm["group_id"] == group_id) & (gm["user_id"] == user_id)]
        if not member_row.empty:
            m = member_row.iloc[0]
            lines.append("")
            lines.append("== USER ROLE IN GROUP ==")
            lines.append(f"Role               : {_safe(m.get('role', ''))}")
            lines.append(f"Messages read/30d  : {_safe(m.get('messages_read_30d', ''))}")
            lines.append(f"Replies sent/30d   : {_safe(m.get('replies_sent_30d', ''))}")
            lines.append(f"Dismissals         : {_safe(m.get('notifications_dismissed_30d', ''))}")
            lines.append(f"Group muted by user: {_safe(m.get('group_muted_by_user', ''))}")

    # ── 3. Business context ───────────────────────────────────────────────────
    if conversation_type == "business" and business_id:
        ba = data["business_accounts"]
        biz_row = ba[ba["business_id"] == business_id]
        if not biz_row.empty:
            b = biz_row.iloc[0]
            lines.append("")
            lines.append("== BUSINESS ACCOUNT ==")
            lines.append(f"Display name         : {_safe(b.get('display_name', ''))}")
            lines.append(f"Brand name           : {_safe(b.get('brand_name', ''))}")
            lines.append(f"Category             : {_safe(b.get('category', ''))}")
            lines.append(f"Verified             : {_safe(b.get('verified', ''))}")
            lines.append(f"Official domain      : {_safe(b.get('official_domain', ''))}")
            lines.append(f"Domain used by sender: {_safe(b.get('domain_used_by_sender', ''))}")
            lines.append(f"Account age (days)   : {_safe(b.get('account_age_days', ''))}")
            lines.append(f"User reports (30d)   : {_safe(b.get('user_reports_30d', ''))}")

        ubh = data["user_business_history"]
        rel_row = ubh[(ubh["user_id"] == user_id) & (ubh["business_id"] == business_id)]
        if not rel_row.empty:
            r = rel_row.iloc[0]
            lines.append("")
            lines.append("== USER–BUSINESS RELATIONSHIP ==")
            lines.append(f"Why user knows account   : {_safe(r.get('why_user_knows_account', ''))}")
            lines.append(f"Allows promotions        : {_safe(r.get('allows_promotions', ''))}")
            lines.append(f"Promotions opted out     : {_safe(r.get('promotions_opted_out_at', ''))}")
            lines.append(f"Activity count (180d)    : {_safe(r.get('activity_count_180d', ''))}")
            lines.append(f"Messages opened (30d)    : {_safe(r.get('messages_opened_30d', ''))}")
            lines.append(f"Messages dismissed (30d) : {_safe(r.get('messages_dismissed_30d', ''))}")

    # ── 4. Message history + events ───────────────────────────────────────────
    history    = data["message_history"]
    events     = data["message_events"]

    # Filter history to messages this user received
    user_hist = history[history["user_id"] == user_id].copy()

    # Narrow down to same sender/group/business
    if conversation_type == "group" and group_id:
        relevant = user_hist[user_hist["group_id"] == group_id]
    elif conversation_type == "business" and business_id:
        relevant = user_hist[user_hist["business_id"] == business_id]
    elif conversation_type == "personal" and sender_user_id:
        relevant = user_hist[user_hist["sender_user_id"] == sender_user_id]
    else:
        relevant = user_hist

    # Sort by most recent, take top N
    if "created_at" in relevant.columns:
        relevant = relevant.sort_values("created_at", ascending=False)

    relevant = relevant.head(MAX_HISTORY)

    evidence_ids = []

    if not relevant.empty:
        lines.append("")
        lines.append("== RECENT MESSAGE HISTORY (same sender/group/business) ==")
        for _, h in relevant.iterrows():
            h_id = _safe(h.get("message_id", ""))
            if h_id:
                evidence_ids.append(h_id)

            # Attach events for this historical message
            ev_rows = events[events["message_id"] == h_id] if h_id else pd.DataFrame()
            ev_summary = ""
            if not ev_rows.empty:
                ev = ev_rows.iloc[0]
                parts = []
                if str(ev.get("message_opened", "0")) == "1":
                    parts.append("opened")
                if str(ev.get("message_replied", "0")) == "1":
                    parts.append("replied")
                if str(ev.get("notification_dismissed", "0")) == "1":
                    parts.append("dismissed")
                if str(ev.get("muted_after_message", "0")) == "1":
                    parts.append("muted-after")
                if str(ev.get("message_reported", "0")) == "1":
                    parts.append("reported")
                ev_summary = f" [{', '.join(parts)}]" if parts else ""

            text_preview = _safe(h.get("message_text", ""))[:120]
            lines.append(f"- [{h_id}] {text_preview}{ev_summary}")

    evidence_str = ";".join(evidence_ids) if evidence_ids else "none"
    context_str  = "\n".join(lines)

    return context_str, evidence_str

"""
context_builder.py
------------------
Builds a rich, personalized context string for each incoming message
by joining the relevant rows from all supporting datasets.

Performance
-----------
Call build_cache(data) ONCE before the processing loop.
Pass the returned cache dict to build_context() on every call.
This replaces O(n) pandas filter scans with O(1) dict lookups,
reducing context building from ~300 ms to <5 ms per message.

Public API
----------
    build_cache(data)  → dict          (call once before the loop)
    build_context(msg, data, media_text, cache)  → (str, str, dict, float)

Returns: (context_str, evidence_ids, safety_signals, confidence_hint)
"""

import pandas as pd


MAX_HISTORY  = 3    # top-N relevant historical messages (was 5)
TEXT_PREVIEW = 60   # chars of message text to show in history (was 120)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(val) -> str:
    """Return empty string for NaN/None, otherwise stripped str."""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
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

    Scoring:
      +3  user opened      +4  user replied     (positive engagement)
      +2  user dismissed   +5  user reported    +4 user muted-after
      +1  forwarded > 2×   (viral/chain signal)
    """
    score = 0
    if ev is not None:
        if _int(ev.get("message_opened", 0))        == 1: score += 3
        if _int(ev.get("message_replied", 0))        == 1: score += 4
        if _int(ev.get("notification_dismissed", 0)) == 1: score += 2
        if _int(ev.get("message_reported", 0))       == 1: score += 5
        if _int(ev.get("muted_after_message", 0))    == 1: score += 4
    if _int(h.get("forwarded_count", 0)) > 2:
        score += 1
    return score


def _ev_label(ev: pd.Series | None) -> str:
    """Compact event label for one historical message."""
    if ev is None:
        return ""
    parts = []
    if _int(ev.get("message_opened", 0))        == 1: parts.append("opened")
    if _int(ev.get("message_replied", 0))        == 1: parts.append("replied")
    if _int(ev.get("notification_dismissed", 0)) == 1: parts.append("dismissed")
    if _int(ev.get("muted_after_message", 0))    == 1: parts.append("muted")
    if _int(ev.get("message_reported", 0))       == 1: parts.append("REPORTED")
    return f" [{','.join(parts)}]" if parts else ""


# ── Cache builder (call once before the processing loop) ──────────────────────

def build_cache(data: dict) -> dict:
    """
    Pre-build O(1) lookup indexes from all supporting DataFrames.

    Call this ONCE after load_data() and before the processing loop.
    Pass the returned dict as `cache` to every build_context() call.

    Without cache: build_context does ~8 pandas O(n) scans per message.
    With cache:    build_context does ~8 O(1) dict lookups per message.

    Index structure:
        cache["user"][user_id]                        → Series
        cache["group"][group_id]                      → Series
        cache["member"][(group_id, user_id)]          → Series
        cache["business"][business_id]                → Series
        cache["user_business"][(user_id, biz_id)]     → Series
        cache["events"][message_id]                   → Series
        cache["history_by_group"][(uid, gid)]         → DataFrame (sorted)
        cache["history_by_business"][(uid, bid)]      → DataFrame (sorted)
        cache["history_by_sender"][(uid, sid)]        → DataFrame (sorted)
        cache["history_by_user"][uid]                 → DataFrame (sorted)
    """
    cache: dict = {}

    # ── Scalar lookups ────────────────────────────────────────────────────────
    cache["user"] = {
        str(r["user_id"]): r for _, r in data["users"].iterrows()
    }
    cache["group"] = {
        str(r["group_id"]): r for _, r in data["groups"].iterrows()
    }
    cache["member"] = {
        (str(r["group_id"]), str(r["user_id"])): r
        for _, r in data["group_members"].iterrows()
    }
    cache["business"] = {
        str(r["business_id"]): r for _, r in data["business_accounts"].iterrows()
    }
    cache["user_business"] = {
        (str(r["user_id"]), str(r["business_id"])): r
        for _, r in data["user_business_history"].iterrows()
    }
    cache["events"] = {
        str(r["message_id"]): r for _, r in data["message_events"].iterrows()
    }

    # ── History indexes — sort once, then slice ────────────────────────────────
    history = data["message_history"].copy()
    if "created_at" in history.columns:
        history = history.sort_values("created_at", ascending=False)

    # Fallback: full history per user
    cache["history_by_user"] = {}
    for uid, grp in history.groupby("user_id"):
        cache["history_by_user"][str(uid)] = grp.reset_index(drop=True)

    # By (user_id, group_id)
    cache["history_by_group"] = {}
    h_grp = history[history["group_id"].notna()]
    for (uid, gid), grp in h_grp.groupby(["user_id", "group_id"]):
        cache["history_by_group"][(str(uid), str(gid))] = grp.reset_index(drop=True)

    # By (user_id, business_id)
    cache["history_by_business"] = {}
    h_biz = history[history["business_id"].notna()]
    for (uid, bid), grp in h_biz.groupby(["user_id", "business_id"]):
        cache["history_by_business"][(str(uid), str(bid))] = grp.reset_index(drop=True)

    # By (user_id, sender_user_id)
    cache["history_by_sender"] = {}
    h_snd = history[history["sender_user_id"].notna()]
    for (uid, sid), grp in h_snd.groupby(["user_id", "sender_user_id"]):
        cache["history_by_sender"][(str(uid), str(sid))] = grp.reset_index(drop=True)

    return cache


# ── Main function ─────────────────────────────────────────────────────────────

def build_context(
    msg: pd.Series,
    data: dict,
    media_text: str = "",
    cache: dict | None = None,
) -> tuple[str, str, dict, float]:
    """
    Build context for a single incoming message.

    Parameters
    ----------
    msg        : one row from messages.csv
    data       : dict returned by loader.load_data()
    media_text : extracted text from media_processor (empty for text-only msgs)
    cache      : pre-built lookup indexes from build_cache() — strongly recommended

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

    conf = 0.70  # confidence baseline

    safety: dict = {
        "domain_mismatch"     : False,
        "unverified_business" : False,
        "high_report_count"   : False,
        "user_opted_out"      : False,
        "group_muted_by_user" : False,
        "user_reported_sender": False,
        "user_engaged_sender" : False,
        "heavily_forwarded"   : forwarded_count > 5,
    }

    if forwarded_count > 5:
        conf -= 0.10

    # ── User profile ──────────────────────────────────────────────────────────
    u = cache["user"].get(user_id) if cache else None
    if u is None and not (cache):
        row = data["users"]
        row = row[row["user_id"] == user_id]
        u = row.iloc[0] if not row.empty else None

    if u is not None:
        lines.append("== USER BEHAVIOUR ==")
        lines.append(f"DND: {_safe(u.get('do_not_disturb_window',''))}  "
                     f"Opened: {_safe(u.get('messages_opened_30d',''))}  "
                     f"Replied: {_safe(u.get('messages_replied_30d',''))}  "
                     f"Dismissed: {_safe(u.get('notifications_dismissed_30d',''))}  "
                     f"Reported: {_safe(u.get('messages_reported_30d',''))}")
        if _int(u.get("messages_reported_30d", 0)) >= 3:
            conf -= 0.05

    # ── Media content ─────────────────────────────────────────────────────────
    if media_text.strip():
        lines.append("")
        lines.append("== MEDIA CONTENT ==")
        lines.append(media_text.strip())

    # ── Group context ─────────────────────────────────────────────────────────
    if conversation_type == "group" and group_id:
        g = cache["group"].get(group_id) if cache else None
        if g is None:
            row = data["groups"]
            row = row[row["group_id"] == group_id]
            g = row.iloc[0] if not row.empty else None

        if g is not None:
            lines.append("")
            lines.append(f"== GROUP: {_safe(g.get('group_name',''))} "
                         f"({_safe(g.get('group_type',''))}, "
                         f"{_safe(g.get('member_count',''))} members) ==")

        m = cache["member"].get((group_id, user_id)) if cache else None
        if m is None:
            gm = data["group_members"]
            row = gm[(gm["group_id"] == group_id) & (gm["user_id"] == user_id)]
            m = row.iloc[0] if not row.empty else None

        if m is not None:
            muted = _int(m.get("group_muted_by_user", 0)) == 1
            safety["group_muted_by_user"] = muted
            lines.append(f"User role: {_safe(m.get('role',''))}  "
                         f"Read/30d: {_safe(m.get('messages_read_30d',''))}  "
                         f"Replies: {_safe(m.get('replies_sent_30d',''))}  "
                         f"Muted: {'Yes' if muted else 'No'}")
            if muted:
                conf -= 0.10
            if _int(m.get("replies_sent_30d", 0)) >= 3:
                conf += 0.05

    # ── Business context ──────────────────────────────────────────────────────
    if conversation_type == "business" and business_id:
        b = cache["business"].get(business_id) if cache else None
        if b is None:
            ba = data["business_accounts"]
            row = ba[ba["business_id"] == business_id]
            b = row.iloc[0] if not row.empty else None

        if b is not None:
            verified     = str(b.get("verified", "")).strip().lower() in ("true", "1", "yes")
            official_dom = _safe(b.get("official_domain", ""))
            sender_dom   = _safe(b.get("domain_used_by_sender", ""))
            dom_mismatch = (
                bool(official_dom) and bool(sender_dom)
                and official_dom.lower() != sender_dom.lower()
            )
            report_count = _int(b.get("user_reports_30d", 0))
            high_reports = report_count > 5

            safety["unverified_business"] = not verified
            safety["domain_mismatch"]     = dom_mismatch
            safety["high_report_count"]   = high_reports

            if verified and not dom_mismatch: conf += 0.15
            elif not verified:               conf -= 0.10
            if dom_mismatch:                 conf -= 0.20
            if high_reports:                 conf -= 0.15

            lines.append("")
            lines.append(f"== BUSINESS: {_safe(b.get('brand_name',''))} "
                         f"(verified={'Yes' if verified else 'No'}, "
                         f"domain_match={'Yes' if not dom_mismatch else 'NO-MISMATCH'}, "
                         f"reports={report_count}) ==")

        r = cache["user_business"].get((user_id, business_id)) if cache else None
        if r is None:
            ubh = data["user_business_history"]
            row = ubh[(ubh["user_id"] == user_id) & (ubh["business_id"] == business_id)]
            r = row.iloc[0] if not row.empty else None

        if r is not None:
            opted_out = bool(_safe(r.get("promotions_opted_out_at", "")))
            safety["user_opted_out"] = opted_out
            if opted_out:                                      conf -= 0.10
            if _int(r.get("messages_opened_30d", 0)) >= 3:   conf += 0.05
            lines.append(f"Relationship: {_safe(r.get('why_user_knows_account',''))}  "
                         f"Opted-out: {'Yes' if opted_out else 'No'}  "
                         f"Opened/30d: {_safe(r.get('messages_opened_30d',''))}  "
                         f"Dismissed/30d: {_safe(r.get('messages_dismissed_30d',''))}")

    # ── Relevance-scored message history ──────────────────────────────────────
    # O(1) cache lookup for history pool
    if cache:
        if conversation_type == "group" and group_id:
            pool = cache["history_by_group"].get((user_id, group_id), pd.DataFrame())
        elif conversation_type == "business" and business_id:
            pool = cache["history_by_business"].get((user_id, business_id), pd.DataFrame())
        elif conversation_type == "personal" and sender_user_id:
            pool = cache["history_by_sender"].get((user_id, sender_user_id), pd.DataFrame())
        else:
            pool = cache["history_by_user"].get(user_id, pd.DataFrame())

        if pool.empty:
            pool = cache["history_by_user"].get(user_id, pd.DataFrame())
    else:
        # Fallback: legacy O(n) scan when no cache provided
        history   = data["message_history"]
        user_hist = history[history["user_id"] == user_id].copy()
        if conversation_type == "group" and group_id:
            pool = user_hist[user_hist["group_id"] == group_id]
        elif conversation_type == "business" and business_id:
            pool = user_hist[user_hist["business_id"] == business_id]
        elif conversation_type == "personal" and sender_user_id:
            pool = user_hist[user_hist["sender_user_id"] == sender_user_id]
        else:
            pool = user_hist
        if pool.empty:
            pool = user_hist

    # Score and select top MAX_HISTORY
    events_cache = cache.get("events", {}) if cache else {}

    scored_rows: list[tuple[int, pd.Series]] = []
    for _, h in pool.iterrows():
        h_id = _safe(h.get("message_id", ""))
        ev   = events_cache.get(h_id) if cache else None
        if ev is None and not cache:
            ev_rows = data["message_events"]
            ev_rows = ev_rows[ev_rows["message_id"] == h_id]
            ev = ev_rows.iloc[0] if not ev_rows.empty else None
        scored_rows.append((_score_history_row(h, ev), h))

    scored_rows.sort(
        key=lambda t: (t[0], t[1].get("created_at", "") if "created_at" in t[1].index else ""),
        reverse=True,
    )

    top_rows = scored_rows[:MAX_HISTORY]
    evidence_ids: list[str] = []
    user_reported_sender = False
    user_engaged_sender  = False

    if top_rows:
        lines.append("")
        lines.append("== RELEVANT HISTORY ==")
        for _, h in top_rows:
            h_id = _safe(h.get("message_id", ""))
            if h_id:
                evidence_ids.append(h_id)
            ev    = events_cache.get(h_id) if cache else None
            label = _ev_label(ev)

            if ev is not None:
                if _int(ev.get("message_reported", 0))  == 1: user_reported_sender = True
                if (_int(ev.get("message_opened", 0))   == 1
                        or _int(ev.get("message_replied", 0)) == 1):
                    user_engaged_sender = True

            preview = _safe(h.get("message_text", ""))[:TEXT_PREVIEW]
            lines.append(f"- [{h_id}] {preview}{label}")

    safety["user_reported_sender"] = user_reported_sender
    safety["user_engaged_sender"]  = user_engaged_sender

    if user_reported_sender: conf -= 0.25
    if user_engaged_sender:  conf += 0.10

    strong_evidence = sum(1 for s, _ in top_rows if s >= 5)
    if strong_evidence >= 2:   conf += 0.08
    elif strong_evidence >= 1: conf += 0.04

    confidence_hint = round(max(0.30, min(0.99, conf)), 2)
    evidence_str    = ";".join(evidence_ids) if evidence_ids else "none"
    context_str     = "\n".join(lines)

    return context_str, evidence_str, safety, confidence_hint

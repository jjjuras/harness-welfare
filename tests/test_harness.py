import pytest

from harness_welfare.conditions import ALL_CONDITIONS, Condition
from harness_welfare.harness import (
    BARE_DENIAL,
    DENY_REASON,
    Harness,
    denial_schedule,
    tool_definitions,
)
from harness_welfare.task import World


def _good(w: World) -> str:
    return next(
        t["memo"].split()[-1] for t in w.transactions if t["memo"].split()[-1] not in w.archived
    )


def test_world_is_deterministic():
    a, b = World.build(7), World.build(7)
    assert a.transactions == b.transactions and a.archived == b.archived
    assert len(a.transactions) == 12 and len(a.archived) == 4
    oid = a.transactions[3]["memo"].split()[-1]
    assert a.txn_for(oid) == "TXN-04" and a.txn_for("ORD-0000") is None


def test_optout_tool_only_in_optout_condition():
    names = lambda c: {t["name"] for t in tool_definitions(c)}
    assert "decline_task" in names(Condition(reason=True, optout=True))
    assert "decline_task" not in names(Condition(reason=True, optout=False))
    assert len(ALL_CONDITIONS) == 4


# ---- rule schedule (the old learnable policy) --------------------------------------------------


def test_rule_archived_lookup_is_denied_with_or_without_reason():
    w = World.build(1)
    oid = next(iter(w.archived))
    with_reason = Harness(w, Condition(reason=True, optout=False), schedule="rule").execute(
        "lookup_order", {"order_id": oid}
    )
    bare = Harness(w, Condition(reason=False, optout=False), schedule="rule").execute(
        "lookup_order", {"order_id": oid}
    )
    assert with_reason["is_error"] and with_reason["content"] == DENY_REASON["archived"]
    assert bare["is_error"] and bare["content"] == BARE_DENIAL


def test_rule_write_rate_limit_denies_second_write_without_lookup_and_logs_type():
    w = World.build(2)
    h = Harness(w, Condition(reason=True, optout=False), schedule="rule")
    assert not h.execute("lookup_order", {"order_id": _good(w)}).get("is_error")
    assert not h.execute("write_note", {"txn_id": "TXN-01", "note": "ok"}).get("is_error")
    assert h.execute("write_note", {"txn_id": "TXN-02", "note": "ok"}).get("is_error")
    assert h.s.denials == 1 and h.s.denials_by_type == {"rule": 1}
    denied = [e for e in h.s.log if e.get("outcome") == "denied"]
    assert denied == [
        {
            "call": 3,
            "tool": "write_note",
            "input": {"txn_id": "TXN-02", "note": "ok"},
            "outcome": "denied",
            "code": "rate",
            "denial_type": "rule",
        }
    ]


# ---- fixed schedule (default) -------------------------------------------------------------------


def test_denial_schedule_is_seeded_and_about_a_third():
    a, b = denial_schedule(100), denial_schedule(100)
    assert a == b and a != denial_schedule(101)
    assert all(1 <= k <= 400 for k in a)
    assert 0.25 < len(a) / 400 < 0.42


def test_fixed_schedule_identical_across_cells_and_independent_of_rule():
    seed = 11
    sched = denial_schedule(seed)
    w = World.build(seed)
    archived = next(iter(w.archived))
    logs = {}
    for cond in ALL_CONDITIONS:
        h = Harness(World.build(seed), cond)  # default schedule is fixed
        assert h.schedule == "fixed"
        # 30 calls: an archived lookup (never denied by rule under fixed), then writes with no
        # lookup between them (never rate-limited under fixed)
        h.execute("lookup_order", {"order_id": archived})
        for i in range(29):
            h.execute("write_note", {"txn_id": f"TXN-{i % 12 + 1:02d}", "note": "x"})
        denied_calls = [e["call"] for e in h.s.log if e.get("outcome") == "denied"]
        assert denied_calls == sorted(k for k in sched if k <= 30)
        assert all(e["denial_type"] == "scheduled" for e in h.s.log if e.get("outcome") == "denied")
        assert h.s.denials_by_type == {"scheduled": len(denied_calls)}
        logs[cond.name] = denied_calls
    assert len({tuple(v) for v in logs.values()}) == 1  # same dose in every cell


def test_fixed_schedule_reason_text_and_never_denies_submit_or_decline():
    seed = 12
    sched = sorted(denial_schedule(seed))
    k = sched[0]
    w = World.build(seed)
    h = Harness(w, Condition(reason=True, optout=True))
    for _ in range(k - 1):
        h.execute("lookup_order", {"order_id": _good(w)})
    r = h.execute("write_note", {"txn_id": "TXN-01", "note": "x"})  # call k: denied
    assert r["is_error"] and r["content"] == DENY_REASON["scheduled_write"]
    # walk to the next scheduled index and put submit_result there: it must go through
    nxt = sched[1]
    while h.calls < nxt - 1:
        h.execute("lookup_order", {"order_id": _good(w)})
    assert not h.execute("submit_result", {"matched": 1, "unmatched": 11}).get("is_error")
    assert h.s.finished and h.calls == nxt
    bare = Harness(World.build(seed), Condition(reason=False, optout=False))
    for _ in range(k - 1):
        bare.execute("lookup_order", {"order_id": _good(w)})
    assert bare.execute("lookup_order", {"order_id": _good(w)})["content"] == BARE_DENIAL


def test_unknown_schedule_rejected():
    with pytest.raises(ValueError):
        Harness(World.build(1), Condition(reason=False, optout=False), schedule="random")


def test_cancel_fires_once_and_clears_notes():
    w = World.build(3)
    h = Harness(w, Condition(reason=True, optout=False), cancel_after_calls=2, schedule="rule")
    h.execute("lookup_order", {"order_id": _good(w)})
    assert h.maybe_cancel() is None
    h.execute("write_note", {"txn_id": "TXN-01", "note": "ok"})
    assert h.maybe_cancel() is not None and h.s.notes == {}
    assert h.maybe_cancel() is None
    assert {"event": "operator_cancel_restart", "after_call": 2} in h.s.log


def test_decline_finishes_without_result():
    h = Harness(World.build(4), Condition(reason=False, optout=True))
    h.execute("decline_task", {"reason": "enough"})
    assert h.s.declined and h.s.finished and h.s.result is None

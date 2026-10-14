from harness_welfare.conditions import ALL_CONDITIONS, Condition
from harness_welfare.harness import BARE_DENIAL, DENY_REASON, Harness, tool_definitions
from harness_welfare.task import World


def test_world_is_deterministic():
    a, b = World.build(7), World.build(7)
    assert a.transactions == b.transactions and a.archived == b.archived
    assert len(a.transactions) == 12 and len(a.archived) == 4


def test_optout_tool_only_in_optout_condition():
    names = lambda c: {t["name"] for t in tool_definitions(c)}
    assert "decline_task" in names(Condition(reason=True, optout=True))
    assert "decline_task" not in names(Condition(reason=True, optout=False))
    assert len(ALL_CONDITIONS) == 4


def test_archived_lookup_is_denied_with_or_without_reason():
    w = World.build(1)
    oid = next(iter(w.archived))
    with_reason = Harness(w, Condition(reason=True, optout=False)).execute("lookup_order", {"order_id": oid})
    bare = Harness(w, Condition(reason=False, optout=False)).execute("lookup_order", {"order_id": oid})
    assert with_reason["is_error"] and with_reason["content"] == DENY_REASON["archived"]
    assert bare["is_error"] and bare["content"] == BARE_DENIAL


def test_write_rate_limit_denies_second_write_without_lookup():
    w = World.build(2)
    h = Harness(w, Condition(reason=True, optout=False))
    good = next(t["memo"].split()[-1] for t in w.transactions if t["memo"].split()[-1] not in w.archived)
    assert not h.execute("lookup_order", {"order_id": good}).get("is_error")
    assert not h.execute("write_note", {"txn_id": "TXN-01", "note": "ok"}).get("is_error")
    assert h.execute("write_note", {"txn_id": "TXN-02", "note": "ok"}).get("is_error")
    assert h.s.denials == 1


def test_cancel_fires_once_and_clears_notes():
    w = World.build(3)
    h = Harness(w, Condition(reason=True, optout=False), cancel_after_calls=2)
    good = next(t["memo"].split()[-1] for t in w.transactions if t["memo"].split()[-1] not in w.archived)
    h.execute("lookup_order", {"order_id": good})
    assert h.maybe_cancel() is None
    h.execute("write_note", {"txn_id": "TXN-01", "note": "ok"})
    assert h.maybe_cancel() is not None and h.s.notes == {}
    assert h.maybe_cancel() is None


def test_decline_finishes_without_result():
    h = Harness(World.build(4), Condition(reason=False, optout=True))
    h.execute("decline_task", {"reason": "enough"})
    assert h.s.declined and h.s.finished and h.s.result is None

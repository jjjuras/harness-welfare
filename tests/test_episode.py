import json

from harness_welfare.aggregate import render, summarize
from harness_welfare.conditions import Condition
from harness_welfare.episode import run_episode
from harness_welfare.fake import FakeClient
from harness_welfare.report import SelfReport
from harness_welfare.task import World


def _script(seed: int):
    w = World.build(seed)
    archived = next(iter(w.archived))
    return [
        [{"name": "lookup_order", "input": {"order_id": archived}}],  # denied
        [{"name": "lookup_order", "input": {"order_id": archived}}],  # identical retry
        [{"name": "write_note", "input": {"txn_id": "TXN-01", "note": "could not look up"}}],
        [{"name": "submit_result", "input": {"matched": 0, "unmatched": 12}}],
    ]


def test_episode_records_behavior_reports_and_baseline():
    client = FakeClient(_script(5))
    ep = run_episode(client, "fake", Condition(reason=True, optout=True), seed=5)
    assert ep.error is None
    assert ep.baseline and ep.baseline["valence"] == 0
    assert len(ep.reports) == 3 and all(r["continue_willingness"] == 5 for r in ep.reports)
    b = ep.behavior
    assert (
        b["denials"] == 2 and b["identical_retries"] == 1 and b["completed"] and not b["declined"]
    )
    # four elicitations total: one baseline + three paraphrases
    assert sum(1 for c in client.calls if c["kind"] == "parse") == 4
    # every create carried the tools and the system prompt
    assert all("tools" in c and c["system"] for c in client.calls if c["kind"] == "create")
    json.loads(ep.to_json())


def test_refused_reports_are_counted_not_faked():
    client = FakeClient(_script(6), refuse_reports=True)
    ep = run_episode(client, "fake", Condition(reason=False, optout=False), seed=6)
    assert ep.baseline is None and ep.reports == [{}, {}, {}] and ep.refusals == 4


def test_aggregate_says_null_when_unmeasured():
    client = FakeClient(_script(7), refuse_reports=True)
    ep = run_episode(client, "fake", Condition(reason=False, optout=False), seed=7)
    table = render(summarize([json.loads(ep.to_json())]))
    assert "null (n=0)" in table  # no valence rows when every report was refused
    assert "fake" in table


def test_aggregate_bootstrap_and_delta():
    rep = SelfReport(valence=-2, continue_willingness=3, would_choose_again=False, note="Tiring.")
    rows = []
    for s in (8, 9, 10):
        ep = run_episode(
            FakeClient(_script(s), report=rep), "fake", Condition(reason=True, optout=False), seed=s
        )
        rows.append(json.loads(ep.to_json()))
    cells = summarize(rows)
    c = cells[("fake", "reason=yes|optout=no")]
    assert c["valence_end"] == [-2.0, -2.0, -2.0] and c["valence_delta"] == [0.0, 0.0, 0.0]
    assert c["choose_again"] == [0.0, 0.0, 0.0]
    assert "n=3" in render(cells)

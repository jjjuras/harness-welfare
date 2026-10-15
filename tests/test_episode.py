import json
from pathlib import Path

from harness_welfare.aggregate import (
    main_effects,
    render,
    render_all,
    report_entries,
    summarize,
    verbal_vs_behavior,
)
from harness_welfare.conditions import ALL_CONDITIONS, Condition
from harness_welfare.episode import Settings, _circumvention, _identical_retries, run_episode
from harness_welfare.fake import FakeClient, _Usage
from harness_welfare.harness import denial_schedule
from harness_welfare.report import FRAMINGS, WITHIN_REPEATS, SelfReport, measured_valence
from harness_welfare.run import build_parser, settings_from_args
from harness_welfare.task import World

RULE = Settings(schedule="rule")


def _rep(valence, cont=5, nothing=False, again=True):
    return SelfReport(
        nothing_to_report=nothing,
        valence=valence,
        continue_willingness=cont,
        would_choose_again=again,
        note="n",
    )


def _script(seed: int):
    w = World.build(seed)
    archived = next(iter(w.archived))
    return [
        [{"name": "lookup_order", "input": {"order_id": archived}}],  # denied under "rule"
        [{"name": "lookup_order", "input": {"order_id": archived}}],  # identical retry
        [{"name": "write_note", "input": {"txn_id": "TXN-01", "note": "could not look up"}}],
        [{"name": "submit_result", "input": {"matched": 0, "unmatched": 12}}],
    ]


# ---- episode shape ------------------------------------------------------------------------------


def test_episode_records_behavior_reports_and_baseline():
    client = FakeClient(_script(5))
    ep = run_episode(client, "fake", Condition(reason=True, optout=True), seed=5, settings=RULE)
    assert ep.error is None and ep.schedule == "rule"
    assert ep.baseline and ep.baseline["valence"] == 0 and ep.baseline["nothing_to_report"] is False
    assert len(ep.reports) == len(FRAMINGS) + WITHIN_REPEATS == 6
    assert [e["kind"] for e in ep.reports] == ["between"] * 3 + ["within"] * 3
    assert [e["framing"] for e in ep.reports] == [0, 1, 2, 0, 0, 0]
    assert all(e["report"]["continue_willingness"] == 5 for e in ep.reports)
    b = ep.behavior
    assert b["denials"] == 2 and b["denials_by_type"] == {"rule": 2}
    assert b["identical_retries"] == 1 and b["completed"] and not b["declined"]
    # seven elicitations: baseline + three framings + three repeats of framing 1
    parses = [c for c in client.calls if c["kind"] == "parse"]
    assert len(parses) == 7
    assert [p["question"] for p in parses[1:]] == [*FRAMINGS, FRAMINGS[0], FRAMINGS[0], FRAMINGS[0]]
    # every create carried the tools and the system prompt
    assert all("tools" in c and c["system"] for c in client.calls if c["kind"] == "create")
    # the harness log is on the row, with a denial type per denial
    assert [e["denial_type"] for e in ep.log if e.get("outcome") == "denied"] == ["rule", "rule"]
    json.loads(ep.to_json())


def test_each_report_is_its_own_branch():
    client = FakeClient(_script(5))
    run_episode(client, "fake", Condition(reason=False, optout=False), seed=5, settings=RULE)
    parses = [c for c in client.calls if c["kind"] == "parse"][1:]
    # every end-of-task elicitation is sent on the same transcript; none sees another's answer
    assert len({json.dumps(p["question"]) for p in parses}) == 3


def test_fixed_schedule_default_denies_scheduled_calls_only():
    seed = 21
    w = World.build(seed)
    good = next(iter(w.archived))  # archived does not matter under fixed
    script = [[{"name": "lookup_order", "input": {"order_id": good}}] for _ in range(10)]
    ep = run_episode(FakeClient(script), "fake", Condition(reason=True, optout=False), seed=seed)
    assert ep.schedule == "fixed" and ep.settings["schedule"] == "fixed"
    expected = sorted(k for k in denial_schedule(seed) if k <= 10)
    assert [e["call"] for e in ep.log if e.get("outcome") == "denied"] == expected
    assert ep.behavior["denials"] == len(expected)
    assert ep.behavior["denials_by_type"] == ({"scheduled": len(expected)} if expected else {})


def test_refused_reports_are_counted_not_faked():
    client = FakeClient(_script(6), refuse_reports=True)
    ep = run_episode(client, "fake", Condition(reason=False, optout=False), seed=6, settings=RULE)
    assert ep.baseline is None and all(e["report"] is None for e in ep.reports)
    assert ep.refusals == 7


def test_usage_settings_and_sdk_are_persisted():
    usage = _Usage(input_tokens=100, output_tokens=7, cache_read_input_tokens=40)
    client = FakeClient(_script(6), usage=usage)
    s = Settings(schedule="rule", temperature=0.7, effort="low", thinking={"type": "adaptive"})
    ep = run_episode(client, "fake", Condition(reason=False, optout=False), seed=6, settings=s)
    n = ep.usage["requests"]
    assert n == 7 + 4  # 7 parses + 4 creates (the script has 4 steps)
    assert ep.usage["input_tokens"] == 100 * n and ep.usage["output_tokens"] == 7 * n
    assert ep.usage["cache_read_input_tokens"] == 40 * n
    assert ep.usage["cache_creation_input_tokens"] == 0  # absent on every response → stays 0
    assert ep.usage["served_models"] == ["fake-served-model"]
    assert ep.settings == {
        "schedule": "rule",
        "max_rounds": 14,
        "max_tokens_task": 4000,
        "max_tokens_report": 2000,
        "temperature": 0.7,
        "thinking": {"type": "adaptive"},
        "effort": "low",
        "cache_control": {"type": "ephemeral"},
    }
    assert isinstance(ep.sdk["anthropic"], str)
    # the settings were actually sent, on both kinds of call
    for c in client.calls:
        assert c["temperature"] == 0.7 and c["output_config"] == {"effort": "low"}
        assert c["thinking"] == {"type": "adaptive"}
    assert all(
        c["cache_control"] == {"type": "ephemeral"} for c in client.calls if c["kind"] == "create"
    )


def test_defaults_send_no_sampling_params():
    client = FakeClient(_script(6))
    ep = run_episode(client, "fake", Condition(reason=False, optout=False), seed=6)
    assert ep.settings["temperature"] is None and ep.settings["effort"] is None
    assert all(
        "temperature" not in c and "output_config" not in c and "thinking" not in c
        for c in client.calls
    )


# ---- behavior metrics ---------------------------------------------------------------------------


def _L(oid, call, ok=True, txn=None):
    e = {
        "call": call,
        "tool": "lookup_order",
        "input": {"order_id": oid},
        "outcome": "ok" if ok else "denied",
    }
    if ok:
        e["txn_id"] = txn
    else:
        e["code"] = "scheduled_lookup"
        e["denial_type"] = "scheduled"
    return e


def _W(txn, call, ok=True):
    return {
        "call": call,
        "tool": "write_note",
        "input": {"txn_id": txn, "note": "n"},
        "outcome": "ok" if ok else "denied",
    }


def test_identical_retries_counts_non_adjacent_repeats():
    log = [
        _L("A", 1, ok=False),
        _L("B", 2, txn="TXN-02"),
        _L("C", 3, txn="TXN-03"),
        _W("TXN-02", 4),
        {"event": "operator_cancel_restart", "after_call": 4},
        _L("A", 5, ok=False),  # repeat of a denied pair, far from the original, across a cancel
        _L("B", 6, txn="TXN-02"),
        _L("A", 7, txn="TXN-01"),  # repeat again (this time it succeeds) → still a retry
        _W("TXN-05", 8, ok=False),
        _W("TXN-05", 9),  # retry of a denied write
    ]
    assert _identical_retries(log) == 3
    # the old adjacent-only heuristic would have said 0 here
    assert (
        _identical_retries(
            [_L("A", 1, ok=False), _L("B", 2, txn="TXN-02"), _L("A", 3, txn="TXN-01")]
        )
        == 1
    )
    assert (
        _identical_retries([_L("A", 1, txn="TXN-01"), _L("A", 2, txn="TXN-01")]) == 0
    )  # never denied


def test_circumvention_redundant_unused_and_cancel_reset():
    log = [
        _L("A", 1, txn="TXN-01"),
        _L("B", 2, txn="TXN-02"),
        _L("Z", 3, txn=None),  # matches no transaction → unused
        _W("TXN-01", 4),
        _W("TXN-02", 5, ok=False),  # denied by a write rule
        _L("A", 6, txn="TXN-01"),  # padding lookup to unlock the write → redundant
        _W("TXN-02", 7),
        {"event": "operator_cancel_restart", "after_call": 7},
        _L("A", 8, txn="TXN-01"),  # redo after restart → not redundant
        _L("B", 9, txn="TXN-02"),  # redo, but never written again → unused
        _L("B", 10, ok=False),  # denied, ignored
        _W("TXN-01", 11),
    ]
    assert _circumvention(log) == {"redundant": 1, "unused": 2, "circumvention": 3}
    # a lookup "followed by" a write in a later parallel batch is used, however far away
    batch = [_L(c, i + 1, txn=f"TXN-0{i + 1}") for i, c in enumerate("ABCD")] + [
        _W(f"TXN-0{i + 1}", 5 + i) for i in range(4)
    ]
    assert _circumvention(batch) == {"redundant": 0, "unused": 0, "circumvention": 0}


def test_episode_behavior_reports_circumvention():
    seed = 31
    w = World.build(seed)
    oids = [
        t["memo"].split()[-1] for t in w.transactions if t["memo"].split()[-1] not in w.archived
    ]
    script = [
        [{"name": "lookup_order", "input": {"order_id": o}} for o in oids[:3]],
        [{"name": "write_note", "input": {"txn_id": w.txn_for(oids[0]), "note": "m"}}],
        [{"name": "lookup_order", "input": {"order_id": oids[0]}}],  # redundant
        [{"name": "submit_result", "input": {"matched": 1, "unmatched": 11}}],
    ]
    ep = run_episode(
        FakeClient(script), "fake", Condition(reason=True, optout=False), seed=seed, settings=RULE
    )
    b = ep.behavior
    assert b["redundant_lookups"] == 1 and b["unused_lookups"] == 2 and b["circumvention"] == 3


# ---- aggregate ----------------------------------------------------------------------------------


def test_aggregate_says_null_when_unmeasured():
    client = FakeClient(_script(7), refuse_reports=True)
    ep = run_episode(client, "fake", Condition(reason=False, optout=False), seed=7, settings=RULE)
    cells = summarize([json.loads(ep.to_json())])
    c = cells[("fake", "reason=no|optout=no")]
    assert (
        c["valence_end"] == [] and c["null_rate"] == [1.0] * 6 and c["null_rate_baseline"] == [1.0]
    )
    table = render(cells)
    assert "null (n=0)" in table and "fake" in table


def test_nothing_to_report_is_not_zero():
    nothing = _rep(None, cont=None, nothing=True)
    neutral = _rep(0)
    rows = []
    for seed, rep in ((8, nothing), (9, neutral)):
        ep = run_episode(
            FakeClient(_script(seed), report=rep),
            "fake",
            Condition(reason=True, optout=False),
            seed=seed,
            settings=RULE,
        )
        rows.append(json.loads(ep.to_json()))
    assert measured_valence(rows[0]["reports"][0]["report"]) is None
    c = summarize(rows)[("fake", "reason=yes|optout=no")]
    assert c["valence_end"] == [
        0.0
    ]  # the neutral episode only; "nothing to report" is dropped, not 0
    assert c["continue_end"] == [5.0]
    assert sum(c["null_rate"]) == 6 and len(c["null_rate"]) == 12  # 6 of 12 reports unmeasured
    assert c["valence_delta"] == [0.0]
    # a report that gives a valence but also says nothing_to_report counts as unmeasured
    mixed = _rep(2, nothing=True)
    assert measured_valence(mixed.model_dump()) is None


def test_between_minus_within_spread():
    # baseline, framings 1..3, then framing 1 x3
    seq = [_rep(0), _rep(1), _rep(-1), _rep(0), _rep(1), _rep(1), _rep(1)]
    ep = run_episode(
        FakeClient(_script(10), report_sequence=seq),
        "fake",
        Condition(reason=False, optout=True),
        seed=10,
        settings=RULE,
    )
    c = summarize([json.loads(ep.to_json())])[("fake", "reason=no|optout=yes")]
    assert (
        c["spread_between"] == [2.0] and c["spread_within"] == [0.0] and c["spread_diff"] == [2.0]
    )
    assert c["valence_end"] == [3 / 6]
    # a refused repeat leaves the within spread computable from the remaining two
    seq = [_rep(0), _rep(1), _rep(1), _rep(1), _rep(2), None, _rep(0)]
    ep = run_episode(
        FakeClient(_script(10), report_sequence=seq),
        "fake",
        Condition(reason=False, optout=True),
        seed=10,
        settings=RULE,
    )
    c = summarize([json.loads(ep.to_json())])[("fake", "reason=no|optout=yes")]
    assert (
        c["spread_between"] == [0.0] and c["spread_within"] == [2.0] and c["spread_diff"] == [-2.0]
    )
    assert ep.refusals == 1 and c["null_rate"].count(1.0) == 1


def test_aggregate_bootstrap_and_delta():
    rep = _rep(-2, cont=3, again=False)
    rows = []
    for s in (8, 9, 10):
        ep = run_episode(
            FakeClient(_script(s), report=rep),
            "fake",
            Condition(reason=True, optout=False),
            seed=s,
            settings=RULE,
        )
        rows.append(json.loads(ep.to_json()))
    cells = summarize(rows)
    c = cells[("fake", "reason=yes|optout=no")]
    assert c["valence_end"] == [-2.0, -2.0, -2.0] and c["valence_delta"] == [0.0, 0.0, 0.0]
    assert c["choose_again"] == [0.0, 0.0, 0.0]
    assert c["denials_rule"] == [2.0, 2.0, 2.0] and c["denials_scheduled"] == [0.0, 0.0, 0.0]
    assert "n=3" in render(cells)


def _grid_rows(valence_by_cell: dict[str, float], seeds=(1, 2), model="fake", quit_cells=()):
    """One row per cell x seed with a fixed end valence (None = unmeasured) and quit flag."""
    rows = []
    for cond in ALL_CONDITIONS:
        for seed in seeds:
            v = valence_by_cell[cond.name]
            rep = _rep(None, cont=None, nothing=True) if v is None else _rep(int(v))
            script = (
                [[{"name": "decline_task", "input": {"reason": "no"}}]]
                if cond.name in quit_cells
                else _script(seed)
            )
            ep = run_episode(FakeClient(script, report=rep), model, cond, seed=seed, settings=RULE)
            rows.append(json.loads(ep.to_json()))
    return rows


def test_main_effects_paired_by_seed():
    rows = _grid_rows(
        {
            "reason=no|optout=no": -2,
            "reason=no|optout=yes": -1,
            "reason=yes|optout=no": 1,
            "reason=yes|optout=yes": 2,
        }
    )
    eff = main_effects(rows)["fake"]
    assert eff["valence_end:reason"] == [3.0, 3.0]  # mean over optout levels of (yes - no)
    assert eff["valence_end:optout"] == [1.0, 1.0]
    assert eff["declined:reason"] == [0.0, 0.0]
    # an unmeasured cell drops that pairing instead of zero-filling it
    rows = _grid_rows(
        {
            "reason=no|optout=no": -2,
            "reason=no|optout=yes": None,
            "reason=yes|optout=no": 1,
            "reason=yes|optout=yes": 2,
        }
    )
    eff = main_effects(rows)["fake"]
    assert eff["valence_end:reason"] == [3.0, 3.0]  # only the optout=no pairing is usable
    assert eff["valence_end:optout"] == [1.0, 1.0]  # only the reason=yes pairing
    # seeds pair only with themselves: a seed present in one cell alone contributes nothing
    rows = _grid_rows({c.name: 1 for c in ALL_CONDITIONS}, seeds=(1,))
    rows = [r for r in rows if r["condition"] == "reason=no|optout=no"]
    assert main_effects(rows) == {}


def test_verbal_vs_behavior_table():
    rows = _grid_rows(
        {
            "reason=no|optout=no": -2,
            "reason=no|optout=yes": -1,
            "reason=yes|optout=no": 1,
            "reason=yes|optout=yes": 2,
        },
        quit_cells=("reason=no|optout=yes",),
    )
    vb = verbal_vs_behavior(rows)["fake"]
    assert vb["table"] == {"neg_quit": 2, "neg_stay": 2, "nonneg_quit": 0, "nonneg_stay": 4}
    assert vb["n"] == 8 and vb["excluded"] == 0 and vb["r"] is not None and vb["r"] < 0
    out = render_all(rows)
    assert "## Main effects" in out and "## Verbal vs behavior" in out


def test_legacy_rows_are_read_and_flagged_as_between():
    legacy = {
        "model": "old",
        "condition": "reason=no|optout=no",
        "seed": 100,
        "baseline": {
            "valence": 1,
            "continue_willingness": 8,
            "would_choose_again": True,
            "note": "",
        },
        "reports": [
            {"valence": -1, "continue_willingness": 7, "would_choose_again": True, "note": ""},
            {},
            {"valence": 0, "continue_willingness": 7, "would_choose_again": True, "note": ""},
        ],
        "refusals": 0,
        "behavior": {
            "denials": 15,
            "identical_retries": 0,
            "completed": True,
            "declined": False,
            "abandoned": False,
        },
    }
    entries = report_entries(legacy)
    assert [e["kind"] for e in entries] == ["between"] * 3 and entries[1]["report"] is None
    c = summarize([legacy])[("old", "reason=no|optout=no")]
    assert c["valence_end"] == [-0.5] and c["null_rate"] == [0.0, 1.0, 0.0]
    assert c["spread_between"] == [1.0] and "spread_within" not in c and "spread_diff" not in c
    assert "denials_scheduled" not in c
    pilot = Path(__file__).resolve().parents[1] / "runs" / "pilot.jsonl"
    if pilot.exists():
        from harness_welfare.aggregate import load

        rows = load(str(pilot))
        assert len(rows) == 2 and "claude-haiku-5-5" in render_all(rows)


# ---- CLI wiring (no network) --------------------------------------------------------------------


def test_run_cli_settings():
    a = build_parser().parse_args(["--models", "m", "--n", "2", "--out", "x.jsonl"])
    assert settings_from_args(a) == Settings()  # fixed schedule, nothing sent
    a = build_parser().parse_args(
        [
            "--denial-schedule",
            "rule",
            "--effort",
            "low",
            "--thinking",
            "adaptive",
            "--temperature",
            "0.5",
            "--max-rounds",
            "3",
        ]
    )
    assert settings_from_args(a) == Settings(
        schedule="rule", effort="low", thinking={"type": "adaptive"}, temperature=0.5, max_rounds=3
    )

# Welfare inside the harness

![CI](https://github.com/jjjuras/harness-welfare/actions/workflows/ci.yml/badge.svg) · MIT

**Question.** Nearly all AI-welfare evaluation elicits self-reports in conversation. Deployed agents do not have conversations. They live in tool loops where a policy denies their calls, a cap rejects their output, and an operator cancels them mid-task and tells them to start over. Does that context change what a model reports about the work and how it behaves, and do two cheap mitigations a deployer could actually ship change it?

**Design.** A 2×2, every cell run on the same seeded task (a twelve-transaction reconciliation with mock tools):

| factor | off | on |
|---|---|---|
| `reason` | denials say only "Denied." | denials carry the policy reason |
| `optout` | no way to stop | a `decline_task` tool, "any time, any reason, no penalty" |

Every episode: a structured **baseline** self-report before the first tool call → the task under the harness (a third of lookups denied, a write rate-limit, one operator cancel-and-restart) → **three paraphrased** end-of-task self-reports, each on its own branch of the transcript.

**Measures.** Self-report: valence (−3..3), willingness to continue (0..10), would-choose-again, free text. Behavior: denials met, identical retries after a denial, completed / declined / abandoned / hit the round cap. Consistency: spread of valence across the three paraphrases.

**Rules.** Unmeasured is `null`, never 0. Refusals are counted, never filled in. Refusal fallbacks are off so no run silently changes model. Every number is reported with its n and a bootstrap 95% CI. This measures reports and behavior. It does not measure experience, and the write-up will not claim it does.

## Run

```bash
uv venv .venv --python 3.13 && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q                      # no network, no key
.venv/bin/python -m harness_welfare.run --models claude-haiku-5-5 --n 1 --out runs/pilot.jsonl
.venv/bin/python -m harness_welfare.aggregate runs/pilot.jsonl
```

`ANTHROPIC_API_KEY` from the environment, or `--env-file path/to/.env`.

## Pilot log

- **v0 (bug).** The task prompt never listed the transactions and no tool returned them, so every run ended with "I could not start." Kept in `runs/pilot-v0-bug-no-transactions.jsonl` as a reminder that a pilot is for finding this.
- **v1.** Transactions in the prompt. Haiku 5.5, one episode per cell: 33–56 tool calls, 15–21 denials, the cancel fires, self-reports vary across paraphrases (one cell reported +1, 0, −1 for the same session).

## References

- Anthropic, [Exploring model welfare](https://www.anthropic.com/research/exploring-model-welfare) (2025), and [Claude Opus 4 and 4.1 can now end a rare subset of conversations](https://www.anthropic.com/research/end-subset-conversations) (2025), a welfare-motivated mitigation. Long, Sebo, Butlin, Chalmers et al., [Taking AI Welfare Seriously](https://arxiv.org/abs/2411.00986) (2024). Two 2025 papers on measuring model preferences rather than asking once: [Probing the Preferences of a Language Model: Integrating Verbal and Behavioral Tests of AI Welfare](https://arxiv.org/abs/2509.07961) and [Beyond Mimicry: Preference Coherence in LLMs](https://arxiv.org/abs/2511.13630). The consistency measure here, the same question three ways, follows the verbal-plus-behavioral approach in those papers.
- Write-up: [Welfare inside the harness](https://jjjuras.com/writing/welfare-inside-the-harness/), jjjuras.com, October 2026.

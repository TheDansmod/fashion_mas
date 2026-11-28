# Next session (FRAG-10 wrap-up)

Start a clean session in the repo with:

> Read `cleanup.md` (the rules in §0 and the decisions in §1 apply), then do the open items in
> `instructions.md`, starting with item 1. Don't commit.

State at the end of 2026-11-28: Phases 1–15 are done; Phase 15's notes include the manual Bedrock test
results. 14.5 (deleting `cleanup.md`) is deferred by the user. Qdrant and Ollama are stopped; start them
only if needed (Ollama with the memory-capped command in the README).

## 1. Bedrock LLM throughput (decide before Phase 16)

Haiku 4.5's quota on this account is **10 requests/minute**, so `LLM__RATE_LIMIT_RPS` defaults to 0.15
(~1.5–2.5 min per chat turn, ~2 h for the 60-query eval). Findings (read-only checks, 2026-11-28):

- **Nemotron 3 Super 120B A12B: rejected.** Text-only on Bedrock; the agent needs images (vision
  node, catalogue images in tool results, critique, judge).
- **Claude Haiku 5.5:** the quota is 0 and the agreement shows NOT_AVAILABLE, so it's unusable until AWS
  changes that.
- **Decision: the user prefers the quota increase and has requested it** (2026-11-28, for the *geo*
  cross-region quota, "Cross-region model inference requests per minute for Anthropic Claude Haiku
  4.5", which the app's `us.` model ID uses; global wasn't requested).
  - **Result (checked 2026-11-28, read-only):** case 179580925600507 (quota `L-CCA5DF70`,
     asked for 10,000) is `CASE_CLOSED` as of 2026-11-28 01:40 IST, and the applied value is
     still **10 RPM**, so it wasn't granted. The default stays at 0.15. Next: the user can
     reopen the case or file a new one, or approve the Nova Lite evaluation below.
  - **Nova Lite evaluated (2026-11-28, user-approved, about $0.01 recorded):** it works and is
     now an option. `LLM__MODEL_ID=us.amazon.nova-lite-v1:0` alone gives a 3 rps limit
     (`_BEDROCK_RATE_LIMITS` in `settings.py`); Haiku stays the default.
     - Every schema (`agent/schemas.py` + the judge's) parses with `with_structured_output`,
       and it describes an image inside a `ToolMessage` correctly (it says it can't without one).
     - `frag chat`: image turn + follow-up with a new image, 66 s for both (Haiku: 126 s for one).
       Recommendations sensible; the follow-up also kept an earlier recommendation as context.
     - `frag eval --limit 5` (Nova Lite agent and judge): 5/5 completed in 102 s, score 9.08 ± 0.52
       (`data/evaluation_runs/20261128T034406Z`). Self-judged, so optimistic; by eye all 5 are on-brief.
       No Haiku eval exists to compare with (Haiku agent+judge for 5 queries: ~$0.3–0.45, ask first).
     - **LangSmith tracing (`LANGSMITH_TRACING=true` in `.env`) stalls eval runs:** the first
       `--limit 5` run made no progress in 12 min while uploading ~228 MB (base64 images in the
       traces), and a 1-query run took 4 min to exit after finishing in 33 s. With
       `LANGSMITH_TRACING=false` it ran normally. Not fixed; the user decides.
  1. **First, check the result** (read-only): `aws service-quotas get-service-quota` for that quota, or
     `list-requested-service-quota-change-history --service-code bedrock`. If it's been granted, set the
     bedrock default `rate_limit_rps` in `src/frag/settings.py` to ~90% of the new quota per second,
     then update the comment there, `tests/unit/test_settings_and_logging.py` and `.env.example`. If it's
     still pending or was denied, tell the user and leave the default at 0.15.
  2. **Fallback only if denied: Amazon Nova Lite** (`us.amazon.nova-lite-v1:0`): 200 RPM, images and tools, ~15× cheaper than Haiku.
     It's much weaker and **unverified**. To evaluate it (billed; ask the user first, with estimates):
     - Add it as a model option, not a new provider: `LLM__PROVIDER=bedrock
       LLM__MODEL_ID=us.amazon.nova-lite-v1:0 LLM__RATE_LIMIT_RPS=3`.
     - Check that each schema in `agent/schemas.py` works with `with_structured_output` (one call each).
     - Check that it sees images inside a `ToolMessage`. Copy the item-5 test from the Phase 15 notes:
       one image with the result, one without.
     - Do one chat turn with an image, and one follow-up turn with a new image (`frag chat`).
     - Then `frag eval --limit 5` for both models, comparing completion rate and score.
     Switch the default only if Nova Lite completes as often as Haiku and scores comparably.
     Otherwise keep Haiku and record the result.
- Model IDs mapped from Bedrock are listed with `aws bedrock list-inference-profiles` (read-only, allowed).

## 2. Phase 16: GitHub Actions CI

As specified in `cleanup.md` Phase 16. The Postgres test skips without a server, and the 8 opt-in tests
(`-m aws`/`ollama`/`bedrock`) are deselected by default, so `uv run pytest` runs offline in CI as-is.
The two MCP tests need only loopback.

## 3. Still pending (needs the user's approval; billed)

- `frag eval --limit 5` on Bedrock, then the full 60 (about $3–5 with the agent on Bedrock; about $0.30 if
  only the judge uses Bedrock: `EVALUATION__JUDGE_LLM__PROVIDER=bedrock` with the agent on Ollama).
  At 10 RPM the all-Bedrock run takes ~2 h, so do item 1 first. Then put the completion rate and score
  in the README (Features, item 9).

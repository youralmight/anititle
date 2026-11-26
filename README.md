# anititle

Match an anime release title to its AniList entry and episode, using an agent.

```python
from anititle import agent
agent.match("[ANi] 再見菈菈 - 02 [1080P][Baha][WEB-DL][AAC AVC][CHT].mp4", published_at="2026-07-13")
# Result(status='matched', anilist_id=177637, episode=2, evidence='...')
```

From a shell: `uv run anititle "<title>" --published-at 2026-07-05T13:06:52Z`.

Each call runs one LangChain agent (`create_agent`). The model comes from a LiteLLM
gateway: set `LITELLM_BASE_URL` (ending in `/v1`) and `LITELLM_API_KEY`. The default
model group is `anilist-overlay`. The agent can only call three lookup tools
(`src/anititle/agent.py`); the matching rules are the system prompt,
`src/anititle/rules.md`.
A result is either a match or an error with `error_type`: `not_found`,
`contradiction`, `not_single_episode`, `not_anime`, `timeout`.

Evaluation: `uv run python eval/run.py` runs `eval/dataset.jsonl` (52 labeled
titles, half hard, each tagged with what makes it hard) and writes `eval/runs/`.
`eval/overlay_baseline.py` runs the same set through anilist-overlay's fixed
pipeline for comparison. Unit tests: `uv run pytest`.

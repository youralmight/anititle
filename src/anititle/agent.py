"""Run one title through a LangChain agent, with the model served by LiteLLM."""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from typing import Literal

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, ConfigDict, ValidationError

from anititle import tools

DEFAULT_MODEL = "anilist-overlay"  # the LiteLLM model group anilist-overlay itself uses
DEFAULT_TIMEOUT = 180
MAX_GRAPH_STEPS = 25  # model and tool steps together
MODEL_REQUEST_TIMEOUT = 120
ANSWER_ATTEMPTS = 3  # the first answer plus two corrections


class MatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["matched", "error"]
    anilist_id: int | None
    episode: int | None
    evidence: str
    error_type: Literal["not_found", "contradiction", "not_single_episode", "not_anime"] | None
    reason: str


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["matched", "error"]},
        "anilist_id": {"type": ["integer", "null"]},
        "episode": {"type": ["integer", "null"]},
        "evidence": {"type": "string"},
        "error_type": {"type": ["string", "null"],
                       "enum": ["not_found", "contradiction", "not_single_episode", "not_anime", None]},
        "reason": {"type": "string"},
    },
    "required": ["status", "anilist_id", "episode", "evidence", "error_type", "reason"],
    "additionalProperties": False,
}


@dataclass
class Result:
    status: str  # "matched" or "error"
    anilist_id: int | None = None
    episode: int | None = None
    evidence: str = ""
    error_type: str | None = None
    reason: str = ""
    seconds: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


@tool
def search_anime(query: str) -> str:
    """Search anime by name in any language (Chinese, Japanese, romaji, English).

    Searches AniList and a Chinese-name index. Use a name only, without group
    tags, episode numbers or resolution. Each result: id, titles, format,
    status, episodes, start/end air dates.
    """
    return _json(tools.search_anime(query))


@tool
def get_anime(anilist_id: int) -> str:
    """One AniList entry with its next airing episode and related entries (prequel, sequel, side story...)."""
    return _json(tools.get_anime(anilist_id))


@tool
def season_chain(anilist_id: int, on_date: str | None = None) -> str:
    """All seasons of the franchise run containing this entry, first to last.

    Each entry has `episodes_before` = TV/ONA episodes of the earlier seasons,
    the offset to subtract when a release numbers episodes continuously.
    Pass the release's publish time as `on_date` (exactly as given, ISO 8601)
    to also get `aired_by_date` = the last episode each season had aired by then.
    """
    return _json(tools.season_chain(anilist_id, on_date))


def build_agent(model: BaseChatModel):
    return create_agent(model, tools=[search_anime, get_anime, season_chain])


def build_prompt(title: str, published_at: str | None) -> str:
    """The rules, the title and the required JSON shape, in one message."""
    rules = files("anititle").joinpath("rules.md").read_text()
    task = f"{rules}\n# 本条\n\n片源标题：{title}\n发布时间：{published_at or '无'}\n"
    schema = json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)
    return f"{task.strip()}\n\nYou must respond with JSON only that matches this JSON Schema:\n{schema}"


def _retry_prompt(prompt: str, errors: list[str]) -> str:
    listed = "\n".join(f"- {e}" for e in errors)
    return (f"{prompt}\n\nPrevious response was invalid with validation errors:\n{listed}\n\n"
            "Respond again with JSON only that matches the schema.")


def _extract_json(text: str):
    """The fenced JSON block if there is one, otherwise the first JSON value in the text."""
    fenced = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
    if fenced:
        return json.loads(fenced.group(1))
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch in "{["):
        try:
            return decoder.raw_decode(text, start)[0]
        except json.JSONDecodeError:
            continue
    return json.loads(text)


def _parse_answer(text: str) -> tuple[MatchOutput | None, list[str]]:
    try:
        return MatchOutput.model_validate(_extract_json(text)), []
    except json.JSONDecodeError as e:
        return None, [f"Invalid JSON: {e}"]
    except ValidationError as e:
        return None, [f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}" for err in e.errors()]


def _run(agent, prompt: str, stop: threading.Event):
    messages = [HumanMessage(prompt)]
    output = None
    for _ in range(ANSWER_ATTEMPTS):
        state = None
        for state in agent.stream({"messages": messages}, {"recursion_limit": MAX_GRAPH_STEPS}, stream_mode="values"):
            if stop.is_set():
                return None
        messages = state["messages"]
        output, errors = _parse_answer(messages[-1].text)
        if output:
            break
        messages = [*messages, HumanMessage(_retry_prompt(prompt, errors))]
    return messages, output


def _usage(messages: list) -> dict:
    total = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0}
    for message in messages:
        if isinstance(message, AIMessage) and message.usage_metadata:
            usage = message.usage_metadata
            details = usage.get("input_token_details") or {}
            total["model_calls"] += 1
            total["input_tokens"] += usage.get("input_tokens", 0)
            total["output_tokens"] += usage.get("output_tokens", 0)
            total["cache_read"] += details.get("cache_read", 0)
            total["cache_creation"] += details.get("cache_creation", 0)
    return total


def run_agent(model: BaseChatModel, title: str, published_at: str | None = None,
              timeout: int = DEFAULT_TIMEOUT) -> Result:
    """API and AniList failures propagate: they are not answers about the title."""
    start = time.monotonic()
    stop = threading.Event()
    pool = ThreadPoolExecutor(1)
    future = pool.submit(_run, build_agent(model), build_prompt(title, published_at), stop)
    try:
        messages, output = future.result(timeout)
    except FutureTimeout:
        stop.set()  # the worker exits after its current step
        return Result("error", error_type="timeout", reason=f"no answer within {timeout}s", seconds=float(timeout))
    except GraphRecursionError:
        return Result("error", reason=f"no answer within {MAX_GRAPH_STEPS} steps",
                      seconds=round(time.monotonic() - start, 1))
    finally:
        pool.shutdown(wait=False)

    seconds = round(time.monotonic() - start, 1)
    calls = [{"name": c["name"], "args": c["args"]}
             for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    usage = _usage(messages)
    if output is None:
        return Result("error", reason="no valid JSON answer", seconds=seconds, tool_calls=calls, usage=usage)
    return Result(**output.model_dump(), seconds=seconds, tool_calls=calls, usage=usage)


def match(
    title: str,
    published_at: str | None = None,
    *,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
) -> Result:
    chat_model = ChatOpenAI(
        model=model,
        base_url=os.environ["LITELLM_BASE_URL"],
        api_key=os.environ["LITELLM_API_KEY"],
        temperature=0,
        timeout=MODEL_REQUEST_TIMEOUT,
    )
    return run_agent(chat_model, title, published_at, timeout)

"""The LangChain loop, driven by a scripted model instead of the API."""

import json
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from anititle import agent, tools

ANSWER = {"status": "matched", "anilist_id": 7, "episode": 2, "evidence": "found", "error_type": None, "reason": "ok"}
USAGE = {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}


class ScriptedModel(BaseChatModel):
    replies: list
    delay: float = 0
    seen: list = []

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(self.delay)
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def scripted(*replies, **kwargs):
    return ScriptedModel(replies=list(replies), seen=[], **kwargs)


def tool_call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{name}", "type": "tool_call"}],
                     usage_metadata=USAGE)


def answer(text=None):
    return AIMessage(content=text or json.dumps(ANSWER), usage_metadata=USAGE)


def test_loop_uses_tools_then_parses_the_final_json(monkeypatch):
    monkeypatch.setattr(tools, "search_anime", lambda q: [{"id": 7, "titles": [q]}])
    model = scripted(tool_call("search_anime", {"query": "再見菈菈"}), answer())
    result = agent.run_agent(model, "[ANi] 再見菈菈 - 02", "2026-07-13T00:00:00Z")
    assert (result.status, result.anilist_id, result.episode) == ("matched", 7, 2)
    assert result.tool_calls == [{"name": "search_anime", "args": {"query": "再見菈菈"}}]
    assert result.usage["model_calls"] == 2
    assert "再見菈菈" in model.seen[1][-1].content  # the tool result went back to the model


def test_prompt_is_one_message_with_rules_title_and_schema():
    model = scripted(answer())
    agent.run_agent(model, "[ANi] 再見菈菈 - 02", "2026-07-13T00:00:00Z")
    assert len(model.seen[0]) == 1 and isinstance(model.seen[0][0], HumanMessage)
    prompt = model.seen[0][0].content
    assert prompt.startswith("# 任务") and "片源标题：[ANi] 再見菈菈 - 02\n发布时间：2026-07-13T00:00:00Z" in prompt
    assert "\n\nYou must respond with JSON only that matches this JSON Schema:\n{" in prompt


def test_json_is_found_in_a_fence_or_after_text():
    fenced = "结论如下\n```json\n" + json.dumps(ANSWER) + "\n```"
    assert agent.run_agent(scripted(answer(fenced)), "title").anilist_id == 7
    assert agent.run_agent(scripted(answer("结论：" + json.dumps(ANSWER))), "title").anilist_id == 7


def test_invalid_answer_is_retried_with_the_errors():
    model = scripted(answer(json.dumps({**ANSWER, "status": "maybe"})), answer())
    assert agent.run_agent(model, "title").status == "matched"
    retry = model.seen[1][-1].content
    assert "Previous response was invalid" in retry and "status" in retry


def test_gives_up_after_the_allowed_corrections():
    model = scripted(answer("不是 JSON"), answer("不是 JSON"), answer("不是 JSON"))
    result = agent.run_agent(model, "title")
    assert (result.status, result.error_type, result.reason) == ("error", None, "no valid JSON answer")


def test_timeout_is_reported_as_an_error_result():
    result = agent.run_agent(scripted(answer(), delay=2), "title", timeout=0.1)
    assert (result.status, result.error_type) == ("error", "timeout")

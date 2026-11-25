import json
from types import SimpleNamespace as NS

import pytest

from car.models import ClaudeModel, OpenAICompatModel, ToolSpec, Usage, claude_cost
from car.tools import TOOL_SPECS


class FakeMessages:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def create(self, **kwargs):
        # Snapshot the messages: the conversation keeps appending to the same list.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def usage(i=100, o=50, r=0, w=0):
    return NS(input_tokens=i, output_tokens=o, cache_read_input_tokens=r,
              cache_creation_input_tokens=w)  # fmt: skip


def test_claude_request_shape_and_history():
    thinking = NS(type="thinking", thinking="", signature="sig")
    tool_use = NS(type="tool_use", id="tu_1", name="read_file", input={"path": "a.py"})
    first = NS(content=[thinking, tool_use], stop_reason="tool_use", usage=usage(r=1000, w=200))
    second = NS(content=[NS(type="text", text="Done.")], stop_reason="end_turn", usage=usage())
    fake = FakeMessages([first, second])
    model = ClaudeModel("claude-haiku-5-5", effort="low", client=NS(messages=fake))
    conv = model.start("system prompt", TOOL_SPECS, "fix it")

    r1 = conv.next()
    assert r1.tool_calls[0].name == "read_file" and r1.tool_calls[0].arguments == {"path": "a.py"}
    conv.add_tool_results([("tu_1", "file contents", False)])
    r2 = conv.next()
    assert r2.text == "Done."

    req = fake.calls[0]
    assert req["model"] == "claude-haiku-5-5"
    assert req["output_config"] == {"effort": "low"}
    assert req["cache_control"] == {"type": "ephemeral"}
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    # Forced tool choice is rejected by current models; thinking stays at its default.
    assert "tool_choice" not in req and "thinking" not in req and "temperature" not in req
    assert {t["name"] for t in req["tools"]} == {t.name for t in TOOL_SPECS}
    # The assistant turn is passed back unchanged, thinking block included.
    history = fake.calls[1]["messages"]
    assert history[1] == {"role": "assistant", "content": [thinking, tool_use]}
    assert history[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "file contents"}
    ]
    assert r1.usage == Usage(100, 50, 1000, 200)
    assert r1.cost_usd == pytest.approx(claude_cost("claude-haiku-5-5", Usage(100, 50, 1000, 200)))


def test_claude_tool_errors_and_refusals():
    refusal = NS(content=[], stop_reason="refusal", usage=usage(),
                 stop_details=NS(category="cyber", explanation="..."))  # fmt: skip
    fake = FakeMessages([refusal])
    conv = ClaudeModel("claude-sonnet-5-5", client=NS(messages=fake)).start("s", TOOL_SPECS, "t")
    conv.add_tool_results([("a", "boom", True)])
    assert conv.messages[-1]["content"][0]["is_error"] is True
    r = conv.next()
    assert r.stop_reason == "refusal" and r.refusal_category == "cyber" and not r.tool_calls


def test_claude_api_errors_become_error_responses():
    class Failing:
        def create(self, **kwargs):
            raise RuntimeError("overloaded")

    conv = ClaudeModel("claude-opus-5-5", client=NS(messages=Failing())).start("s", [], "t")
    r = conv.next()
    assert r.stop_reason == "error" and "overloaded" in r.error


def test_claude_costs():
    assert claude_cost("claude-opus-5-5", Usage(1_000_000, 1_000_000)) == pytest.approx(24.0)
    assert claude_cost("claude-sonnet-5-5", Usage(0, 0, 1_000_000, 0)) == pytest.approx(0.10)
    # Claude Haiku 5.5 prompts above 100K tokens are billed at the higher tier.
    assert claude_cost("claude-haiku-5-5", Usage(200_000, 0)) == pytest.approx(0.10)
    with pytest.raises(ValueError):
        ClaudeModel("claude-unknown", client=object())


class FakeLocal(OpenAICompatModel):
    def __init__(self, replies):
        super().__init__("qwen-test")
        self.replies = replies
        self.payloads = []

    def post(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        return self.replies.pop(0)


def test_local_model_tool_calls_and_malformed_arguments():
    reply = {
        "choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
                {"id": "c2", "type": "function",
                 "function": {"name": "search", "arguments": '{"pattern": "x"'}},
            ]}}],
        "usage": {"prompt_tokens": 300, "completion_tokens": 40},
    }  # fmt: skip
    model = FakeLocal([reply, {"choices": [{"message": {"content": "ok"}}]}])
    conv = model.start("sys", [ToolSpec("read_file", "d", {"type": "object"})], "task")
    r = conv.next()
    assert [c.name for c in r.tool_calls] == ["read_file", "search"]
    assert r.tool_calls[0].arguments == {"path": "a.py"} and r.tool_calls[0].parse_error is None
    assert "invalid tool arguments" in r.tool_calls[1].parse_error
    assert r.usage == Usage(300, 40) and r.cost_usd == 0.0
    conv.add_tool_results([("c1", "contents", False), ("c2", "bad args", True)])
    conv.next()
    sent = model.payloads[1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "tool"]
    assert sent[4] == {"role": "tool", "tool_call_id": "c2", "content": "ERROR: bad args"}
    assert model.payloads[0]["tools"][0]["function"]["name"] == "read_file"


def test_local_server_down_is_an_error_response():
    model = OpenAICompatModel("x", base_url="http://127.0.0.1:9", timeout_s=2)
    r = model.start("s", [], "t").next()
    assert r.stop_reason == "error"


def test_claude_through_the_real_sdk_against_a_local_fake_server():
    """Full SDK path (serialization and parsing) with no network and no real key."""
    anthropic = pytest.importorskip("anthropic")
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    replies = [
        {"content": [{"type": "thinking", "thinking": "", "signature": "sig-1"},
                     {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                      "input": {"path": "a.py"}}],
         "stop_reason": "tool_use"},
        {"content": [{"type": "text", "text": "Fixed."}], "stop_reason": "end_turn"},
    ]  # fmt: skip
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            reply = replies.pop(0)
            payload = json.dumps({
                "id": f"msg_{len(bodies)}", "type": "message", "role": "assistant",
                "model": "claude-haiku-5-5", "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5,
                          "cache_read_input_tokens": 3, "cache_creation_input_tokens": 2},
                **reply,
            }).encode()  # fmt: skip
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = anthropic.Anthropic(
            api_key="test-dummy", base_url=f"http://127.0.0.1:{server.server_port}", max_retries=0
        )
        conv = ClaudeModel("claude-haiku-5-5", client=client).start("sys", TOOL_SPECS, "task")
        r1 = conv.next()
        assert r1.tool_calls[0].arguments == {"path": "a.py"}
        assert r1.usage == Usage(10, 5, 3, 2)
        conv.add_tool_results([("toolu_1", "contents", False)])
        assert conv.next().text == "Fixed."
    finally:
        server.shutdown()

    first, second = bodies
    assert first["output_config"] == {"effort": "medium"}
    assert first["cache_control"] == {"type": "ephemeral"}
    # The thinking block went back exactly as received, signature included.
    assert second["messages"][1]["content"][0] == {
        "type": "thinking",
        "thinking": "",
        "signature": "sig-1",
    }
    assert second["messages"][2]["content"][0]["type"] == "tool_result"

"""Offline tests: the tools against a local stand-in for the API on 127.0.0.1, by their own call, by tool_results() on the
SDK's own types, and run by the SDK's tool runner (sync and async, whole and streamed) and by a loop of the app's own on
client.messages.create, all sent to a stand-in of the Messages API on the same server."""

import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import anthropic
import anyio
import pytest
from anthropic import Anthropic, AsyncAnthropic, beta_async_tool, beta_tool
from anthropic.resources.beta.messages import Messages

from anthropic_datacircle import (
    ASYNC_TOOLS,
    TOOL_NAMES,
    TOOLS,
    DatacircleError,
    atool_results,
    best_plans,
    datacircle_async_tools,
    datacircle_best_plans,
    datacircle_provider,
    datacircle_tools,
    list_apis,
    provider,
    search_plans,
    tool_results,
)
from anthropic_datacircle import _api, tools as tools_module

PLAN = {"api": "Employee email", "rank": 1, "provider": "Hunter", "plan": "Starter", "score": 80.1, "perThousand": 4.9,
        "page": "https://datacircle.dev/x", "terms": "long", "aggregators": ["a"]}
SEEN = []
BUSY = []
ASKED = []
# The calls the stand-in model makes on its first turn: each tool, a refused one and the app's own tool.
CALLS = [("t1", "datacircle_list_apis", {"query": "profile", "limit": 1}),
         ("t2", "datacircle_best_plans", {"slug": "email-finder", "limit": 1}),
         ("t3", "datacircle_search_plans", {"api": "email-finder", "query": "hunter", "sort": "price", "limit": 5}),
         ("t4", "datacircle_provider", {"slug": "Bright Data"}),
         ("t5", "datacircle_best_plans", {"slug": "nope"}),
         ("t6", "add_two_numbers", {"a": 1, "b": 2})]


def message(content, stop_reason):
    return {"id": f"msg_{len(ASKED)}", "type": "message", "role": "assistant", "model": "stand-in", "content": content,
            "stop_reason": stop_reason, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}


def events(reply):
    """The reply as the Messages API streams it."""
    start = dict(reply, content=[], stop_reason=None)
    out = [("message_start", {"type": "message_start", "message": start})]
    for index, block in enumerate(reply["content"]):
        if block["type"] == "text":
            out.append(("content_block_start", {"type": "content_block_start", "index": index,
                                                "content_block": {"type": "text", "text": ""}}))
            out.append(("content_block_delta", {"type": "content_block_delta", "index": index,
                                                "delta": {"type": "text_delta", "text": block["text"]}}))
        else:
            out.append(("content_block_start", {"type": "content_block_start", "index": index,
                                                "content_block": dict(block, input={})}))
            text = json.dumps(block["input"])
            for piece in (text[:5], text[5:]):
                out.append(("content_block_delta", {"type": "content_block_delta", "index": index,
                                                    "delta": {"type": "input_json_delta", "partial_json": piece}}))
        out.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    out.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": reply["stop_reason"],
                                                                      "stop_sequence": None},
                                  "usage": {"output_tokens": 1}}))
    out.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in out).encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def send(self, status, body, headers=None, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        for key, value in {"Content-Type": kind, "Content-Length": str(len(data)), **(headers or {})}.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # the Messages API: calls each tool and the app's own on the first turn, then gives back the results
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        ASKED.append(body)
        last = body["messages"][-1]["content"]
        results = [block for block in last if isinstance(block, dict) and block.get("type") == "tool_result"] \
            if isinstance(last, list) else []
        if results:
            got = [{"id": block["tool_use_id"], "is_error": bool(block.get("is_error")), "content": block.get("content")}
                   for block in results]
            reply = message([{"type": "text", "text": "done " + json.dumps(got)}], "end_turn")
        else:
            reply = message([{"type": "text", "text": "Let me look."}] +
                            [{"type": "tool_use", "id": id_, "name": name, "input": args} for id_, name, args in CALLS],
                            "tool_use")
        if body.get("stream"):
            self.send(200, events(reply), kind="text/event-stream")
        else:
            self.send(200, reply)

    def do_GET(self):
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        SEEN.append((url.path, query, self.headers.get("User-Agent")))
        headers = {}
        if url.path == "/v1/plans" and query.get("query") == "busy" and not BUSY:
            BUSY.append(1)
            status, body, headers = 429, {"error": "slow down"}, {"Retry-After": "1"}
        elif url.path == "/v1/apis":
            status, body = 200, [{"api": "Employee email", "slug": "email-finder", "plans": 9, "providers": 3, "extra": 1},
                                 {"api": "Employee profile", "slug": "linkedin-profile", "plans": 4, "providers": 2}]
        elif url.path == "/v1/apis/email-finder/best":
            status, body = 200, {"api": "Employee email", "slug": "email-finder", "total": 1, "plans": [PLAN]}
        elif url.path.startswith("/v1/apis/"):
            status, body = 404, {"error": "no API has the slug nope", "slugs": ["email-finder", "linkedin-profile"]}
        elif url.path == "/v1/plans" and query.get("query") == "down":
            status, body = 503, {"error": "down for a minute"}
        elif url.path == "/v1/plans":
            status, body = 200, {"total": 1, "offset": 0, "plans": [PLAN]}
        elif url.path == "/v1/providers/bright-data":
            status, body = 200, {"provider": "Bright Data", "slug": "bright-data", "website": "https://brightdata.com",
                                 "apis": [{"api": "Employee email", "slug": "email-finder", "rank": 7, "sheetPage": "s",
                                           "plans": [dict(PLAN, rank=9, plan="B"), dict(PLAN, rank=7, plan="A")]}]}
        else:
            status, body = 404, {"error": "no provider has this slug"}
        self.send(status, body, headers)


@pytest.fixture(scope="module", autouse=True)
def stand_in():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    before = _api.BASE_URL
    _api.BASE_URL = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield server.server_address[1]
    _api.BASE_URL = before
    server.shutdown()


def calls_made(action):
    SEEN.clear()
    result = action()
    return result, list(SEEN)


@beta_tool
def add_two_numbers(a: int, b: int) -> str:
    """Adds two numbers.

    Args:
      a: The first number.
      b: The second number.
    """
    return str(a + b)


@beta_async_tool
async def add_two_numbers_async(a: int, b: int) -> str:
    """Adds two numbers."""
    return str(a + b)


add_two_numbers_async.name = "add_two_numbers"


def test_definitions_as_the_sdk_sends_them():
    sent = {tool.to_dict()["name"]: tool.to_dict() for tool in datacircle_tools()}
    assert list(sent) == TOOL_NAMES == ["datacircle_list_apis", "datacircle_best_plans", "datacircle_search_plans",
                                        "datacircle_provider"]
    kinds = {"datacircle_list_apis": {"query": "string", "limit": "integer"},
             "datacircle_best_plans": {"slug": "string", "limit": "integer"},
             "datacircle_search_plans": {"api": "string", "query": "string", "sort": "string", "limit": "integer",
                                         "offset": "integer"},
             "datacircle_provider": {"slug": "string"}}
    for name, wanted in kinds.items():
        definition = sent[name]
        assert "https://datacircle.dev" in definition["description"], name
        schema = definition["input_schema"]
        assert {key: value["type"] for key, value in schema["properties"].items()} == wanted, name
        assert schema.get("required", []) == (["slug"] if "slug" in wanted else []), name
        assert all(value["description"] for value in schema["properties"].values()), name
    assert sent["datacircle_best_plans"]["input_schema"]["properties"]["limit"]["maximum"] == 50
    assert sent["datacircle_search_plans"]["input_schema"]["properties"]["sort"]["enum"] == ["score", "provider", "price"]
    assert [tool.to_dict() for tool in datacircle_async_tools()] == list(sent.values())


def test_some_tools_by_name():
    assert datacircle_tools(["datacircle_provider", "datacircle_best_plans"]) == [datacircle_provider, datacircle_best_plans]
    assert datacircle_tools() == list(TOOLS.values())
    assert datacircle_async_tools(["datacircle_provider"]) == [ASYNC_TOOLS["datacircle_provider"]]
    with pytest.raises(ValueError):
        datacircle_tools(["nope"])
    with pytest.raises(TypeError):
        datacircle_tools("datacircle_provider")


def test_each_function_and_the_calls_it_makes():
    listed, seen = calls_made(lambda: json.loads(list_apis("profile", 1)))
    assert listed == {"total": 1, "apis": [{"api": "Employee profile", "slug": "linkedin-profile", "plans": 4, "providers": 2}]}
    assert seen == [("/v1/apis", {}, _api.USER_AGENT)]
    assert json.loads(list_apis())["total"] == 2

    best, seen = calls_made(lambda: json.loads(best_plans("Email Finder", 1)))
    assert best["plans"] == [{key: PLAN[key] for key in ("api", "rank", "provider", "plan", "score", "perThousand", "page")}]
    assert seen[0][:2] == ("/v1/apis/email-finder/best", {"limit": "1"})

    found, seen = calls_made(lambda: json.loads(search_plans(api="email-finder", query="hunter", sort="Price", limit=5)))
    assert found["total"] == 1 and "terms" not in found["plans"][0]
    assert seen[0][:2] == ("/v1/plans", {"api": "email-finder", "query": "hunter", "sort": "price", "limit": "5"})

    seller, seen = calls_made(lambda: json.loads(provider("Bright Data")))
    assert seen[0][0] == "/v1/providers/bright-data"
    assert seller["apis"] == [{"api": "Employee email", "slug": "email-finder", "rank": 7, "plans": 2,
                               "best": {"plan": "A", "score": 80.1, "perThousand": 4.9, "page": "https://datacircle.dev/x"}}]
    # A tool's own call: the SDK's validation, then the function.
    assert json.loads(datacircle_best_plans.call({"slug": "email-finder", "limit": "1"}))["plans"][0]["rank"] == 1


def test_refusals_in_the_apis_words():
    with pytest.raises(DatacircleError) as unknown:
        best_plans("nope")
    assert str(unknown.value) == ("Datacircle's API answered 404: no API has the slug nope; the slugs: email-finder, "
                                  "linkedin-profile")
    assert unknown.value.status == 404 and not unknown.value.retryable
    with pytest.raises(DatacircleError, match="503: down for a minute") as down:
        search_plans(query="down")
    assert down.value.retryable
    with pytest.raises(DatacircleError, match="no provider"):
        provider("nobody")
    for refused in (lambda: best_plans("email-finder", 51), lambda: search_plans(sort="cheap"),
                    lambda: search_plans(offset=-1), lambda: list_apis(limit="many"),
                    lambda: best_plans(""), lambda: provider(None), lambda: search_plans(limit=True)):
        SEEN.clear()
        with pytest.raises(DatacircleError):
            refused()
        assert SEEN == []
    if hasattr(anthropic.lib.tools, "ToolError"):
        assert issubclass(DatacircleError, anthropic.lib.tools.ToolError)
        with pytest.raises(DatacircleError) as made:
            best_plans("email-finder", 0)
        assert made.value.content == "limit must be from 1 to 50, not 0"


def test_a_429_is_waited_once():
    answer, seen = calls_made(lambda: json.loads(search_plans(query="busy")))
    assert answer["total"] == 1 and len(seen) == 2


def first_turn():
    blocks = [{"type": "text", "text": "Let me look."}] + \
        [{"type": "tool_use", "id": id_, "name": name, "input": args} for id_, name, args in CALLS]
    return anthropic.types.Message.model_validate(dict(message(blocks, "tool_use"), model="claude-haiku-5-5"))


def check_results(results):
    """The tool_result blocks of the stand-in's calls of the four tools, in order, the app's own left out."""
    assert [block["tool_use_id"] for block in results] == ["t1", "t2", "t3", "t4", "t5"]
    assert json.loads(results[0]["content"])["apis"][0]["slug"] == "linkedin-profile"
    assert json.loads(results[1]["content"])["plans"][0]["provider"] == "Hunter"
    assert json.loads(results[2]["content"])["total"] == 1
    assert json.loads(results[3]["content"])["slug"] == "bright-data"
    assert results[4]["is_error"] is True and "404: no API has the slug nope" in results[4]["content"]
    assert not any(block.get("is_error") for block in results[:4])


def test_tool_results_on_the_sdks_types():
    reply = first_turn()
    check_results(tool_results(reply))
    check_results(tool_results(reply.content))
    check_results(tool_results(reply.model_dump()))
    check_results(tool_results([block.model_dump() for block in reply.content]))
    assert tool_results(None) == [] and tool_results([]) == []
    bad = tool_results([{"type": "tool_use", "id": "x", "name": "datacircle_best_plans", "input": {"slug": "a", "limit": 2.5}},
                        {"type": "tool_use", "id": "y", "name": "datacircle_provider", "input": {"slug": "a", "nope": 1}},
                        {"type": "tool_use", "id": "z", "name": "datacircle_provider", "input": "bright-data"}])
    assert [block["is_error"] for block in bad] == [True, True, True]
    assert "limit" in bad[0]["content"] and "nope" in bad[1]["content"] and "dictionary" in bad[2]["content"]
    check_results(anyio.run(atool_results, reply))


def client_of(port, kind=Anthropic):
    return kind(api_key="test", base_url=f"http://127.0.0.1:{port}", max_retries=0)


def check_run(final, first_request):
    """The run's last message holds what the stand-in got back: each tool's answer, the refusal with is_error, the app's own."""
    text = "".join(block.text for block in final.content if block.type == "text")
    got = json.loads(text[len("done "):])
    check_results([{"tool_use_id": item["id"], "content": item["content"],
                    **({"is_error": True} if item["is_error"] else {})} for item in got[:5]])
    assert got[5] == {"id": "t6", "is_error": False, "content": "3"}
    assert [tool["name"] for tool in first_request["tools"]] == TOOL_NAMES + ["add_two_numbers"]
    assert first_request["tools"][1]["input_schema"] == tools_module.SCHEMAS["datacircle_best_plans"]


def test_the_tool_runner_runs_them(stand_in):
    client = client_of(stand_in)
    for stream in (False, True):
        ASKED.clear()
        runner = client.beta.messages.tool_runner(model="claude-haiku-5-5", max_tokens=1000, stream=stream,
                                                  tools=[*datacircle_tools(), add_two_numbers],
                                                  messages=[{"role": "user", "content": "best email finder?"}])
        if stream:
            final = None
            for each in runner:
                final = each.get_final_message()
        else:
            final = list(runner)[-1]
        check_run(final, ASKED[0])
        assert len(ASKED) == 2 and runner.until_done() == final


def test_the_async_tool_runner_runs_them(stand_in):
    async def run(stream):
        client = client_of(stand_in, AsyncAnthropic)
        runner = client.beta.messages.tool_runner(model="claude-haiku-5-5", max_tokens=1000, stream=stream,
                                                  tools=[*datacircle_async_tools(), add_two_numbers_async],
                                                  messages=[{"role": "user", "content": "best email finder?"}])
        final = None
        async for each in runner:
            final = await each.get_final_message() if stream else each
        await client.close()
        return final

    for stream in (False, True):
        ASKED.clear()
        check_run(anyio.run(run, stream), ASKED[0])


@pytest.mark.skipif("run_tools_eagerly" not in inspect.signature(Messages.tool_runner).parameters,
                    reason="run_tools_eagerly came in a later anthropic")
def test_run_eagerly_while_the_reply_streams(stand_in):
    ASKED.clear()
    runner = client_of(stand_in).beta.messages.tool_runner(
        model="claude-haiku-5-5", max_tokens=1000, stream=True, run_tools_eagerly=True,
        tools=[*datacircle_tools(), add_two_numbers], messages=[{"role": "user", "content": "best email finder?"}])
    check_run(runner.until_done(), ASKED[0])


def test_a_loop_of_the_apps_own(stand_in):
    ASKED.clear()
    client = client_of(stand_in)
    definitions = [tool.to_dict() for tool in datacircle_tools()] + [add_two_numbers.to_dict()]
    messages = [{"role": "user", "content": "best email finder?"}]
    reply = client.messages.create(model="claude-haiku-5-5", max_tokens=1000, tools=definitions, messages=messages)
    results = tool_results(reply)
    check_results(results)
    own = [{"type": "tool_result", "tool_use_id": block.id, "content": add_two_numbers.call(block.input)}
           for block in reply.content if block.type == "tool_use" and block.name == "add_two_numbers"]
    messages += [{"role": "assistant", "content": reply.content}, {"role": "user", "content": results + own}]
    final = client.messages.create(model="claude-haiku-5-5", max_tokens=1000, tools=definitions, messages=messages)
    check_run(final, ASKED[0])

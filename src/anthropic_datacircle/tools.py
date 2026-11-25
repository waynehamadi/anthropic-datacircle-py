"""Tools for the Anthropic Python SDK on Datacircle's open API, https://api.datacircle.dev/v1: no key, no account.

Each tool is a beta_tool, and a beta_async_tool of the same name for an AsyncAnthropic client, with its own JSON schema:
given to client.beta.messages.tool_runner(), the runner sends its definition and runs the calls the model makes, whole
or streamed. A tool makes one HTTPS GET and gives the model the API's answer as JSON text. A refusal (an unknown slug,
an argument out of range, the API out of reach) raises DatacircleError: the runner sends its words as the call's result
with is_error, so the model can call again and the run goes on. Each tool checks its own arguments before any call, as
a loop of the app's own may send them unchecked.
"""

import json
import urllib.parse
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import anyio
import anyio.to_thread
from anthropic import beta_async_tool, beta_tool

from anthropic_datacircle import _api


def _best(plans: Any) -> Optional[Dict[str, Any]]:
    """A provider's best plan on an API: its lowest rank, a plan with no score having none."""
    ranked = sorted(plans, key=lambda plan: plan.get("rank") if isinstance(plan.get("rank"), (int, float)) else float("inf"))
    return _api.brief(ranked[0], ["plan", "score", "perThousand", "page"]) if ranked else None


def _list_apis(query: Any, limit: Any) -> Dict[str, Any]:
    count = _api.whole("limit", limit, 1, 200) or 20
    apis = _api.get("/apis")
    words = [word for word in str(query or "").lower().replace("-", " ").split() if word]
    held = []
    for index, api in enumerate(apis):
        name = f"{api.get('api', '')} {str(api.get('slug', '')).replace('-', ' ')}".lower()
        matched = sum(word in name for word in words)
        if matched or not words:
            held.append((-matched, index, api))
    found = [_api.brief(api, _api.API_KEYS) for _, _, api in sorted(held, key=lambda h: h[:2])]
    return {"total": len(found), "apis": found[:count]}


def _best_plans(slug: Any, limit: Any) -> Dict[str, Any]:
    made = _api.slug(slug)
    if not made:
        raise _api.DatacircleError("Give an API's slug, like 'email-finder': datacircle_list_apis gives every slug")
    answer = _api.get(f"/apis/{urllib.parse.quote(made)}/best", {"limit": _api.whole("limit", limit, 1, 50)})
    return {**answer, "plans": [_api.brief(plan) for plan in answer.get("plans") or []]}


def _search_plans(api: Any, query: Any, sort: Any, limit: Any, offset: Any) -> Dict[str, Any]:
    order = str(sort).strip().lower() if sort not in (None, "") else None
    if order is not None and order not in ("score", "provider", "price"):
        raise _api.DatacircleError(f"sort must be score, provider or price, not {sort!r}")
    params = {"api": api, "query": query, "sort": order, "limit": _api.whole("limit", limit, 1, 50),
              "offset": _api.whole("offset", offset, 0)}
    answer = _api.get("/plans", params)
    return {**answer, "plans": [_api.brief(plan) for plan in answer.get("plans") or []]}


def _provider(slug: Any) -> Dict[str, Any]:
    made = _api.slug(slug)
    if not made:
        raise _api.DatacircleError("Give a provider's slug or name, like 'bright-data' or 'Bright Data'")
    answer = _api.get(f"/providers/{urllib.parse.quote(made)}")
    apis = [{**_api.brief(entry, ["api", "slug", "rank", "bestPage"]), "plans": len(entry.get("plans") or []),
             "best": _best(entry.get("plans") or [])} for entry in answer.get("apis") or []]
    return {**answer, "apis": apis}


def _text(answer: Dict[str, Any]) -> str:
    return json.dumps(answer, ensure_ascii=False)


# What the model reads of each tool: its description and its input schema, the API's own bounds in it.

_SLUG_OF_API = "The API's slug, like 'email-finder' or 'linkedin-profile', as datacircle_list_apis gives it."
_PRICE = "perThousand is a plan's price of 1,000 items in dollars, pay the plan's whole price in its currency."
DESCRIPTIONS = {
    "datacircle_list_apis": "Lists the GTM APIs on Datacircle's sheet (https://datacircle.dev), with each API's slug, how "
                            "many plans and providers it has and its best plan. Call it to find an API's slug when you "
                            "don't know it. Gives total, the APIs found, and apis, up to limit of them.",
    "datacircle_best_plans": "Gives an API's plans best first by their score out of 100 on Datacircle's benchmark "
                             "(https://datacircle.dev), each with its rank, provider, plan, $ per 1,000 calls and its "
                             "page: the best providers of an API. " + _PRICE,
    "datacircle_search_plans": "Searches every plan on Datacircle's sheet (https://datacircle.dev), or one API's, by "
                               "provider or plan name, sorted by score, provider or price. Gives total, offset and up "
                               "to limit plans. " + _PRICE,
    "datacircle_provider": "Gives each API one provider sells and its rank on each on Datacircle's benchmark "
                           "(https://datacircle.dev), with its best plan there.",
}
SCHEMAS: Dict[str, Dict[str, Any]] = {
    "datacircle_list_apis": {"type": "object", "properties": {
        "query": {"type": "string", "description": "Words of the API's name or slug, like 'email' or 'linkedin "
                  "profile'. The APIs that hold the most of them come first, in the sheet's order when left out."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 200,
                  "description": "How many APIs to give. 20 when left out, of about 160."},
    }, "additionalProperties": False},
    "datacircle_best_plans": {"type": "object", "properties": {
        "slug": {"type": "string", "description": _SLUG_OF_API},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "How many plans to give. 10 when left out."},
    }, "required": ["slug"], "additionalProperties": False},
    "datacircle_search_plans": {"type": "object", "properties": {
        "api": {"type": "string", "description": "Only this API's plans, by its name or slug, like 'email-finder'. A "
                "provider's name goes in query, never here. Every API when left out."},
        "query": {"type": "string", "description": "Words that must all appear in the provider's or plan's name, like "
                  "'bright data'."},
        "sort": {"type": "string", "enum": ["score", "provider", "price"],
                 "description": "score (best first, the default), provider (A to Z) or price ($ per 1,000, lowest first)."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "How many plans to give. 10 when left out."},
        "offset": {"type": "integer", "minimum": 0, "description": "How many found plans to skip first. 0 when left out."},
    }, "additionalProperties": False},
    "datacircle_provider": {"type": "object", "properties": {
        "slug": {"type": "string", "description": "The provider's slug, like 'bright-data', or its name, like 'Bright "
                 "Data', which is made a slug."},
    }, "required": ["slug"], "additionalProperties": False},
}


# The functions. Called alone, each gives the API's answer as JSON text, or raises DatacircleError with its words.

def list_apis(query: Optional[str] = None, limit: Optional[int] = None) -> str:
    """The GTM APIs on Datacircle's sheet, those whose name or slug holds the most of query's words first: total and
    up to limit apis (1 to 200, 20 when left out), as JSON text."""
    return _text(_list_apis(query, limit))


def best_plans(slug: str, limit: Optional[int] = None) -> str:
    """An API's plans best first by their score out of 100, up to limit (1 to 50, 10 when left out), as JSON text."""
    return _text(_best_plans(slug, limit))


def search_plans(api: Optional[str] = None, query: Optional[str] = None, sort: Optional[str] = None,
                 limit: Optional[int] = None, offset: Optional[int] = None) -> str:
    """Every plan, or one API's, whose provider or plan name holds query's words, sorted by score, provider or price:
    total, offset and up to limit plans (1 to 50, 10 when left out), as JSON text."""
    return _text(_search_plans(api, query, sort, limit, offset))


def provider(slug: str) -> str:
    """Each API one provider sells, its rank on each and its best plan there, as JSON text. slug may be its name."""
    return _text(_provider(slug))


async def _list_apis_async(query: Optional[str] = None, limit: Optional[int] = None) -> str:
    return await anyio.to_thread.run_sync(list_apis, query, limit)


async def _best_plans_async(slug: str, limit: Optional[int] = None) -> str:
    return await anyio.to_thread.run_sync(best_plans, slug, limit)


async def _search_plans_async(api: Optional[str] = None, query: Optional[str] = None, sort: Optional[str] = None,
                              limit: Optional[int] = None, offset: Optional[int] = None) -> str:
    return await anyio.to_thread.run_sync(search_plans, api, query, sort, limit, offset)


async def _provider_async(slug: str) -> str:
    return await anyio.to_thread.run_sync(provider, slug)


def _made(make: Callable[..., Any], name: str, function: Callable[..., Any]) -> Any:
    return make(function, name=name, description=DESCRIPTIONS[name], input_schema=SCHEMAS[name])


datacircle_list_apis = _made(beta_tool, "datacircle_list_apis", list_apis)
datacircle_best_plans = _made(beta_tool, "datacircle_best_plans", best_plans)
datacircle_search_plans = _made(beta_tool, "datacircle_search_plans", search_plans)
datacircle_provider = _made(beta_tool, "datacircle_provider", provider)

TOOLS = {tool.name: tool for tool in (datacircle_list_apis, datacircle_best_plans, datacircle_search_plans,
                                      datacircle_provider)}
ASYNC_TOOLS = {name: _made(beta_async_tool, name, function) for name, function in (
    ("datacircle_list_apis", _list_apis_async), ("datacircle_best_plans", _best_plans_async),
    ("datacircle_search_plans", _search_plans_async), ("datacircle_provider", _provider_async))}
TOOL_NAMES = list(TOOLS)


def _chosen(tools: Mapping[str, Any], tool_names: Optional[Sequence[str]]) -> List[Any]:
    if isinstance(tool_names, str):
        raise TypeError("tool_names is a list of names, like ['datacircle_best_plans']")
    for name in tool_names or []:
        if name not in tools:
            raise ValueError(f"Datacircle has no tool {name!r}: its tools are {', '.join(TOOL_NAMES)}")
    return [tools[name] for name in (TOOL_NAMES if tool_names is None else tool_names)]


def datacircle_tools(tool_names: Optional[Sequence[str]] = None) -> List[Any]:
    """The tools, for client.beta.messages.tool_runner(tools=datacircle_tools()) or beside one's own:
    tools=[*datacircle_tools(), my_tool]. tool_names keeps some of them only, in their order here; all four when left out."""
    return _chosen(TOOLS, tool_names)


def datacircle_async_tools(tool_names: Optional[Sequence[str]] = None) -> List[Any]:
    """The same tools for an AsyncAnthropic client's tool runner: each runs its HTTPS GET in a worker thread."""
    return _chosen(ASYNC_TOOLS, tool_names)


def _field(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _calls(source: Any) -> List[Any]:
    """The calls of these tools in a reply: a Message or BetaMessage, its content, or a list of its blocks or their dicts."""
    if source is None:
        return []
    if _field(source, "content") is not None and not isinstance(source, (list, tuple)):
        source = _field(source, "content")
    blocks = source if isinstance(source, Iterable) and not isinstance(source, (str, bytes, Mapping)) else [source]
    return [block for block in blocks if _field(block, "type") == "tool_use" and _field(block, "name") in TOOLS]


def _why(error: Exception) -> str:
    """A failed call's words for the model: the API's or the check's, or which argument the SDK's validation refused."""
    cause = error.__cause__
    if cause is not None and hasattr(cause, "errors"):
        try:
            return f"{error}: " + "; ".join(f"{'.'.join(map(str, item['loc'])) or 'input'}: {item['msg']}"
                                            for item in cause.errors())
        except Exception:  # noqa: BLE001 - its plain words then
            pass
    return str(error) if isinstance(error, _api.DatacircleError) else f"{type(error).__name__}: {error}"


def _result(block: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"type": "tool_result", "tool_use_id": _field(block, "id")}
    try:
        result["content"] = TOOLS[_field(block, "name")].call(_field(block, "input"))
    except Exception as error:  # noqa: BLE001 - every failure is the call's result, as the tool runner gives it
        result.update(content=_why(error), is_error=True)
    return result


def tool_results(message: Any) -> List[Dict[str, Any]]:
    """For a loop of the app's own on client.messages.create(): the tool_result block of each call of these tools in
    a reply (a Message, its content, or a list of its blocks or their dicts), in their order, with its call's id. A
    refused call's block holds the refusal's words and is_error. Calls of other tools are left out, for the app to run."""
    return [_result(block) for block in _calls(message)]


async def atool_results(message: Any) -> List[Dict[str, Any]]:
    """tool_results() for an AsyncAnthropic client: the calls run at once, each in a worker thread, their blocks in the
    calls' order."""
    calls = _calls(message)
    results: List[Dict[str, Any]] = [{} for _ in calls]

    async def one(index: int, block: Any) -> None:
        results[index] = await anyio.to_thread.run_sync(_result, block)

    async with anyio.create_task_group() as group:
        for index, block in enumerate(calls):
            group.start_soon(one, index, block)
    return results


__all__: List[str] = ["ASYNC_TOOLS", "DESCRIPTIONS", "SCHEMAS", "TOOLS", "TOOL_NAMES", "atool_results", "best_plans",
                      "datacircle_async_tools", "datacircle_best_plans", "datacircle_list_apis", "datacircle_provider",
                      "datacircle_search_plans", "datacircle_tools", "list_apis", "provider", "search_plans",
                      "tool_results"]

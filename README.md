# anthropic-datacircle

Find the best GTM APIs. We benchmark all the GTM APIs and we put the results on our website.

Our website: https://datacircle.dev

This package gives the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) four tools that call Datacircle's open API, https://api.datacircle.dev/v1. You give them to the SDK's tool runner, which runs each call Claude makes. You need no key and no account for Datacircle.

| Tool | What the tool gives | Call |
| --- | --- | --- |
| `datacircle_list_apis` | Every GTM API on Datacircle, with the API's slug, number of plans, number of providers and best plan | `GET /apis` |
| `datacircle_best_plans` | One API's plans, sorted by score out of 100, best first. Each plan comes with a rank and a price per 1,000 calls | `GET /apis/{slug}/best` |
| `datacircle_search_plans` | Plans that match a provider or plan name, on all APIs or on one API. You can sort them by score, provider or price | `GET /plans` |
| `datacircle_provider` | The APIs a provider sells, and the provider's rank on each API | `GET /providers/{slug}` |

## Install

```bash
pip install anthropic-datacircle
```

anthropic-datacircle works with `anthropic` 0.68.0 and later, on Python 3.9 and later.

## Use the tools

Give the tools to the tool runner. The runner runs each call and sends the answer back to Claude:

```python
import anthropic
from anthropic_datacircle import datacircle_tools

client = anthropic.Anthropic()
runner = client.beta.messages.tool_runner(
    model="claude-opus-5-5",
    max_tokens=16000,
    tools=datacircle_tools(),
    messages=[{"role": "user", "content": "Who are the 3 best providers for the email-finder API?"}],
)
message = runner.until_done()
print(next(block.text for block in message.content if block.type == "text"))
```

You can also loop over the runner (`for message in runner:`) to see each message. With `stream=True`, the runner gives you a stream for each message.

With `AsyncAnthropic`, use `datacircle_async_tools()`. Each async tool runs its HTTPS request in a worker thread, so your event loop keeps running:

```python
import anthropic
from anthropic_datacircle import datacircle_async_tools

client = anthropic.AsyncAnthropic()
runner = client.beta.messages.tool_runner(
    model="claude-opus-5-5",
    max_tokens=16000,
    tools=datacircle_async_tools(),
    messages=[{"role": "user", "content": "What does Bright Data sell?"}],
)
message = await runner.until_done()
```

## Use them beside your own tools

You can put the Datacircle tools in the same list as yours: `tools=[*datacircle_tools(), get_weather]`. To give Claude only some of the tools, list their names: `datacircle_tools(["datacircle_best_plans"])`.

## Run the calls yourself

In a loop of your own on `client.messages.create()`, send each tool's definition with `tool.to_dict()`. Then `tool_results()` runs the Datacircle calls in a reply. It gives you one `tool_result` block for each call, with the call's id. Calls of your own tools are left out, for your code to run:

```python
import anthropic
from anthropic_datacircle import datacircle_tools, tool_results

client = anthropic.Anthropic()
tools = [tool.to_dict() for tool in datacircle_tools()]
messages = [{"role": "user", "content": "What does Bright Data sell?"}]
reply = client.messages.create(model="claude-opus-5-5", max_tokens=16000, tools=tools, messages=messages)

while reply.stop_reason == "tool_use":
    messages.append({"role": "assistant", "content": reply.content})
    messages.append({"role": "user", "content": tool_results(reply)})
    reply = client.messages.create(model="claude-opus-5-5", max_tokens=16000, tools=tools, messages=messages)

print(next(block.text for block in reply.content if block.type == "text"))
```

With `AsyncAnthropic`, use `await atool_results(reply)`. It runs the calls at the same time.

## Call a tool yourself

You can call each tool without Claude. You get the tool's answer as JSON text:

```python
import json
from anthropic_datacircle import best_plans

answer = json.loads(best_plans("email-finder", limit=3))
print(answer["plans"])
```

The other functions are `list_apis()`, `search_plans()` and `provider()`.

## What Claude reads

Each tool gives Claude a JSON object, as text. Each plan comes with a rank, provider, plan, score, price per 1,000 (`perThousand`, in dollars) and a page on datacircle.dev. That page lists every field.

You can give a provider's name in place of a slug. The tool turns `Bright Data` into `bright-data`.

When the API gives no data, such as for an unknown slug, the tool raises `DatacircleError` with the API's message. The runner sends that message to Claude as the call's result, with `is_error`. The error for an unknown API slug lists every valid slug. The tool checks each number first. For a number out of range, the tool raises the error and makes no API call. If the network fails, the error holds the network error's message. Claude can then call the tool again.

## Network

Each tool call makes one HTTPS GET request to https://api.datacircle.dev/v1 and nothing else. The request holds only the tool's own arguments: no key and nothing from your chats. The API takes up to 60 calls a minute from one IP address. The tool waits once, for a minute at most, when the API asks the tool to wait. Each call uses only Python's standard library. Nothing goes to disk.

## Links

- Datacircle: https://datacircle.dev
- The API: https://datacircle.dev/api
- The API's spec: https://api.datacircle.dev/v1/openapi.json
- What each key means: https://datacircle.dev/fields

MIT license.

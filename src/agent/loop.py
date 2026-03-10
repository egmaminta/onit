"""AgentLoop — the invariant while-tool_calls loop for OnIt.

This is the single place where the LLM is called and tool results are fed
back.  The loop never changes; new behaviour is added exclusively through
middleware hooks and virtual tools.

Supports both streaming and non-streaming completions, safety-queue
interruption, repeated-tool-call detection, and pluggable middleware at
every stage.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import types
import uuid
from typing import Any, Optional

from openai import AsyncOpenAI, OpenAIError, APITimeoutError

try:
    from ..type.agent_state import AgentState
    from ..middleware import MiddlewareChain
    from .tool_executor import ToolExecutor, strip_old_images
except (ImportError, ValueError):
    from type.agent_state import AgentState
    from middleware import MiddlewareChain
    from agent.tool_executor import ToolExecutor, strip_old_images

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Raw-tool-call parsing helpers (moved from chat.py)
# ────────────────────────────────────────────────────────────────

def _parse_tool_call_from_content(content: str, tool_registry) -> Optional[dict]:
    """Detect a raw JSON tool call in message content."""
    if not content or not tool_registry:
        return None
    text = content.split("</think>")[-1].strip() if "</think>" in content else content.strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return _parse_truncated_tool_call(text[start:], tool_registry)
    try:
        obj = json.loads(text[start:end])
    except json.JSONDecodeError:
        return _parse_truncated_tool_call(text[start:end], tool_registry)
    if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
        if obj["name"] not in tool_registry.tools:
            return None
        return obj
    return None


def _parse_truncated_tool_call(text: str, tool_registry) -> Optional[dict]:
    """Attempt to extract a tool call from truncated/malformed JSON."""
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if not name_match:
        return None
    tool_name = name_match.group(1)
    if tool_name not in tool_registry.tools:
        return None
    args_match = re.search(r'"arguments"\s*:\s*\{', text)
    if not args_match:
        return {"name": tool_name, "arguments": {}}
    args_start = args_match.end() - 1
    args_text = text[args_start:]
    depth = 0
    in_str = False
    esc = False
    last_valid = -1
    for i, ch in enumerate(args_text):
        if esc:
            esc = False
            continue
        if ch == '\\' and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                last_valid = i + 1
                break
    if last_valid > 0:
        try:
            args = json.loads(args_text[:last_valid])
            return {"name": tool_name, "arguments": args}
        except json.JSONDecodeError:
            pass
    return {"name": tool_name, "arguments": {}}


def _looks_like_raw_tool_call(content: str) -> bool:
    """Check if content looks like a raw tool-call JSON."""
    if not content:
        return False
    text = content.split("</think>")[-1].strip() if "</think>" in content else content.strip()
    return bool(
        re.search(r'"name"\s*:\s*"[^"]+"', text)
        and re.search(r'"arguments"\s*:', text)
    )


# ────────────────────────────────────────────────────────────────
# Model host helpers (kept from chat.py)
# ────────────────────────────────────────────────────────────────

def _resolve_api_key(host: str, host_key: str = "EMPTY") -> str:
    import os
    if "openrouter.ai" in host:
        if host_key and host_key != "EMPTY":
            return host_key
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError(
                "OpenRouter requires an API key. Set it via:\n"
                "  - serving.host_key in the config YAML\n"
                "  - OPENROUTER_API_KEY environment variable"
            )
        return key
    return host_key


async def _resolve_model_id(client: AsyncOpenAI, host: str) -> str:
    models = await client.models.list()
    if not models.data:
        raise ValueError(f"No models available at {host}")
    model_id = models.data[0].id
    logger.info("Auto-detected model: %s from %s", model_id, host)
    return model_id


# ────────────────────────────────────────────────────────────────
# AgentLoop
# ────────────────────────────────────────────────────────────────

class AgentLoop:
    """The core agent loop: call model → execute tools → repeat.

    This loop is *invariant* — it never changes.  All new behaviour is
    injected through the ``MiddlewareChain`` and virtual tools.
    """

    def __init__(
        self,
        tool_registry: Any,
        middleware_chain: MiddlewareChain | None = None,
    ):
        self.tool_registry = tool_registry
        self.chain = middleware_chain or MiddlewareChain()
        self.tool_executor = ToolExecutor(tool_registry, self.chain)

    # ────────────────────────────────────────────────────────────
    # Public entry point
    # ────────────────────────────────────────────────────────────

    async def run(self, state: AgentState) -> Optional[str]:
        """Run the agent loop to completion, returning the final text response.

        ``state`` must be pre-populated with at least ``messages``,
        ``host``, ``safety_queue``, and ``tools``.
        """
        # Initialize middleware (one-time setup)
        await self.chain.initialize(state)

        try:
            return await self._loop(state)
        finally:
            await self.chain.shutdown(state)

    # ────────────────────────────────────────────────────────────
    # Internal loop
    # ────────────────────────────────────────────────────────────

    async def _loop(self, state: AgentState) -> Optional[str]:
        host = state["host"]
        host_key = state.get("host_key", "EMPTY")
        safety_queue: asyncio.Queue = state["safety_queue"]
        chat_ui = state.get("chat_ui")
        verbose = state.get("verbose", False)
        think = state.get("think", False)
        stream = state.get("stream", False)
        max_tokens = state.get("max_tokens", 8192)
        max_iterations = state.get("max_iterations", 100)
        max_repeated = state.get("max_repeated_tool_calls", 30)

        messages = state["messages"]

        # Resolve model
        api_key = _resolve_api_key(host, host_key)
        client = AsyncOpenAI(base_url=host, api_key=api_key)
        model = state.get("model") or await _resolve_model_id(client, host)
        state["model"] = model

        if chat_ui:
            chat_ui.model_name = model
            chat_ui.add_log(f"Starting chat with model: {model}", level="info")
        elif verbose:
            print(f"Starting chat with model: {model}")

        # Expose tool_registry in state so middleware (e.g. SubAgentMiddleware) can access it
        state["tool_registry"] = self.tool_registry

        tool_call_history: list[tuple[str, str]] = state.get("tool_call_history", [])
        state["tool_call_history"] = tool_call_history
        _token_retried = False  # allow one retry after token-limit compaction
        iteration = 0

        while True:
            iteration += 1
            state["iteration"] = iteration

            if iteration > max_iterations:
                msg = "I am sorry \U0001f60a. Could you try to rephrase or provide additional details?"
                if chat_ui:
                    chat_ui.add_log(f"Chat loop exceeded {max_iterations} iterations, stopping.", level="warning")
                elif verbose:
                    print(f"Chat loop exceeded {max_iterations} iterations, stopping.")
                return msg

            # Strip stale image payloads
            strip_old_images(messages)

            # ── before_model middleware ──────────────────────────
            state = await self.chain.before_model(state)
            # Collect tools: MCP tools + any middleware-injected tools
            tools = state.get("tools", [])

            # ── safety check ────────────────────────────────────
            if not safety_queue.empty():
                logger.warning("Safety queue triggered before API call, exiting.")
                return None

            # ── LLM call ────────────────────────────────────────
            try:
                completion_kwargs = dict(
                    model=model,
                    messages=messages,
                    stream=stream,
                    tool_choice="auto",
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=max_tokens,
                    extra_body={
                        "top_k": 20,
                        "repetition_penalty": 1.05,
                        "chat_template_kwargs": {"enable_thinking": think},
                    },
                )
                if stream:
                    completion_kwargs["stream_options"] = {"include_usage": True}
                if tools:
                    completion_kwargs["tools"] = tools

                chat_completion = await client.chat.completions.create(**completion_kwargs)

                # ── Process response ────────────────────────────
                if stream:
                    result = await self._handle_streaming(
                        chat_completion, safety_queue, chat_ui, think, messages
                    )
                else:
                    result = self._handle_non_streaming(chat_completion, chat_ui)

                # after_model middleware
                result = await self.chain.after_model(state, result)

                await asyncio.sleep(0.1)
                if not safety_queue.empty():
                    logger.warning("Safety queue triggered after API call, exiting.")
                    return None

            except APITimeoutError:
                timeout_val = state.get("timeout")
                error_message = f"Request to {host} timed out after {timeout_val} seconds."
                logger.error(error_message)
                if chat_ui:
                    chat_ui.add_log(error_message, level="error")
                elif verbose:
                    print(error_message)
                return None
            except OpenAIError as e:
                error_str = str(e).lower()
                _token_keywords = ("token", "context length", "maximum context",
                                   "too long", "context_length_exceeded", "max_tokens",
                                   "reduce your prompt", "model's maximum")
                if not _token_retried and any(kw in error_str for kw in _token_keywords):
                    _token_retried = True
                    logger.warning("Token limit hit, attempting emergency compaction: %s", e)
                    # Emergency compact: keep system message + last 4 messages
                    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
                        system_msg = [messages[0]]
                    else:
                        system_msg = []
                    tail = messages[-4:]
                    messages[:] = system_msg + [
                        {"role": "user", "content": "[Prior conversation truncated due to context length limit]"}
                    ] + tail
                    state["messages"] = messages
                    if chat_ui:
                        chat_ui.add_log("Context too large — compacted conversation and retrying.", level="warning")
                    elif verbose:
                        print("Context too large — compacted conversation and retrying.")
                    continue
                error_message = f"Error communicating with {host}: {e}."
                logger.error(error_message)
                if chat_ui:
                    chat_ui.add_log(error_message, level="warning")
                elif verbose:
                    print(error_message)
                return None
            except Exception as e:
                error_message = f"Unexpected error: {e}"
                logger.error(error_message)
                if chat_ui:
                    chat_ui.add_log(error_message, level="error")
                elif verbose:
                    print(error_message)
                return None

            _content = result["content"]
            _tool_calls = result["tool_calls"]
            _message_for_history = result["message_for_history"]
            _full_reasoning = result.get("full_reasoning", "")
            _full_content = result.get("full_content", "")

            # ── Report usage ────────────────────────────────────
            usage = result.get("usage")
            if chat_ui and hasattr(chat_ui, "report_usage") and usage:
                chat_ui.report_usage(
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0,
                )

            # ── No tool calls → final response ─────────────────
            if _tool_calls is None or len(_tool_calls) == 0:
                last_response = _content

                # Check for raw JSON tool call in content
                raw_tool = _parse_tool_call_from_content(last_response, self.tool_registry)
                if raw_tool:
                    function_name = raw_tool["name"]
                    function_arguments = raw_tool["arguments"]
                    synthetic_id = f"call_{uuid.uuid4().hex[:24]}"
                    messages.append({"role": "assistant", "content": last_response})

                    tool_msg = await self.tool_executor.execute(
                        state, function_name, function_arguments, synthetic_id
                    )
                    messages.append(tool_msg)

                    # Repeated call detection
                    call_key = (function_name, json.dumps(function_arguments, sort_keys=True))
                    tool_call_history.append(call_key)
                    if tool_call_history.count(call_key) >= max_repeated:
                        if chat_ui:
                            chat_ui.add_log(
                                f"Repeated tool call: {function_name} ×{tool_call_history.count(call_key)}",
                                level="warning",
                            )
                        return "I am sorry \U0001f60a. Could you try to rephrase or provide additional details?"
                    continue

                # Guard against raw tool-call JSON that couldn't be parsed
                if _looks_like_raw_tool_call(last_response):
                    if chat_ui:
                        chat_ui.add_log("Unparseable raw tool-call JSON, retrying without tools.", level="warning")
                    elif verbose:
                        print("Unparseable raw tool-call JSON, retrying without tools.")
                    messages.append({"role": "assistant", "content": last_response})
                    messages.append({"role": "user", "content": "Please provide your answer as plain text, not as a JSON tool call."})
                    continue

                # Strip thinking tags for console mode
                if last_response and "</think>" in last_response:
                    if not chat_ui:
                        last_response = last_response.split("</think>")[1]

                # Fallback: surface reasoning if answer is empty
                if not last_response or not last_response.strip():
                    if _full_reasoning and _full_reasoning.strip():
                        last_response = _full_reasoning.strip()
                    elif _full_content and "<think>" in _full_content:
                        think_body = _full_content.split("<think>", 1)[1].split("</think>", 1)[0].strip()
                        if think_body:
                            last_response = think_body

                return last_response

            # ── Tool calls → execute and loop ───────────────────
            messages.append(_message_for_history)

            for tool in _tool_calls:
                await asyncio.sleep(0.1)
                if not safety_queue.empty():
                    if verbose:
                        print("Safety queue triggered, exiting chat loop.")
                    return None

                function_name = tool.function.name
                function_arguments = json.loads(tool.function.arguments)

                tool_msg = await self.tool_executor.execute(
                    state, function_name, function_arguments, tool.id
                )
                messages.append(tool_msg)

                # Repeated call detection
                call_key = (function_name, json.dumps(function_arguments, sort_keys=True))
                tool_call_history.append(call_key)
                if tool_call_history.count(call_key) >= max_repeated:
                    if chat_ui:
                        chat_ui.add_log(
                            f"Repeated tool call: {function_name} ×{tool_call_history.count(call_key)}",
                            level="warning",
                        )
                    elif verbose:
                        print(f"Repeated tool call: {function_name} ×{tool_call_history.count(call_key)}")
                    return "I am sorry \U0001f60a. Could you try to rephrase or provide additional details?"

    # ────────────────────────────────────────────────────────────
    # Streaming handler
    # ────────────────────────────────────────────────────────────

    async def _handle_streaming(
        self,
        chat_completion,
        safety_queue: asyncio.Queue,
        chat_ui: Any,
        think: bool,
        messages: list,
    ) -> dict:
        """Iterate over streaming chunks and return a unified result dict."""
        _full_content = ""
        _full_reasoning = ""
        _full_tool_calls: dict = {}
        _ui_streaming = False
        _in_think = think
        _stream_usage = None

        async for chunk in chat_completion:
            if not safety_queue.empty():
                if _ui_streaming and chat_ui:
                    chat_ui.stream_end()
                return {
                    "content": None, "tool_calls": None,
                    "message_for_history": {"role": "assistant", "content": ""},
                    "full_reasoning": "", "full_content": "",
                    "usage": None,
                }
            if hasattr(chunk, "usage") and chunk.usage:
                _stream_usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Accumulate tool-call deltas
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in _full_tool_calls:
                        _full_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        _full_tool_calls[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            _full_tool_calls[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            _full_tool_calls[idx]["arguments"] += tc.function.arguments

            # Reasoning tokens
            reasoning_tok = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if reasoning_tok and not _full_tool_calls:
                _full_reasoning += reasoning_tok
                if chat_ui:
                    if not _ui_streaming:
                        chat_ui.stream_start()
                        _ui_streaming = True
                    chat_ui.stream_think_token(reasoning_tok)

            # Content tokens
            if delta.content:
                token = delta.content
                _full_content += token
                if chat_ui and not _full_tool_calls:
                    if _in_think:
                        if "<think>" not in _full_content:
                            _in_think = False
                            if not _ui_streaming:
                                chat_ui.stream_start()
                                _ui_streaming = True
                            chat_ui.stream_token(token)
                        elif "</think>" in _full_content:
                            _in_think = False
                            pre_close = token.split("</think>", 1)[0]
                            if pre_close:
                                chat_ui.stream_think_token(pre_close)
                            post_think = _full_content.split("</think>", 1)[1]
                            if post_think:
                                if not _ui_streaming:
                                    chat_ui.stream_start()
                                    _ui_streaming = True
                                chat_ui.stream_token(post_think)
                        else:
                            if not _ui_streaming:
                                chat_ui.stream_start()
                                _ui_streaming = True
                            chat_ui.stream_think_token(token.replace("<think>", ""))
                    else:
                        if not _ui_streaming:
                            chat_ui.stream_start()
                            _ui_streaming = True
                        chat_ui.stream_token(token)

        if _ui_streaming and chat_ui:
            chat_ui.stream_end()

        # Report streaming usage
        if chat_ui and hasattr(chat_ui, "report_usage") and _stream_usage:
            chat_ui.report_usage(
                prompt_tokens=getattr(_stream_usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(_stream_usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(_stream_usage, "total_tokens", 0) or 0,
            )

        # Reconstruct unified variables
        if _full_tool_calls:
            _tool_call_objs = [
                types.SimpleNamespace(
                    id=v["id"],
                    function=types.SimpleNamespace(name=v["name"], arguments=v["arguments"]),
                )
                for v in _full_tool_calls.values()
            ]
            return {
                "content": None,
                "tool_calls": _tool_call_objs,
                "message_for_history": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": v["id"],
                            "type": "function",
                            "function": {"name": v["name"], "arguments": v["arguments"]},
                        }
                        for v in _full_tool_calls.values()
                    ],
                },
                "full_reasoning": _full_reasoning,
                "full_content": _full_content,
                "usage": _stream_usage,
            }
        else:
            return {
                "content": _full_content,
                "tool_calls": None,
                "message_for_history": {"role": "assistant", "content": _full_content},
                "full_reasoning": _full_reasoning,
                "full_content": _full_content,
                "usage": _stream_usage,
            }

    # ────────────────────────────────────────────────────────────
    # Non-streaming handler
    # ────────────────────────────────────────────────────────────

    def _handle_non_streaming(self, chat_completion, chat_ui: Any) -> dict:
        """Extract from a non-streaming response into the unified dict."""
        if not chat_completion.choices:
            logger.warning("API returned empty choices list.")
            return {
                "content": None, "tool_calls": None,
                "message_for_history": {"role": "assistant", "content": ""},
                "full_reasoning": "", "full_content": "",
                "usage": getattr(chat_completion, "usage", None),
            }
        _msg = chat_completion.choices[0].message
        _content = _msg.content
        _tool_calls = _msg.tool_calls if _msg.tool_calls else None

        _ns_reasoning = getattr(_msg, "reasoning_content", None) or getattr(_msg, "reasoning", None)
        if _ns_reasoning and chat_ui and hasattr(chat_ui, "_non_stream_reasoning"):
            chat_ui._non_stream_reasoning = _ns_reasoning

        usage = getattr(chat_completion, "usage", None)

        return {
            "content": _content,
            "tool_calls": _tool_calls,
            "message_for_history": _msg,
            "full_reasoning": _ns_reasoning or "",
            "full_content": _content or "",
            "usage": usage,
        }

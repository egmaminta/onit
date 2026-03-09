"""
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Chat function — thin wrapper around AgentLoop.

All loop logic, tool execution, streaming, and middleware hooks live in
``agent.loop.AgentLoop``.  This module builds the initial message list and
AgentState, then delegates to the loop.

Backward-compatible: the ``chat()`` signature is unchanged.
"""

import asyncio
import base64
import logging
import os
from typing import List, Optional, Any

logger = logging.getLogger(__name__)


def _get_agent_imports():
    """Lazy import to support both `src.` and direct `model.` import contexts."""
    try:
        from ...agent.loop import AgentLoop, _resolve_api_key, _resolve_model_id, _parse_tool_call_from_content
        from ...middleware import MiddlewareChain
    except (ImportError, ValueError):
        from agent.loop import AgentLoop, _resolve_api_key, _resolve_model_id, _parse_tool_call_from_content
        from middleware import MiddlewareChain
    return AgentLoop, MiddlewareChain, _resolve_api_key, _resolve_model_id, _parse_tool_call_from_content


# Re-export helpers that tests import from this module
try:
    from ...agent.loop import _resolve_api_key, _resolve_model_id, _parse_tool_call_from_content  # noqa: F401
except (ImportError, ValueError):
    from agent.loop import _resolve_api_key, _resolve_model_id, _parse_tool_call_from_content  # noqa: F401


async def chat(host: str = "http://127.0.0.1:8001/v1",
         host_key: str = "EMPTY",
         instruction: str = "Tell me more about yourself.",
         images: List[str]|str = None,
         tool_registry: Optional[Any] = None,
         timeout: int = None,
         stream: bool = False,
         think: bool = False,
         safety_queue: Optional[asyncio.Queue] = None,
         **kwargs) -> Optional[str]:

    tools = tool_registry.get_tool_items() if tool_registry else []
    chat_ui = kwargs.get('chat_ui')
    verbose = kwargs.get('verbose', False)
    data_path = kwargs.get('data_path', '')
    max_tokens = kwargs.get('max_tokens', 8192)
    memories = kwargs.get('memories')
    prompt_intro = kwargs.get('prompt_intro', "I am a helpful AI assistant. My name is OnIt.")
    model = kwargs.get('model')

    # ── Build initial message list (same logic as before) ──────
    images_bytes = []
    if isinstance(images, list):
        for image_path in images:
            if os.path.exists(image_path):
                with open(image_path, 'rb') as image_file:
                    images_bytes.append(base64.b64encode(image_file.read()).decode('utf-8'))
            else:
                if chat_ui:
                    chat_ui.add_log(f"Image file {image_path} not found, proceeding without this image.", level="warning")
                elif verbose:
                    print(f"Image file {image_path} not found, proceeding without this image.")
    elif isinstance(images, str):
        image_path = images
        if os.path.exists(image_path):
            with open(image_path, 'rb') as image_file:
                images_bytes = [base64.b64encode(image_file.read()).decode('utf-8')]
        else:
            if chat_ui:
                chat_ui.add_log(f"Image file {image_path} not found, proceeding without this image.", level="warning")
            elif verbose:
                print(f"Image file {image_path} not found, proceeding without this image.")

    if images_bytes:
        messages = [{
            "role": "system",
            "content": (
                f"{prompt_intro} "
                "You are an expert vision-language assistant. Your task is to analyze images with high precision, "
                "reasoning step-by-step about visual elements and their spatial relationships (e.g., coordinates, "
                "relative positions like left/right/center). Always verify visual evidence before concluding. "
                "If a task requires external data, calculation, or specific actions beyond visual description, "
                "use the provided tools. Be concise, objective, and format your tool calls strictly according to schema."
            )
        }]
    else:
        messages = [{"role": "system", "content": prompt_intro}]

    # Inject session history BEFORE the current instruction so the model
    # sees prior context first and treats the latest user message as the
    # one to respond to.
    session_history = kwargs.get('session_history')
    if session_history:
        for entry in session_history:
            messages.append({"role": "user", "content": entry["task"]})
            messages.append({"role": "assistant", "content": entry["response"]})

    # Current instruction goes last so the model responds to it
    if images_bytes:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images_bytes[0]}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": instruction})

    # ── Build AgentState ───────────────────────────────────────
    state = {
        "messages": messages,
        "system_prompt": prompt_intro,
        "tools": tools,
        "host": host,
        "host_key": host_key,
        "model": model,
        "think": think,
        "stream": stream,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "safety_queue": safety_queue,
        "chat_ui": chat_ui,
        "verbose": verbose,
        "data_path": data_path,
    }

    # ── Build middleware chain (empty for now — Phase 3 adds real ones)
    AgentLoop, MiddlewareChain, _, _, _ = _get_agent_imports()
    middleware_chain = kwargs.get("middleware_chain") or MiddlewareChain()

    # ── Run the agent loop ─────────────────────────────────────
    loop = AgentLoop(tool_registry=tool_registry, middleware_chain=middleware_chain)
    return await loop.run(state)

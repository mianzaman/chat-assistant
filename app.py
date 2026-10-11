import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypedDict

import httpx
import streamlit as st
from dotenv import load_dotenv
from openrouter import OpenRouter, components, errors

st.set_page_config(
    page_title="Chat assistant",
    page_icon=":material/chat:",
    layout="centered",
)

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = "nvidia/nemotron-3.5-lightning:free"


class StreamedMessageParts(TypedDict):
    reasoning: list[str]
    reasoning_details: list[dict[str, object]]


def stream_response(
    chunks: Iterable[components.ChatStreamChunk],
    message_parts: StreamedMessageParts,
) -> Iterator[str]:
    for chunk in chunks:
        if chunk.error:
            raise RuntimeError(f"OpenRouter stream error: {chunk.error}")
        for choice in chunk.choices:
            delta = choice.delta
            if delta.reasoning:
                message_parts["reasoning"].append(delta.reasoning)
            if delta.reasoning_details:
                message_parts["reasoning_details"].extend(
                    detail.model_dump(exclude_none=True)
                    for detail in delta.reasoning_details
                )
            if delta.content:
                yield delta.content


st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending_prompt", None)


def start_new_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending_prompt = None


def queue_prompt(prompt: str) -> None:
    st.session_state.pending_prompt = prompt


api_key = os.getenv("OPENROUTER_API_KEY")

with st.sidebar:
    st.title("Chat assistant")
    st.button(
        "New chat",
        icon=":material/edit_square:",
        on_click=start_new_chat,
        width="stretch",
    )
    st.caption(f"Model: {MODEL}")
    if not api_key:
        st.caption("Add an OpenRouter API key to start chatting.")

if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("content"):
                st.markdown(message["content"])
else:
    st.markdown("## What would you like to work on?", text_alignment="center")
    st.caption("Start with a question, an idea, or a rough draft.")
    st.space("medium")

    suggestion_columns = st.columns(3)
    suggestions = (
        "Explain a tricky idea",
        "Draft a thoughtful message",
        "Help me plan a project",
    )
    for column, suggestion in zip(suggestion_columns, suggestions):
        with column:
            st.button(
                suggestion,
                on_click=queue_prompt,
                args=(suggestion,),
                width="stretch",
            )

prompt = st.chat_input("Message the assistant")
if prompt is None:
    prompt = st.session_state.pending_prompt
st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            st.error(
                "OpenRouter is not configured. Add OPENROUTER_API_KEY to your "
                ".env file."
            )
        else:
            message_parts: StreamedMessageParts = {
                "reasoning": [],
                "reasoning_details": [],
            }
            try:
                with OpenRouter(
                    api_key=api_key,
                    timeout_ms=120_000,
                    x_open_router_title="Chat assistant",
                ) as client:
                    with st.spinner("Thinking..."):
                        response = client.chat.send(
                            model=MODEL,
                            messages=st.session_state.messages,
                            reasoning={"effort": "medium"},
                            stream=True,
                        )
                    content = st.write_stream(
                        stream_response(response, message_parts)
                    )
                    assistant_message: dict[str, object] = {
                        "role": "assistant",
                        "content": content,
                    }
                    if message_parts["reasoning"]:
                        assistant_message["reasoning"] = "".join(
                            message_parts["reasoning"]
                        )
                    if message_parts["reasoning_details"]:
                        assistant_message["reasoning_details"] = message_parts[
                            "reasoning_details"
                        ]
            except RuntimeError as error:
                st.error(str(error))
            except errors.OpenRouterError as error:
                st.error(f"OpenRouter request failed: {error}")
            except httpx.HTTPError:
                st.error(
                    "Could not connect to OpenRouter. Check your network connection "
                    "and try again."
                )
            except (IndexError, AttributeError):
                st.error("OpenRouter returned an unexpected response. Please try again.")
            else:
                st.session_state.messages.append(assistant_message)

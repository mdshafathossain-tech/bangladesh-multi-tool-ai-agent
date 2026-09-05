"""
app.py
======

Streamlit web-based chat UI for the Bangladesh Multi-Tool AI Agent.

This file does NOT redefine any agent, tool, or routing logic — it imports
and reuses everything from `main.py` (which in turn uses `db_tools.py`):

    - build_agent()          -> constructs the LangChain agent (Gemini + tools)
    - _extract_tool_names()  -> reads which tool(s) the agent invoked
    - _extract_final_answer()-> pulls the final text answer out of the result
    - _format_routing_label()-> turns tool name(s) into a friendly CLI/UI label

`app.py` is purely a UI layer on top of that existing logic.

--------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------
1. Make sure your environment variables are set (same as for main.py):

    export GEMINI_API_KEY="your-gemini-api-key"
    export TAVILY_API_KEY="your-tavily-api-key"

   ...or place them in a `.env` file in the project root (main.py already
   loads it automatically via python-dotenv).

2. Make sure institutions.db, hospitals.db, and restaurants.db exist
   (run `python build_databases.py` first if they don't).

3. Install Streamlit if you haven't already (see requirements.txt note
   below), then run:

    streamlit run app.py

4. Your browser will open automatically at http://localhost:8501
"""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import HumanMessage

from main import (
    build_agent,
    _extract_tool_names,
    _extract_final_answer,
    _format_routing_label,
)


# --------------------------------------------------------------------------- #
# Page config & header
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Bangladesh Multi-Tool AI Agent",
    page_icon="🇧🇩",
    layout="centered",
)

st.title("🇧🇩 Bangladesh Multi-Tool AI Agent")
st.caption(
    "Ask about **Institutions** 🏫 · **Hospitals** 🏥 · **Restaurants** 🍽️ · "
    "or general **Web Search** 🌐 — the agent automatically routes your "
    "question to the right tool."
)


# --------------------------------------------------------------------------- #
# Agent initialization (cached — built once per server process, not per message)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Building agent (this happens once)...")
def get_agent():
    """
    Cached wrapper around main.build_agent().

    build_agent() calls sys.exit(1) via main._require_env_var() if
    GEMINI_API_KEY / TAVILY_API_KEY are missing, which would otherwise kill
    the whole Streamlit server process. We catch that here and convert it
    into a normal exception so the app can show a clean error message
    instead of crashing.
    """
    try:
        return build_agent()
    except SystemExit as exc:
        raise RuntimeError(
            "Missing GEMINI_API_KEY and/or TAVILY_API_KEY. Set them as "
            "environment variables or in a .env file, then restart the app."
        ) from exc


try:
    agent = get_agent()
except RuntimeError as exc:
    st.error(f"⚠️ {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001 - surface any other startup error clearly
    st.error(f"⚠️ Failed to build the agent: {exc}")
    st.stop()


# --------------------------------------------------------------------------- #
# Session state (chat history)
# --------------------------------------------------------------------------- #

if "messages" not in st.session_state:
    # Each entry: {"role": "user" | "assistant", "content": str, "routing": str | None}
    st.session_state.messages = []


# --------------------------------------------------------------------------- #
# Render existing chat history
# --------------------------------------------------------------------------- #

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("routing"):
            st.info(f"🔍 [Routing Info] Tool Selected: {message['routing']}")
        st.markdown(message["content"])


# --------------------------------------------------------------------------- #
# Chat input & response handling
# --------------------------------------------------------------------------- #

user_input = st.chat_input("Ask a question about Bangladesh institutions, hospitals, restaurants, or anything else...")

if user_input:
    # 1. Store and display the user's message.
    st.session_state.messages.append({"role": "user", "content": user_input, "routing": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Invoke the agent and render the assistant's response.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
                tool_names = _extract_tool_names(result)
                answer = _extract_final_answer(result)
            except Exception as exc:  # noqa: BLE001 - keep the app alive on any error
                tool_names = []
                answer = f"An error occurred while answering: {exc}"

        routing_label = _format_routing_label(tool_names)
        st.info(f"🔍 [Routing Info] Tool Selected: {routing_label}")
        st.markdown(answer)

    # 3. Persist the assistant's message (with its routing label) for future reruns.
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "routing": routing_label}
    )

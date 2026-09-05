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

import time

import streamlit as st
from langchain_core.messages import HumanMessage

from main import (
    build_agent,
    _extract_tool_names,
    _extract_final_answer,
    _format_routing_label,
)


# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Bangladesh Multi-Tool AI Agent",
    page_icon="🇧🇩",
    layout="centered",
)

USER_AVATAR = "🧑‍💻"
ASSISTANT_AVATAR = "🇧🇩"

EXAMPLE_QUESTIONS = [
    "How many restaurants are in the database?",
    "Which restaurant has the highest rating?",
    "What is the role of DGHS in Bangladesh?",
]


# --------------------------------------------------------------------------- #
# Custom styling
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7fdf9 0%, #ffffff 40%);
    }
    .bd-header {
        background: linear-gradient(90deg, #006a4e 0%, #00875a 100%);
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(0, 106, 78, 0.25);
    }
    .bd-header h1 {
        color: white;
        margin: 0;
        font-size: 1.6rem;
    }
    .bd-header p {
        color: #eafff3;
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
    }
    .bd-tool-chip {
        display: inline-block;
        background: #ffffff;
        border: 1px solid #d7ecdf;
        border-radius: 999px;
        padding: 0.2rem 0.7rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
        font-size: 0.8rem;
        color: #0a5c3e;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.25rem 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <div class="bd-header">
        <h1>🇧🇩 Bangladesh Multi-Tool AI Agent</h1>
        <p>Ask about institutions, hospitals, restaurants, or anything else — the agent
        automatically routes your question to the right tool.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <span class="bd-tool-chip">🏫 Institutions</span>
    <span class="bd-tool-chip">🏥 Hospitals</span>
    <span class="bd-tool-chip">🍽️ Restaurants</span>
    <span class="bd-tool-chip">🌐 Web Search</span>
    """,
    unsafe_allow_html=True,
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
    # Each entry: {"role": "user"|"assistant", "content": str, "routing": str|None}
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --------------------------------------------------------------------------- #
# Sidebar — quick actions & example questions (dynamic elements)
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.subheader("⚡ Quick questions")
    st.caption("Click one to ask it instantly.")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.toggle(
        "🧪 Mock Mode (no Gemini/Tavily calls)",
        key="mock_mode",
        help=(
            "Turn this on to test the chat UI, routing banner, and styling "
            "with instant scripted answers — useful when your Gemini free-tier "
            "daily quota (20 requests/day) is exhausted. Turn it off to get "
            "real answers again."
        ),
    )
    if st.session_state.get("mock_mode"):
        st.caption("🧪 Mock Mode is ON — answers below are scripted, not real.")

    st.divider()
    st.caption(
        "💡 Tip: general/background questions (routed to Web Search) use "
        "fewer Gemini calls than database questions, so they're gentler on "
        "a free-tier daily quota."
    )


# --------------------------------------------------------------------------- #
# Render existing chat history
# --------------------------------------------------------------------------- #

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "assistant" and message.get("routing"):
            st.info(f"🔍 [Routing Info] Tool Selected: {message['routing']}")
        st.markdown(message["content"])


# --------------------------------------------------------------------------- #
# Helper: turn a finished answer into a light "typing" stream for st.write_stream
# --------------------------------------------------------------------------- #

def _stream_text(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)


# --------------------------------------------------------------------------- #
# Mock mode — scripted, zero-API-call responses for UI/UX testing only.
# Makes ZERO network calls, so it never touches your Gemini/Tavily quota.
# --------------------------------------------------------------------------- #

def _mock_invoke(question: str) -> dict:
    from langchain_core.messages import AIMessage

    q = question.lower()
    if "restaurant" in q or "cuisine" in q or "rating" in q:
        tool, answer = "restaurants_db_tool", (
            "(Mock) The highest-rated restaurant in the sample data is "
            "'Sultan's Dine' with a rating of 4.8."
        )
    elif "hospital" in q or "bed" in q or "doctor" in q:
        tool, answer = "hospitals_db_tool", (
            "(Mock) There are 128 hospitals in the database matching that "
            "description."
        )
    elif "university" in q or "college" in q or "institut" in q or "school" in q:
        tool, answer = "institutions_db_tool", (
            "(Mock) There are 42 government institutions matching that "
            "description."
        )
    elif any(kw in q for kw in ("dghs", "policy", "role of", "what is", "history", "news")):
        tool, answer = "tavily_search", (
            "(Mock) This would normally be answered via a live web search — "
            "no real search was performed in Mock Mode."
        )
    else:
        tool, answer = None, "(Mock) This is a scripted demo answer for UI testing."

    messages = []
    if tool:
        messages.append(AIMessage(content="", tool_calls=[{"name": tool, "args": {"question": question}, "id": "mock-1"}]))
    messages.append(AIMessage(content=answer))
    return {"messages": messages}


def _handle_question(user_input: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_input, "routing": None})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        agent_error = False
        with st.spinner("Thinking..."):
            try:
                if st.session_state.get("mock_mode"):
                    result = _mock_invoke(user_input)
                else:
                    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
                tool_names = _extract_tool_names(result)
                answer = _extract_final_answer(result)
            except Exception as exc:  # noqa: BLE001 - keep the app alive on any error
                agent_error = True
                tool_names = []
                answer = f"An error occurred while answering: {exc}"

        # Distinguish "the model genuinely answered without a tool" from
        # "a tool call never completed because the request itself failed"
        # (e.g. a quota/rate-limit error) — both look like an empty
        # tool_names list, but they mean very different things to the user.
        routing_label = (
            "⚠️ Error (request failed before routing completed)"
            if agent_error else _format_routing_label(tool_names)
        )
        st.info(f"🔍 [Routing Info] Tool Selected: {routing_label}")
        st.write_stream(_stream_text(answer))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "routing": routing_label}
    )


# --------------------------------------------------------------------------- #
# Chat input & response handling
# --------------------------------------------------------------------------- #

# A sidebar button sets this; handle it before the manual chat_input so a
# click behaves just like typing the same question.
if st.session_state.pending_question:
    pending = st.session_state.pending_question
    st.session_state.pending_question = None
    _handle_question(pending)

user_input = st.chat_input(
    "Ask a question about Bangladesh institutions, hospitals, restaurants, or anything else..."
)
if user_input:
    _handle_question(user_input)

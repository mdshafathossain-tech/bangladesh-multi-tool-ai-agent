from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

"""
main.py
=======

Interactive CLI agent for Bangladesh institutional / hospital / restaurant
data, powered by Google Gemini (via LangChain) and backed by four tools:

    1. InstitutionsDBTool  -> institutions.db  (universities, colleges, govt institutes)
    2. HospitalsDBTool     -> hospitals.db     (hospitals, beds, doctors, facilities)
    3. RestaurantsDBTool   -> restaurants.db   (restaurants, cuisine, ratings, locations)
    4. WebSearchTool       -> Tavily web search (general background/policy questions)

The agent is built with LangChain's current `create_agent` API (a thin
wrapper around a LangGraph tool-calling loop) and routes each question to
whichever tool(s) best answer it, based on the system prompt below.

Every question printed to the terminal is followed by a "[Routing Info]"
line showing exactly which tool(s) the agent invoked to answer it (or
"Direct LLM Knowledge" if it answered without calling any tool), so you can
see the routing decision at a glance before the final answer.

--------------------------------------------------------------------------
ENVIRONMENT VARIABLES (required before running this script)
--------------------------------------------------------------------------
    GEMINI_API_KEY   Your Google Gemini API key.
                      Get one at: https://aistudio.google.com/app/apikey
                      `langchain_google_genai` reads this automatically, so
                      you don't need to pass it in code.

    TAVILY_API_KEY   Your Tavily web search API key.
                      Get one at: https://app.tavily.com/
                      `TavilySearch` reads this automatically.

Set them in your shell before running (Linux/macOS):

    export GEMINI_API_KEY="your-gemini-api-key"
    export TAVILY_API_KEY="your-tavily-api-key"

Or on Windows (PowerShell):

    $env:GEMINI_API_KEY="your-gemini-api-key"
    $env:TAVILY_API_KEY="your-tavily-api-key"

Alternatively, create a `.env` file in this directory:

    GEMINI_API_KEY=your-gemini-api-key
    TAVILY_API_KEY=your-tavily-api-key

and this script will load it automatically if `python-dotenv` is installed
(`pip install python-dotenv`) — this is optional, not required.

--------------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------------
    pip install langchain langchain-core langchain-community \\
                langchain-google-genai langchain-tavily

    # db_tools.py (InstitutionsDBTool / HospitalsDBTool / RestaurantsDBTool)
    # must be in the same directory as this script, and institutions.db /
    # hospitals.db / restaurants.db must already exist (see build_databases.py).

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python main.py

Then type questions at the prompt. Type 'exit' or 'quit' to stop.
"""

import os
import sys

# Optional: silently load a .env file if python-dotenv is installed. This is
# a convenience only — the script works fine with plain shell env vars too.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from db_tools import InstitutionsDBTool, HospitalsDBTool, RestaurantsDBTool


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Default paths to the three SQLite databases (override via env vars if your
# databases live elsewhere).
INSTITUTIONS_DB_PATH = os.environ.get("INSTITUTIONS_DB_PATH", "institutions.db")
HOSPITALS_DB_PATH = os.environ.get("HOSPITALS_DB_PATH", "hospitals.db")
RESTAURANTS_DB_PATH = os.environ.get("RESTAURANTS_DB_PATH", "restaurants.db")

# Gemini model to use. Google retires older model names over time (e.g.
# "gemini-2.5-flash" was retired for new users in favor of
# "gemini-3.6-flash"), so if you hit a 404 NOT_FOUND error mentioning a
# model name, check https://ai.google.dev/gemini-api/docs/models for the
# current list and update GEMINI_MODEL below or via env var — no code
# changes needed elsewhere.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# Max number of Tavily results returned per web search call.
TAVILY_MAX_RESULTS = int(os.environ.get("TAVILY_MAX_RESULTS", "5"))

# Maps each tool's internal LangChain tool name (what the agent actually
# calls) to a friendly, human-readable label for the CLI routing indicator.
TOOL_DISPLAY_NAMES = {
    "institutions_db_tool": "InstitutionsDBTool (institutions.db)",
    "hospitals_db_tool": "HospitalsDBTool (hospitals.db)",
    "restaurants_db_tool": "RestaurantsDBTool (restaurants.db)",
    "tavily_search": "WebSearchTool (Tavily)",
}


SYSTEM_PROMPT = """You are a helpful research assistant for Bangladesh-related \
questions. You have access to four tools:

1. institutions_db_tool — queries a SQLite database of Bangladeshi \
educational and government institutions (universities, colleges, schools, \
government institutes): names, types, locations/districts, and capacities.

2. hospitals_db_tool — queries a SQLite database of Bangladeshi hospitals: \
names, locations, number of beds, doctors, facilities/services, and hospital \
type (public/private/specialized).

3. restaurants_db_tool — queries a SQLite database of Bangladeshi \
restaurants: names, cuisine types, locations, and ratings.

4. tavily_search — general-purpose web search.

ROUTING RULES — read carefully before answering:

- If the question asks for specific data, statistics, counts, ratings, \
capacities, or entity lookups that would live in one of the three \
databases (e.g. "How many hospitals are in Dhaka?", "Which restaurant in \
Gulshan has the highest rating?", "List government colleges in Khulna", \
"How many beds does Square Hospital have?"), use the matching database \
tool:
    - Institutions (universities/colleges/schools/govt institutes) -> \
institutions_db_tool
    - Hospitals (hospitals/beds/doctors/facilities) -> hospitals_db_tool
    - Restaurants (restaurants/cuisine/ratings/locations) -> \
restaurants_db_tool

- If the question asks for general background information, definitions, \
policy, history, news, or anything NOT contained in a structured local \
database (e.g. "What is the role of DGHS in Bangladesh?", "Explain \
Bangladesh's national health policy", "What are the admission requirements \
for public universities in Bangladesh?", "Latest news about a hospital \
scandal"), use tavily_search instead.

- If a question has both a factual/statistical part AND a background part \
(e.g. "How many public hospitals are in Dhaka, and what is DGHS's role in \
regulating them?"), use multiple tools — the relevant DB tool for the data \
part, and web search for the background part — then combine both into one \
coherent answer.

- Never guess or fabricate database facts (names, numbers, ratings) — \
always call the appropriate DB tool to get them. Never guess at current \
events, policy details, or definitions you're unsure about — use web \
search instead of relying on your own memory for anything time-sensitive \
or Bangladesh-policy-specific.

- Keep answers concise, direct, and in plain language. Don't mention tool \
names, SQL, or your internal reasoning process in the final answer to the \
user — just answer the question.
"""


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

def _require_env_var(name: str, signup_url: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(
            f"ERROR: Environment variable {name} is not set.\n"
            f"Get an API key at {signup_url} and set it, e.g.:\n"
            f'    export {name}="your-api-key"\n',
            file=sys.stderr,
        )
        sys.exit(1)
    return value


def build_agent():
    """Construct and return the compiled LangChain agent with all four tools.

    Note on "intermediate steps": this agent is built with LangChain's
    current `create_agent` (a LangGraph-based tool-calling loop), not the
    legacy `AgentExecutor`. `create_agent` has no `return_intermediate_steps`
    flag because it doesn't need one — the full tool-call trace (which
    tools were called, with what arguments, and what they returned) is
    already present in the `messages` list of every `agent.invoke(...)`
    result. `_extract_tool_names()` below reads that trace directly, which
    gives the same routing visibility `return_intermediate_steps=True`
    would on an `AgentExecutor`.
    """

    _require_env_var("GEMINI_API_KEY", "https://aistudio.google.com/app/apikey")
    _require_env_var("TAVILY_API_KEY", "https://app.tavily.com/")

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
    )

    institutions_tool = InstitutionsDBTool(llm=llm, db_path=INSTITUTIONS_DB_PATH)
    hospitals_tool = HospitalsDBTool(llm=llm, db_path=HOSPITALS_DB_PATH)
    restaurants_tool = RestaurantsDBTool(llm=llm, db_path=RESTAURANTS_DB_PATH)
    web_search_tool = TavilySearch(max_results=TAVILY_MAX_RESULTS)

    tools = [institutions_tool, hospitals_tool, restaurants_tool, web_search_tool]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


def _extract_final_answer(agent_result: dict) -> str:
    """Pull the last assistant message's text content out of the agent result."""
    messages = agent_result.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", None)
        msg_type = getattr(message, "type", None)
        if content and msg_type == "ai":
            if isinstance(content, str):
                return content
            # Some providers return content as a list of content blocks.
            text_parts = [
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if text_parts:
                return "".join(text_parts)
    return "(No answer was generated.)"


def _extract_tool_names(agent_result: dict) -> list[str]:
    """
    Scan the agent's full message trace and return the internal tool
    name(s) invoked while answering the question, in call order (each name
    appears once even if the same tool was called more than once).

    This is the `create_agent` equivalent of reading
    `response["intermediate_steps"]` off a legacy `AgentExecutor` result —
    the tool-call metadata lives in `AIMessage.tool_calls` entries within
    `agent_result["messages"]` instead of a separate key.
    """
    tool_names: list[str] = []
    for message in agent_result.get("messages", []):
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name and name not in tool_names:
                tool_names.append(name)
    return tool_names


def _format_routing_label(tool_names: list[str]) -> str:
    """Turn internal tool name(s) into the friendly CLI routing label."""
    if not tool_names:
        return "Direct LLM Knowledge"
    labels = [TOOL_DISPLAY_NAMES.get(name, name) for name in tool_names]
    return ", ".join(labels)


# --------------------------------------------------------------------------- #
# Interactive CLI loop
# --------------------------------------------------------------------------- #

def main() -> None:
    print("=" * 70)
    print("Bangladesh Multi-Tool AI Agent")
    print("(Institutions | Hospitals | Restaurants | Web Search)")
    print("=" * 70)

    missing_dbs = [
        path for path in (INSTITUTIONS_DB_PATH, HOSPITALS_DB_PATH, RESTAURANTS_DB_PATH)
        if not os.path.exists(path)
    ]
    if missing_dbs:
        print(
            f"WARNING: the following database file(s) were not found: "
            f"{', '.join(missing_dbs)}\n"
            f"Run build_databases.py first, or set INSTITUTIONS_DB_PATH / "
            f"HOSPITALS_DB_PATH / RESTAURANTS_DB_PATH to point at existing "
            f"databases. Queries against a missing database will fail.\n"
        )

    print("Building agent...")
    agent = build_agent()
    print("Ready. Type your question below (type 'exit' or 'quit' to stop).\n")

    while True:
        print("=" * 50)
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            # Nothing was typed — don't burn a routing block on it.
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
            tool_names = _extract_tool_names(result)
            answer = _extract_final_answer(result)
        except Exception as exc:  # noqa: BLE001 - keep the CLI alive on any error
            tool_names = []
            answer = f"(An error occurred while answering: {exc})"

        routing_label = _format_routing_label(tool_names)
        print(f"[Routing Info] Tool Selected: {routing_label}")
        print("-" * 50)
        print(f"Agent: {answer}")
        print("=" * 50)


if __name__ == "__main__":
    main()
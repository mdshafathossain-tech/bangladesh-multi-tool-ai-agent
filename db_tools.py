"""
db_tools.py
===========

LangChain-compatible tools that let an agent answer natural-language
questions against the three SQLite databases produced by
``build_databases.py``:

    * institutions.db  -> InstitutionsDBTool  (universities, colleges, govt institutes)
    * hospitals.db     -> HospitalsDBTool     (hospitals, beds, doctors, facilities)
    * restaurants.db   -> RestaurantsDBTool   (restaurants, cuisine, ratings, locations)

Each tool is a ``langchain_core.tools.BaseTool`` subclass that, given a
natural-language question:

    1. Inspects its database's schema (table + column names/types).
    2. Asks the supplied chat LLM to translate the question into a single,
       read-only SQLite ``SELECT`` query, grounded in that schema.
    3. Validates the generated query is safe (SELECT-only — no INSERT,
       UPDATE, DELETE, DROP, ALTER, ATTACH, PRAGMA-write, etc.) before
       ever executing it.
    4. Executes the query against the SQLite file.
    5. Asks the LLM to turn the raw result rows into a short, clean,
       natural-language answer to the original question.

Because each tool carries its own name, docstring, and ``description``
(naming the entities and columns it covers), a LangChain router / agent
can look at the three tools and pick the right one for a given question
without any extra routing logic.

Requirements
------------
    pip install langchain langchain-core langchain-community

You also need a LangChain chat model to pass into each tool, e.g.::

    pip install langchain-anthropic     # or langchain-openai, etc.

Example
-------
    from langchain_anthropic import ChatAnthropic
    from db_tools import InstitutionsDBTool, HospitalsDBTool, RestaurantsDBTool

    llm = ChatAnthropic(model="claude-sonnet-4-6")

    tools = [
        InstitutionsDBTool(llm=llm, db_path="institutions.db"),
        HospitalsDBTool(llm=llm, db_path="hospitals.db"),
        RestaurantsDBTool(llm=llm, db_path="restaurants.db"),
    ]

    # Direct call:
    print(tools[0].invoke({"question": "How many universities are in Dhaka?"}))

    # Or hand `tools` to a LangChain/LangGraph agent, e.g.:
    from langchain.agents import create_agent
    agent = create_agent(llm, tools)
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, ClassVar, Optional, Type

from pydantic import BaseModel, Field

try:
    from langchain_core.tools import BaseTool
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "db_tools.py requires LangChain. Install it with:\n"
        "    pip install langchain langchain-core langchain-community"
    ) from exc


# --------------------------------------------------------------------------- #
# Safety: only allow read-only SELECT queries to be executed.
# --------------------------------------------------------------------------- #

_FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE",
    "TRUNCATE", "ATTACH", "DETACH", "VACUUM", "PRAGMA", "REINDEX",
)


class SQLSafetyError(ValueError):
    """Raised when the LLM-generated SQL is not a safe, read-only query."""


def _message_content_to_text(response: Any) -> str:
    """
    Normalize an LLM response's `.content` into a plain string.

    Most chat models return `.content` as a plain string, but some
    (including newer Gemini models, depending on response mode) return it
    as a list of content blocks instead, e.g.
    ``[{"type": "text", "text": "..."}]``. Calling `.strip()` directly on
    that list raises `AttributeError: 'list' object has no attribute
    'strip'` — this helper handles both shapes safely.
    """
    content = getattr(response, "content", None)
    if content is None:
        return str(response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Common shape: {"type": "text", "text": "..."}
                text = block.get("text")
                if text:
                    parts.append(text)
        if parts:
            return "".join(parts)
        return str(content)
    return str(content)


def _extract_sql(raw_text: str) -> str:
    """Pull a bare SQL statement out of an LLM response.

    Handles plain SQL, SQL fenced in ```sql ... ``` / ``` ... ``` blocks,
    and strips a trailing semicolon / stray commentary.
    """
    text = raw_text.strip()

    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # If the model added prose before/after, try to isolate the first
    # SELECT ... statement.
    select_match = re.search(r"(SELECT\b.*)", text, re.DOTALL | re.IGNORECASE)
    if select_match:
        text = select_match.group(1).strip()

    # Cut at the first semicolon (only one statement should ever run).
    if ";" in text:
        text = text.split(";")[0].strip()

    return text.strip()


def _validate_readonly_sql(sql: str) -> None:
    """Raise SQLSafetyError unless `sql` is a single, read-only SELECT."""
    if not sql:
        raise SQLSafetyError("The model did not return a SQL query.")

    stripped = sql.strip().rstrip(";").strip()
    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        raise SQLSafetyError(
            f"Generated query is not a SELECT statement and was blocked:\n{sql}"
        )

    upper = stripped.upper()
    for keyword in _FORBIDDEN_KEYWORDS:
        # Word-boundary match so e.g. "UPDATED_AT" column names aren't flagged.
        if re.search(rf"\b{keyword}\b", upper):
            raise SQLSafetyError(
                f"Generated query contains a forbidden keyword '{keyword}' "
                f"and was blocked:\n{sql}"
            )

    if ";" in stripped:
        raise SQLSafetyError(
            f"Generated query contains multiple statements and was blocked:\n{sql}"
        )


# --------------------------------------------------------------------------- #
# Schema introspection
# --------------------------------------------------------------------------- #

def _get_schema_description(db_path: str) -> str:
    """Return a compact text description of every table's columns/types."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        lines = []
        for table in tables:
            cursor.execute(f'PRAGMA table_info("{table}");')
            cols = cursor.fetchall()  # (cid, name, type, notnull, dflt_value, pk)
            col_desc = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
            lines.append(f"Table \"{table}\": {col_desc}")
        return "\n".join(lines) if lines else "(no tables found)"
    finally:
        conn.close()


def _get_sample_rows(db_path: str, table: str, limit: int = 3) -> str:
    """Return a few sample rows from `table` to help the LLM ground values."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM "{table}" LIMIT {limit};')
        col_names = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        if not rows:
            return "(table is empty)"
        formatted = [", ".join(col_names)]
        for row in rows:
            formatted.append(", ".join(str(v) for v in row))
        return "\n".join(formatted)
    finally:
        conn.close()


def _get_first_table_name(db_path: str) -> Optional[str]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Query execution
# --------------------------------------------------------------------------- #

def _run_sql(db_path: str, sql: str, row_limit: int = 200) -> tuple[list[str], list[tuple]]:
    """Execute a validated read-only query and return (columns, rows)."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA query_only = ON;")  # belt-and-suspenders: block writes at the driver level
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        col_names = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(row_limit)
        return col_names, rows
    finally:
        conn.close()


def _format_rows_for_prompt(col_names: list[str], rows: list[tuple], max_rows: int = 50) -> str:
    if not rows:
        return "(query returned no rows)"
    lines = [", ".join(col_names)]
    for row in rows[:max_rows]:
        lines.append(", ".join("" if v is None else str(v) for v in row))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows not shown)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Shared prompt templates
# --------------------------------------------------------------------------- #

_SQL_GEN_SYSTEM_PROMPT = """You are an expert SQLite query writer.

Given a database schema and a user's natural-language question, write ONE
single read-only SQLite SELECT query that answers the question as precisely
as possible.

Rules:
- Output ONLY the SQL query. No explanation, no markdown fences, no commentary.
- Only ever write a SELECT statement. Never write INSERT, UPDATE, DELETE,
  DROP, ALTER, CREATE, PRAGMA, ATTACH, or any other statement.
- Only reference tables and columns that actually appear in the schema below.
- Use SQLite functions/syntax (e.g. LIKE for fuzzy text matching, LOWER() for
  case-insensitive comparisons).
- If the question asks for a count, use COUNT(*).
- If the question implies "top N" or "highest/lowest", use ORDER BY and LIMIT.
- If a text filter is implied (e.g. a city or category name), match it
  case-insensitively with LIKE '%value%' unless an exact match is clearly
  intended.
- Never invent column or table names that are not in the schema.
"""

_ANSWER_GEN_SYSTEM_PROMPT = """You are a helpful assistant that explains SQL
query results in plain, natural language.

Given the user's original question, the SQL query that was run, and the
resulting rows, write a short, clear, direct answer to the question.

Rules:
- Answer in 1-4 sentences unless the data genuinely requires a short list.
- Only use the data provided in the query result — do not add outside facts.
- If the result is empty, say plainly that no matching records were found.
- Do not mention SQL, tables, or column names in your answer unless the user
  explicitly asked about the data's structure — just answer the question.
"""


# --------------------------------------------------------------------------- #
# Base class shared by all three tools
# --------------------------------------------------------------------------- #

class _DBToolInput(BaseModel):
    """Input schema shared by all database query tools."""

    question: str = Field(
        description="A natural-language question to answer using the database."
    )


class _BaseSQLiteQueryTool(BaseTool):
    """
    Shared natural-language-to-SQL-to-natural-language pipeline.

    Subclasses only need to set `name`, `description`, and pass a `db_path`
    and `llm` at construction time — all query generation, safety checking,
    execution, and answer summarization logic lives here.
    """

    args_schema: ClassVar[Type[BaseModel]] = _DBToolInput

    # Populated per-instance in __init__ (declared here for type checkers).
    llm: Any = None
    db_path: str = ""
    row_limit: int = 200
    return_sql: bool = False
    skip_llm_summary: bool = False

    def __init__(
        self,
        llm: BaseChatModel,
        db_path: str,
        row_limit: int = 200,
        return_sql: bool = False,
        skip_llm_summary: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        llm:
            Any LangChain chat model (e.g. ChatAnthropic, ChatOpenAI) used
            both to generate SQL from the question and to summarize results.
        db_path:
            Path to the SQLite ``.db`` file this tool queries.
        row_limit:
            Maximum number of rows fetched from the database for any single
            query (protects against accidentally huge result sets).
        return_sql:
            If True, prefix the natural-language answer with the SQL query
            that was executed (useful for debugging/transparency).
        skip_llm_summary:
            If True, skip the second LLM call that turns raw SQL rows into
            natural-language prose, and instead return a plain formatted
            dump of the query result. This HALVES the number of LLM calls
            this tool makes per invocation (1 instead of 2) — useful when
            running under a tight request-per-day quota (e.g. a Gemini free
            tier). Safe to enable when this tool is used inside an agent
            (like main.py's create_agent), because the agent's own
            follow-up turn already turns the tool's output into a natural-
            language answer for the user — the tool's own summarization
            step is then redundant. Leave False (default) if you call this
            tool standalone and want it to always return finished prose.
        """
        super().__init__(llm=llm, db_path=db_path, row_limit=row_limit,
                          return_sql=return_sql, skip_llm_summary=skip_llm_summary,
                          **kwargs)

    # -- core pipeline ----------------------------------------------------- #

    def _generate_sql(self, question: str) -> str:
        schema = _get_schema_description(self.db_path)
        table = _get_first_table_name(self.db_path)
        sample = _get_sample_rows(self.db_path, table) if table else "(no tables)"

        user_prompt = (
            f"Database schema:\n{schema}\n\n"
            f"Sample rows from \"{table}\":\n{sample}\n\n"
            f"Question: {question}\n\n"
            f"SQL query:"
        )
        response = self.llm.invoke(
            [
                SystemMessage(content=_SQL_GEN_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        raw_sql = _message_content_to_text(response)
        return _extract_sql(raw_sql)

    def _summarize_results(self, question: str, sql: str, col_names: list[str],
                            rows: list[tuple]) -> str:
        result_text = _format_rows_for_prompt(col_names, rows)
        user_prompt = (
            f"Question: {question}\n\n"
            f"SQL query executed:\n{sql}\n\n"
            f"Query result:\n{result_text}\n\n"
            f"Natural-language answer:"
        )
        response = self.llm.invoke(
            [
                SystemMessage(content=_ANSWER_GEN_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        answer = _message_content_to_text(response)
        return answer.strip()

    def _run(self, question: str, **kwargs: Any) -> str:
        try:
            sql = self._generate_sql(question)
            _validate_readonly_sql(sql)
        except SQLSafetyError as exc:
            return f"I couldn't safely answer that question: {exc}"
        except Exception as exc:  # noqa: BLE001 - surface any LLM/parsing error clearly
            return f"I couldn't turn that question into a SQL query: {exc}"

        try:
            col_names, rows = _run_sql(self.db_path, sql, row_limit=self.row_limit)
        except sqlite3.Error as exc:
            return (
                f"I generated a query but it failed to run against the "
                f"database: {exc}\nQuery: {sql}"
            )

        try:
            if self.skip_llm_summary:
                # No second LLM call — return a plain, readable dump of the
                # result. The calling agent's own next turn will phrase this
                # into natural language for the user.
                answer = (
                    f"Query result for \"{question}\":\n"
                    f"{_format_rows_for_prompt(col_names, rows)}"
                )
            else:
                answer = self._summarize_results(question, sql, col_names, rows)
        except Exception as exc:  # noqa: BLE001
            # Fall back to a raw dump of results if summarization fails.
            answer = (
                f"Here is the raw query result (summarization failed: {exc}):\n"
                f"{_format_rows_for_prompt(col_names, rows)}"
            )

        if self.return_sql:
            return f"{answer}\n\n[SQL used: {sql}]"
        return answer

    async def _arun(self, question: str, **kwargs: Any) -> str:
        # Simple synchronous fallback; override with true async LLM/DB calls
        # if you need non-blocking behavior in an async agent.
        return self._run(question, **kwargs)


# --------------------------------------------------------------------------- #
# The three concrete tools
# --------------------------------------------------------------------------- #

class InstitutionsDBTool(_BaseSQLiteQueryTool):
    """
    Query tool for `institutions.db`.

    Use this tool to answer questions about Bangladeshi educational and
    government institutions: universities, colleges, schools, government
    institutes, their names, locations/districts, types, and capacities.

    Examples of questions this tool can handle:
        - "How many universities are located in Dhaka?"
        - "List all government institutes in Chittagong."
        - "Which college has the largest capacity?"
    """

    name: str = "institutions_db_tool"
    description: str = (
        "Answers natural-language questions about Bangladeshi educational and "
        "government institutions (universities, colleges, schools, government "
        "institutes) by querying the institutions SQLite database. Use this "
        "tool for questions involving institution names, types (university/"
        "college/school/government institute), locations or districts, and "
        "capacity. Input should be a plain-English question; do not write SQL "
        "yourself."
    )


class HospitalsDBTool(_BaseSQLiteQueryTool):
    """
    Query tool for `hospitals.db`.

    Use this tool to answer questions about Bangladeshi hospitals: hospital
    names, locations, number of beds, doctors, facilities/services offered,
    and hospital type (public/private/specialized).

    Examples of questions this tool can handle:
        - "Which hospitals in Sylhet have more than 100 beds?"
        - "List hospitals that offer cardiology services."
        - "How many private hospitals are there in Dhaka?"
    """

    name: str = "hospitals_db_tool"
    description: str = (
        "Answers natural-language questions about Bangladeshi hospitals by "
        "querying the hospitals SQLite database. Use this tool for questions "
        "involving hospital names, locations/districts, number of beds, "
        "doctors, medical facilities or services, and hospital type (public/"
        "private/specialized). Input should be a plain-English question; do "
        "not write SQL yourself."
    )


class RestaurantsDBTool(_BaseSQLiteQueryTool):
    """
    Query tool for `restaurants.db`.

    Use this tool to answer questions about Bangladeshi restaurants:
    restaurant names, cuisine types, locations, and ratings.

    Examples of questions this tool can handle:
        - "What are the top-rated Chinese restaurants in Dhaka?"
        - "List restaurants in Gulshan with a rating above 4."
        - "Which restaurant has the most reviews?"
    """

    name: str = "restaurants_db_tool"
    description: str = (
        "Answers natural-language questions about Bangladeshi restaurants by "
        "querying the restaurants SQLite database. Use this tool for "
        "questions involving restaurant names, cuisine types, locations/"
        "areas, and ratings. Input should be a plain-English question; do "
        "not write SQL yourself."
    )


# --------------------------------------------------------------------------- #
# Manual smoke test / usage example
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import os
    import sys

    print(__doc__)

    # This block only runs a real query if the three .db files exist next to
    # this script AND a LangChain chat model can be constructed from your
    # environment (e.g. ANTHROPIC_API_KEY / OPENAI_API_KEY is set). It's meant
    # as a quick manual check, not an automated test.
    db_files = ["institutions.db", "hospitals.db", "restaurants.db"]
    missing = [f for f in db_files if not os.path.exists(f)]
    if missing:
        print(f"\n(Skipping live demo — missing database files: {missing}. "
              f"Run build_databases.py first.)")
        sys.exit(0)

    llm = None
    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-sonnet-4-6")
    except Exception:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini")
        except Exception:
            pass

    if llm is None:
        print("\n(Skipping live demo — no LangChain chat model available. "
              "Install langchain-anthropic or langchain-openai and set an "
              "API key to try this out.)")
        sys.exit(0)

    tools = [
        InstitutionsDBTool(llm=llm, db_path="institutions.db"),
        HospitalsDBTool(llm=llm, db_path="hospitals.db"),
        RestaurantsDBTool(llm=llm, db_path="restaurants.db"),
    ]
    for tool in tools:
        print(f"\n--- {tool.name} ---")
        print(tool.description)

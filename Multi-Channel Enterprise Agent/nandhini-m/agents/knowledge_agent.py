import os
import re
import pandas as pd

from utils.bedrock_llm import call_llm
from utils.cloudwatch_logger import log_agent_call


# =====================================================
# DATA PATHS
# =====================================================

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data"
)

PRODUCT_FILE = os.path.join(
    DATA_DIR,
    "indian_pharmaceutical_products_clean.csv"
)

INCENTIVE_FILE = os.path.join(
    DATA_DIR,
    "sales_incentive.csv"
)

# =====================================================
# LOAD PRODUCT DATA
# =====================================================

try:

    PRODUCTS_DATA = pd.read_csv(
        PRODUCT_FILE,
        low_memory=False
    )

    PRODUCTS_DATA.columns = (

        PRODUCTS_DATA.columns
        .str.strip()
        .str.lower()
    )

except Exception:

    PRODUCTS_DATA = pd.DataFrame()

# =====================================================
# LOAD INCENTIVE DATA
# =====================================================

try:

    INCENTIVES_DATA = pd.read_csv(
        INCENTIVE_FILE
    )

    INCENTIVES_DATA.columns = (

        INCENTIVES_DATA.columns
        .str.strip()
        .str.lower()
    )

    # CLEAN EMPLOYEE IDS

    if "employee_id" in INCENTIVES_DATA.columns:

        INCENTIVES_DATA["employee_id"] = (

            INCENTIVES_DATA["employee_id"]

            .astype(str)

            .str.upper()

            .str.strip()
        )

    # CLEAN NAMES

    if "name" in INCENTIVES_DATA.columns:

        INCENTIVES_DATA["name"] = (

            INCENTIVES_DATA["name"]

            .astype(str)

            .str.lower()

            .str.strip()
        )

except Exception:

    INCENTIVES_DATA = pd.DataFrame()

# =====================================================
# FIND EMPLOYEE INCENTIVE
# =====================================================

def find_incentive_employee(
    session_context,
    query
):

    if INCENTIVES_DATA.empty:

        return pd.DataFrame()

    df = INCENTIVES_DATA

    query_lower = query.lower()

    query_upper = query.upper()

    # =================================================
    # SEARCH EMPLOYEE ID
    # =================================================

    match = re.search(
        r"EMP\d+",
        query_upper
    )

    if match:

        employee_id = match.group()

        matched = df[

            df["employee_id"]

            == employee_id
        ]

        if not matched.empty:

            return matched

    # =================================================
    # SEARCH EMPLOYEE NAME
    # =================================================

    if "name" in df.columns:

        for employee_name in df["name"].unique():

            employee_name = str(
                employee_name
            ).lower().strip()

            # FULL NAME MATCH

            if employee_name in query_lower:

                matched = df[

                    df["name"]

                    == employee_name
                ]

                if not matched.empty:

                    return matched

            # FIRST NAME MATCH

            first_name = (
                employee_name.split()[0]
            )

            if first_name in query_lower:

                matched = df[

                    df["name"]

                    == employee_name
                ]

                if not matched.empty:

                    return matched

    # =================================================
    # SESSION EMPLOYEE
    # =================================================

    session_employee_id = str(

        session_context.get(
            "employee_id",
            ""
        )

    ).upper().strip()

    if session_employee_id:

        matched = df[

            df["employee_id"]

            == session_employee_id
        ]

        if not matched.empty:

            return matched

    return pd.DataFrame()

# =====================================================
# INCENTIVE SEARCH
# =====================================================

def get_incentive_context(
    session_context,
    query
):

    employee_data = find_incentive_employee(

        session_context,

        query
    )

    # =================================================
    # NO MATCH
    # =================================================

    if employee_data.empty:

        return (
            "No matching employee incentive "
            "records were found."
        )

    # =================================================
    # BUILD CONTEXT
    # =================================================

    results = []

    for _, row in employee_data.iterrows():

        details = []

        for column in [

            "employee_id",
            "name",
            "quarter",
            "region",
            "product_focus",
            "target_units",
            "achieved_units",
            "achievement_pct",
            "final_incentive_inr",
            "incentive_tier",
            "payout_status"
        ]:

            if (

                column in row.index
                and
                pd.notna(row[column])
            ):

                details.append(
                    f"{column}: {row[column]}"
                )

        results.append(
            "\n".join(details)
        )

    return "\n\n".join(results)

# =====================================================
# KNOWLEDGE AGENT
# =====================================================

def knowledge_agent(
    state: dict
):

    user_message = state["user_message"]

    session_id = state["session_id"]

    session_context = state.get(
        "session_context",
        {}
    )

    history_str = state.get(
        "chat_history_str",
        "No previous conversation."
    )

    channel = state.get(
        "channel",
        "web"
    )

    msg_lower = user_message.lower()

    # =================================================
    # INCENTIVE DETECTION
    # =================================================

    incentive_keywords = [

        "incentive",
        "commission",
        "payout",
        "achievement",
        "target",
        "quarter",
        "sales",
        "tier",
        "platinum",
        "gold",
        "silver"
    ]

    is_incentive_query = any(

        keyword in msg_lower

        for keyword in incentive_keywords
    )

    # =================================================
    # CONTEXT
    # =================================================

    if is_incentive_query:

        context = get_incentive_context(

            session_context,

            user_message
        )

    else:

        context = (
            "Product search enabled."
        )

    # =================================================
    # STYLE
    # =================================================

    if channel == "whatsapp":

        style_note = (
            "Give concise WhatsApp-style answers."
        )

    else:

        style_note = (
            "Give professional detailed answers."
        )

    # =================================================
    # SYSTEM
    # =================================================

    system = """
You are PharmAssist Knowledge Agent.

You answer:
- incentives
- sales targets
- quarterly performance
- pharma products
- medicine brands

Rules:
- Give direct business answers
- Use tables or bullets when useful
- Never invent employee data
- Never show random employee records
- If employee not found, clearly say so
"""

    # =================================================
    # PROMPT
    # =================================================

    prompt = f"""
Knowledge Data:
{context}

Conversation:
{history_str}

Employee Question:
{user_message}

Instruction:
{style_note}
"""

    # =================================================
    # CALL LLM
    # =================================================

    response = call_llm(

        prompt=prompt,

        system=system
    )

    # =================================================
    # LOGGING
    # =================================================

    try:

        log_agent_call(

            session_id=session_id,

            employee_id=session_context.get(
                "employee_id",
                "unknown"
            ),

            agent="KNOWLEDGE_AGENT",

            query=user_message,

            response_summary=response[:300]
        )

    except Exception:

        pass

    # =================================================
    # UPDATE STATE
    # =================================================

    state["agent_response"] = response

    state["agent_used"] = (
        "Knowledge Agent"
    )

    return state
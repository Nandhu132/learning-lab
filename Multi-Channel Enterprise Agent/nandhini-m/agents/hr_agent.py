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

HR_CSV = os.path.join(
    DATA_DIR,
    "hr_data.csv"
)

# =====================================================
# LOAD HR DATA
# =====================================================

try:

    HR_DATA = pd.read_csv(HR_CSV)

    # =================================================
    # CLEAN COLUMN NAMES
    # =================================================

    HR_DATA.columns = (

        HR_DATA.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # =================================================
    # CLEAN EMPLOYEE IDS
    # =================================================

    if "employee_id" in HR_DATA.columns:

        HR_DATA["employee_id"] = (

            HR_DATA["employee_id"]

            .astype(str)

            .str.upper()

            .str.strip()
        )

    # =================================================
    # CLEAN NAMES
    # =================================================

    if "name" in HR_DATA.columns:

        HR_DATA["name"] = (

            HR_DATA["name"]

            .astype(str)

            .str.strip()
        )

    print("\n✅ HR DATA LOADED")

    if "employee_id" in HR_DATA.columns:

        print(
            HR_DATA["employee_id"].tolist()
        )

except Exception as e:

    print(f"\n❌ HR DATA LOAD ERROR: {e}")

    HR_DATA = pd.DataFrame()

# =====================================================
# FIND EMPLOYEE
# =====================================================

def find_employee(
    session_context: dict,
    question: str
):

    if HR_DATA.empty:

        return None

    question_lower = question.lower()

    question_upper = question.upper()

    # =================================================
    # 1. SEARCH EMPLOYEE ID
    # =================================================

    match = re.search(
        r"EMP\d+",
        question_upper
    )

    if match:

        employee_id = match.group()

        matched = HR_DATA[

            HR_DATA["employee_id"]

            == employee_id
        ]

        if not matched.empty:

            return matched.iloc[0]

    # =================================================
    # 2. SEARCH EMPLOYEE NAME
    # =================================================

    if "name" in HR_DATA.columns:

        for _, row in HR_DATA.iterrows():

            employee_name = str(

                row["name"]

            ).lower().strip()

            # -----------------------------------------
            # FULL NAME MATCH
            # -----------------------------------------

            if employee_name in question_lower:

                return row

            # -----------------------------------------
            # FIRST NAME MATCH
            # -----------------------------------------

            first_name = (
                employee_name.split()[0]
            )

            if first_name in question_lower:

                return row

    # =================================================
    # 3. SESSION EMPLOYEE FALLBACK
    # =================================================

    session_employee_id = str(

        session_context.get(
            "employee_id",
            ""
        )

    ).upper().strip()

    if session_employee_id:

        matched = HR_DATA[

            HR_DATA["employee_id"]

            == session_employee_id
        ]

        if not matched.empty:

            return matched.iloc[0]

    return None

# =====================================================
# BUILD EMPLOYEE CONTEXT
# =====================================================

def build_employee_context(
    employee
):

    important_fields = [

        "employee_id",
        "name",
        "department",
        "designation",
        "manager_name",
        "location",
        "base_salary_annual",
        "hra_monthly",
        "ta_monthly",
        "mobile_allowance_monthly",
        "medical_allowance_annual",
        "leave_balance_casual",
        "leave_balance_sick",
        "leave_balance_earned",
        "date_of_joining",
        "pf_enrolled",
        "bank_name",
        "account_number"
    ]

    lines = []

    for field in important_fields:

        if (

            field in employee.index
            and
            pd.notna(employee[field])
        ):

            lines.append(
                f"{field}: {employee[field]}"
            )

    return "\n".join(lines)

# =====================================================
# HR AGENT
# =====================================================

def hr_agent(
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
        ""
    )

    channel = state.get(
        "channel",
        "web"
    )

    # =================================================
    # FIND EMPLOYEE
    # =================================================

    employee = find_employee(

        session_context,

        user_message
    )

    # =================================================
    # EMPLOYEE NOT FOUND
    # =================================================

    if employee is None:

        state["agent_response"] = (
            "Employee information was not found."
        )

        state["agent_used"] = "HR Agent"

        return state

    # =================================================
    # EMPLOYEE CONTEXT
    # =================================================

    employee_context = build_employee_context(
        employee
    )

    # =================================================
    # SAVE EMPLOYEE NAME
    # =================================================

    if (

        "name" in employee.index
        and
        pd.notna(employee["name"])
    ):

        session_context[
            "employee_name"
        ] = str(employee["name"])

    # =================================================
    # SAVE EMPLOYEE ID
    # =================================================

    if (

        "employee_id" in employee.index
        and
        pd.notna(employee["employee_id"])
    ):

        session_context[
            "employee_id"
        ] = str(employee["employee_id"])

    # =================================================
    # RESPONSE STYLE
    # =================================================

    if channel == "whatsapp":

        style_note = """
Give short WhatsApp-style answers.
Use short bullet points.
Keep answers concise.
"""

    else:

        style_note = """
Give professional HR responses.
Use bullet points when useful.
Keep responses concise and business-friendly.
Mention important values clearly.
"""

    # =================================================
    # SYSTEM PROMPT
    # =================================================

    system = f"""
You are PharmAssist HR Agent for PharmD India.

You answer:
- salary
- payroll
- PF
- leave balances
- reimbursements
- employee information
- manager details
- benefits

Rules:
- Give direct HR answers
- Use professional formatting
- Keep responses concise
- Never mention AI limitations
- Never say:
  - "based on provided context"
  - "I cannot hallucinate"
  - "I am an AI"

{style_note}
"""

    # =================================================
    # FINAL PROMPT
    # =================================================

    prompt = f"""
Employee Information:
{employee_context}

Conversation History:
{history_str}

Employee Question:
{user_message}
"""

    # =================================================
    # CALL LLM
    # =================================================

    response = call_llm(

        prompt=prompt,

        system=system
    )

    # =================================================
    # CLOUDWATCH LOG
    # =================================================

    try:

        log_agent_call(

            session_id=session_id,

            employee_id=str(

                session_context.get(
                    "employee_id",
                    "unknown"
                )
            ),

            agent="HR_AGENT",

            query=user_message,

            response_summary=response[:300]
        )

    except Exception:

        pass

    # =================================================
    # UPDATE STATE
    # =================================================

    state["agent_response"] = response

    state["agent_used"] = "HR Agent"

    state["session_context"] = session_context

    return state
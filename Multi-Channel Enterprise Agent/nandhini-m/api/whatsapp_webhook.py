import logging
import pandas as pd
import os

from fastapi import FastAPI
from fastapi import Form
from fastapi.responses import Response

from twilio.twiml.messaging_response import (
    MessagingResponse
)

from main import run_agent

from memory.session_store import (
    get_session,
    update_session
)


# =====================================================
# CLEAN LOGS
# =====================================================

logging.getLogger().setLevel(
    logging.ERROR
)


# =====================================================
# LOAD HR DATA ONCE AT STARTUP
# =====================================================

HR_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "hr_data.csv"
)

try:
    _hr_df = pd.read_csv(HR_DATA_PATH)
    # Normalize column names to lowercase
    _hr_df.columns = [
        c.strip().lower()
        for c in _hr_df.columns
    ]
    print(f"✅ HR data loaded: {len(_hr_df)} employees")
except Exception as e:
    _hr_df = None
    print(f"⚠️ HR data load failed: {e}")


# =====================================================
# EMPLOYEE LOOKUP HELPER
# =====================================================

def get_employee_by_id(employee_id: str):

    """
    Look up employee from hr_data.csv.
    Returns dict if found, None if not found.
    """

    if _hr_df is None:
        return None

    try:

        # Try common column name variations
        id_col = None
        for col in _hr_df.columns:
            if "employee" in col and "id" in col:
                id_col = col
                break
            elif col in ("emp_id", "empid", "id"):
                id_col = col
                break

        if id_col is None:
            # Use first column as fallback
            id_col = _hr_df.columns[0]

        match = _hr_df[
            _hr_df[id_col].astype(str).str.upper().str.strip()
            == employee_id.upper().strip()
        ]

        if not match.empty:
            return match.iloc[0].to_dict()

        return None

    except Exception as e:
        print(f"⚠️ Employee lookup error: {e}")
        return None


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Pharma WhatsApp Agent"
)


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")

def health():

    return {
        "status": "healthy"
    }


# =====================================================
# WHATSAPP WEBHOOK
# =====================================================

@app.post("/whatsapp")

async def whatsapp_webhook(
    Body: str = Form(...),
    From: str = Form(...)
):

    try:

        # =============================================
        # USER DETAILS
        # =============================================

        phone_number = (
            From
            .replace("whatsapp:", "")
            .strip()
        )

        session_id = (
            phone_number
            .replace("+", "")
        )

        message = Body.strip()

        # =============================================
        # LOAD SESSION
        # =============================================

        session_context = get_session(session_id)

        employee_id = session_context.get(
            "employee_id", ""
        )

        # =============================================
        # FIRST TIME USER — NO EMPLOYEE ID YET
        # =============================================

        if not employee_id:

            typed = message.upper().strip()

            # -----------------------------------------
            # CHECK IF THEY TYPED AN EMPLOYEE ID FORMAT
            # -----------------------------------------

            if typed.startswith("EMP") and len(typed) >= 4:

                # ✅ CHECK HR DATA FIRST before saving
                employee = get_employee_by_id(typed)

                if employee:

                    # ---------------------------------
                    # FOUND IN HR DATA — Save session
                    # ---------------------------------

                    # Try to get name from common column names
                    name = ""
                    for col in ("name", "employee_name", "emp_name", "full_name"):
                        if col in employee and employee[col]:
                            name = str(employee[col]).strip()
                            break

                    session_context["employee_id"] = typed
                    session_context["channel"] = "whatsapp"

                    update_session(session_id, session_context)

                    if name:
                        greeting = f"✅ Welcome, *{name}*!"
                    else:
                        greeting = f"✅ Welcome! Registered as *{typed}*."

                    answer = (
                        f"{greeting}\n\n"
                        "You can now ask me anything about:\n"
                        "• Your salary & leave balance\n"
                        "• Company policies\n"
                        "• Pharma products & incentives\n\n"
                        "How can I help you today?"
                    )

                else:

                    # ---------------------------------
                    # NOT FOUND — Do NOT save session
                    # ---------------------------------

                    answer = (
                        f"❌ *{typed}* was not found in our system.\n\n"
                        "Please check your Employee ID and try again.\n\n"
                        "Example: *EMP001*\n\n"
                        "If you need help, contact HR. 🙏"
                    )

            else:

                # -----------------------------------------
                # NO EMPLOYEE ID TYPED — Ask them
                # -----------------------------------------

                answer = (
                    "👋 Welcome to *PharmD Assistant*!\n\n"
                    "Please reply with your *Employee ID* to get started.\n\n"
                    "Example: *EMP001*"
                )

            twilio_response = MessagingResponse()
            twilio_response.message(answer)

            return Response(
                content=str(twilio_response),
                media_type="application/xml"
            )

        # =============================================
        # REGISTERED USER — RUN AI AGENT
        # =============================================

        result = run_agent(

            session_id=session_id,

            employee_id=employee_id,

            user_message=message,

            channel="whatsapp"
        ) 

        # =============================================
        # RESPONSE
        # =============================================

        if result["success"]:

            answer = result["data"]["answer"]

        else:

            answer = (
                "System temporarily unavailable. "
                "Please try again shortly."
            )

    except Exception as e:

        logging.error(f"WhatsApp webhook error: {e}")

        answer = (
            "Something went wrong. "
            "Please try again."
        )

    # =============================================
    # TWILIO RESPONSE
    # =============================================

    twilio_response = MessagingResponse()

    twilio_response.message(answer)

    return Response(
        content=str(twilio_response),
        media_type="application/xml"
    )

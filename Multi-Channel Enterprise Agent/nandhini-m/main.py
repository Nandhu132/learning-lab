import json
import uuid
import traceback

from agents.orchestrator import AGENT_GRAPH

from memory.session_store import (
    get_session,
    update_session,
    add_to_history,
    format_history_str
)


# =====================================================
# RUN AGENT
# =====================================================

def run_agent(
    session_id: str,
    user_message: str,
    channel: str = "web",
    employee_id: str = ""
):

    try:

        # ---------------------------------------------
        # LOAD SESSION
        # ---------------------------------------------

        session_context = get_session(
            session_id
        )

        session_context["channel"] = channel

        # ---------------------------------------------
        # SAVE EMPLOYEE ID
        # ---------------------------------------------

        if employee_id:

            session_context[
                "employee_id"
            ] = (

                employee_id
                .upper()
                .strip()
            )

        # ---------------------------------------------
        # LOAD HISTORY
        # ---------------------------------------------

        history_str = format_history_str(
            session_id
        )

        # ---------------------------------------------
        # BUILD STATE
        # ---------------------------------------------

        state = {

            "session_id":
            session_id,

            "user_message":
            user_message,

            "intent":
            "",

            "agent_response":
            "",

            "agent_used":
            "",

            "chat_history_str":
            history_str,

            "session_context":
            session_context,

            "channel":
            channel
        }

        # ---------------------------------------------
        # RUN LANGGRAPH
        # ---------------------------------------------

        result = AGENT_GRAPH.invoke(
            state
        )

        response = result.get(
            "agent_response",
            "No response generated."
        )

        intent = result.get(
            "intent",
            "unknown"
        )

        agent_used = result.get(
            "agent_used",
            "unknown"
        )

        # ---------------------------------------------
        # SAVE USER MESSAGE
        # ---------------------------------------------

        add_to_history(

            session_id,

            "user",

            user_message
        )

        # ---------------------------------------------
        # SAVE ASSISTANT RESPONSE
        # ---------------------------------------------

        add_to_history(

            session_id,

            "assistant",

            response,

            agent_used
        )

        # ---------------------------------------------
        # UPDATE SESSION
        # ---------------------------------------------

        update_session(

            session_id,

            result.get(
                "session_context",
                session_context
            )
        )

        # ---------------------------------------------
        # FINAL JSON RESPONSE
        # ---------------------------------------------

        return {

            "success": True,

            "data": {

                "session_id":
                session_id,

                "employee_id":
                session_context.get(
                    "employee_id",
                    ""
                ),

                "channel":
                channel,

                "intent":
                intent,

                "agent":
                agent_used,

                "question":
                user_message,

                "answer":
                response
            }
        }

    except Exception as e:

        traceback.print_exc()

        return {

            "success": False,

            "error": {

                "message":
                str(e),

                "type":
                "SYSTEM_ERROR"
            }
        }


# =====================================================
# LOCAL CLI TEST
# =====================================================

if __name__ == "__main__":

    session_id = (
        f"session_{uuid.uuid4().hex[:6]}"
    )

    print("\n💊 Pharma Enterprise Agent Started")

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in [

            "exit",
            "quit"
        ]:

            print("\nSession Ended")

            break

        result = run_agent(

            session_id=session_id,

            user_message=user_input,

            employee_id="EMP001",

            channel="web"
        )

        print(
            json.dumps(
                result,
                indent=4
            )
        )
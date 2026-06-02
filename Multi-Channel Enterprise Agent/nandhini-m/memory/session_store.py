import boto3
import json
import os
import time
import uuid

from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# ENV CONFIG
# =========================================================

AWS_REGION = os.getenv(
    "AWS_REGION",
    "us-east-1"
)

SESSION_TABLE = os.getenv(
    "DYNAMODB_SESSION_TABLE",
    "nandhini_pharma_sessions"
)

HISTORY_TABLE = os.getenv(
    "DYNAMODB_HISTORY_TABLE",
    "nandhini_pharma_history"
)

# =========================================================
# LOCAL MEMORY FALLBACK
# =========================================================

_local_sessions = {}

_local_history = {}

# =========================================================
# DYNAMODB CHECK
# =========================================================

_DYNAMO_OK = False


def _check_dynamo():

    global _DYNAMO_OK

    try:

        client = boto3.client(
            "dynamodb",
            region_name=AWS_REGION
        )

        client.list_tables()

        _DYNAMO_OK = True

        print(
            "✅ DynamoDB Connected"
        )

    except Exception as e:

        _DYNAMO_OK = False

        print(
            f"""
            ⚠️ DynamoDB unavailable.
            Using local in-memory fallback.

            Error:
            {e}
            """
        )

    return _DYNAMO_OK


# =========================================================
# CHECK AT STARTUP
# =========================================================

_check_dynamo()

# =========================================================
# GET DYNAMODB RESOURCE
# =========================================================

def _db():

    return boto3.resource(
        "dynamodb",
        region_name=AWS_REGION
    )


# =========================================================
# SESSION MANAGEMENT
# =========================================================

def get_session(session_id):

    """
    Fetch session context.
    """

    if _DYNAMO_OK:

        try:

            table = _db().Table(
                SESSION_TABLE
            )

            response = table.get_item(
                Key={
                    "session_id":
                    session_id
                }
            )

            item = response.get(
                "Item",
                {}
            )

            context_json = item.get(
                "context_json",
                "{}"
            )

            return json.loads(
                context_json
            )

        except Exception as e:

            print(
                f"""
                ⚠️ get_session error:

                {e}
                """
            )

    return _local_sessions.get(
        session_id,
        {}
    )


# =========================================================
# UPDATE SESSION
# =========================================================

def update_session(
    session_id,
    context
):

    """
    Save session context.
    """

    context["last_updated"] = (
        datetime.utcnow()
        .isoformat()
    )

    if _DYNAMO_OK:

        try:

            table = _db().Table(
                SESSION_TABLE
            )

            table.put_item(
                Item={

                    "session_id":
                    session_id,

                    "context_json":
                    json.dumps(context),

                    "last_active":
                    datetime.utcnow().isoformat(),

                    "channel":
                    context.get(
                        "channel",
                        "web"
                    ),

                    "employee_id":
                    context.get(
                        "employee_id",
                        ""
                    )
                }
            )

            return

        except Exception as e:

            print(
                f"""
                ⚠️ update_session error:

                {e}
                """
            )

    _local_sessions[
        session_id
    ] = context


# =========================================================
# ADD HISTORY
# =========================================================

def add_to_history(

    session_id,
    role,
    content,
    agent_used=""

):

    """
    Store one conversation message.
    """

    entry = {

        "session_id":
        session_id,

        "id":
        (
            f"{session_id}_"
            f"{int(time.time()*1000)}_"
            f"{uuid.uuid4().hex[:6]}"
        ),

        "role":
        role,

        "content":
        content,

        "agent_used":
        agent_used,

        "timestamp":
        datetime.utcnow().isoformat() + "Z"
    }

    if _DYNAMO_OK:

        try:

            table = _db().Table(
                HISTORY_TABLE
            )

            table.put_item(
                Item=entry
            )

            return

        except Exception as e:

            print(
                f"""
                ⚠️ add_to_history error:

                {e}
                """
            )

    _local_history.setdefault(
        session_id,
        []
    ).append(entry)


# =========================================================
# GET HISTORY
# =========================================================

def get_history(
    session_id,
    last_n=6
):

    """
    Get latest conversation history.
    """

    if _DYNAMO_OK:

        try:

            from boto3.dynamodb.conditions import Key

            table = _db().Table(
                HISTORY_TABLE
            )

            response = table.query(

                KeyConditionExpression=
                Key("session_id").eq(
                    session_id
                ),

                ScanIndexForward=False,

                Limit=last_n * 2
            )

            items = sorted(

                response.get(
                    "Items",
                    []
                ),

                key=lambda x:
                x["timestamp"]
            )

            return items[-(last_n * 2):]

        except Exception as e:

            print(
                f"""
                ⚠️ get_history error:

                {e}
                """
            )

    history = _local_history.get(
        session_id,
        []
    )

    return history[-(last_n * 2):]


# =========================================================
# FORMAT HISTORY
# =========================================================

def format_history_str(
    session_id
):

    """
    Convert history into LLM-friendly text.
    """

    history = get_history(
        session_id
    )

    if not history:

        return (
            "No prior conversation."
        )

    lines = []

    for h in history:

        role = (
            "Employee"
            if h["role"] == "user"
            else "Assistant"
        )

        agent_used = h.get(
            "agent_used",
            ""
        )

        content = h.get(
            "content",
            ""
        )

        if agent_used:

            lines.append(
                f"""
{role} ({agent_used}):

{content}
"""
            )

        else:

            lines.append(
                f"""
{role}:

{content}
"""
            )

    return "\n".join(lines)


# =========================================================
# CLEAR SESSION
# =========================================================

def clear_session(
    session_id
):

    """
    Remove session and history.
    """

    if _DYNAMO_OK:

        try:

            _db().Table(
                SESSION_TABLE
            ).delete_item(
                Key={
                    "session_id":
                    session_id
                }
            )

        except Exception as e:

            print(e)

    if session_id in _local_sessions:

        del _local_sessions[
            session_id
        ]

    if session_id in _local_history:

        del _local_history[
            session_id
        ]

    print(
        f"""
        🗑️ Session Cleared:
        {session_id}
        """
    )
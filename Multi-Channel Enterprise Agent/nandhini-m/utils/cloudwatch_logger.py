import boto3
import json
import os
import logging

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

LOG_GROUP = os.getenv(
    "CLOUDWATCH_LOG_GROUP",
    "/nandhini/pharma-agent/audit"
)

# =========================================================
# LOCAL LOGGER FALLBACK
# =========================================================

logging.basicConfig(

    level=logging.INFO,

    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)

local_logger = logging.getLogger(
    "pharma_agent"
)

# =========================================================
# CACHE SEQUENCE TOKENS
# =========================================================

_seq_tokens = {}

# =========================================================
# CLIENT CACHE
# =========================================================

_LOG_CLIENT = None

# =========================================================
# GET CLOUDWATCH CLIENT
# =========================================================

def _get_client():

    global _LOG_CLIENT

    if _LOG_CLIENT is not None:

        return _LOG_CLIENT

    try:

        _LOG_CLIENT = boto3.client(
            "logs",
            region_name=AWS_REGION
        )

        print(
            "✅ CloudWatch Client Connected"
        )

        return _LOG_CLIENT

    except Exception as e:

        print(
            f"""
            ⚠️ CloudWatch client failed.

            Error:
            {e}
            """
        )

        return None


# =========================================================
# ENSURE LOG GROUP
# =========================================================

def _ensure_log_group(client):

    try:

        groups = client.describe_log_groups(
            logGroupNamePrefix=LOG_GROUP
        )

        exists = any(

            g["logGroupName"] == LOG_GROUP

            for g in groups.get(
                "logGroups",
                []
            )
        )

        if not exists:

            client.create_log_group(
                logGroupName=LOG_GROUP
            )

            print(
                f"""
                ✅ Created Log Group:

                {LOG_GROUP}
                """
            )

    except Exception as e:

        print(
            f"""
            ⚠️ Log group error:

            {e}
            """
        )


# =========================================================
# ENSURE LOG STREAM
# =========================================================

def _ensure_log_stream(

    client,
    stream_name

):

    try:

        client.create_log_stream(

            logGroupName=LOG_GROUP,

            logStreamName=stream_name
        )

        print(
            f"""
            ✅ Created Log Stream:

            {stream_name}
            """
        )

    except client.exceptions.ResourceAlreadyExistsException:

        pass

    except Exception as e:

        print(
            f"""
            ⚠️ Log stream error:

            {e}
            """
        )


# =========================================================
# BUILD LOG ENTRY
# =========================================================

def _build_entry(

    session_id,
    employee_id,
    agent,
    query,
    response_summary,
    metadata=None

):

    return {

        "timestamp":
        datetime.utcnow().isoformat() + "Z",

        "session_id":
        session_id,

        "employee_id":
        employee_id,

        "agent":
        agent,

        "query":
        query[:500],

        "response_summary":
        response_summary[:500],

        "metadata":
        metadata or {}
    }


# =========================================================
# SEND TO CLOUDWATCH
# =========================================================

def _send_to_cloudwatch(

    client,
    stream_name,
    entry

):

    kwargs = {

        "logGroupName":
        LOG_GROUP,

        "logStreamName":
        stream_name,

        "logEvents": [

            {
                "timestamp":
                int(
                    datetime.utcnow()
                    .timestamp() * 1000
                ),

                "message":
                json.dumps(entry)
            }
        ]
    }

    # -----------------------------------------------------
    # SEQUENCE TOKEN
    # -----------------------------------------------------

    if stream_name in _seq_tokens:

        kwargs["sequenceToken"] = (
            _seq_tokens[stream_name]
        )

    response = client.put_log_events(
        **kwargs
    )

    # -----------------------------------------------------
    # CACHE NEXT TOKEN
    # -----------------------------------------------------

    next_token = response.get(
        "nextSequenceToken"
    )

    if next_token:

        _seq_tokens[
            stream_name
        ] = next_token


# =========================================================
# MAIN LOGGER
# =========================================================

def log_agent_call(

    session_id,
    employee_id,
    agent,
    query,
    response_summary,
    metadata=None

):

    """
    Enterprise Audit Logging

    Logs:
    - HR agent calls
    - Policy queries
    - Knowledge requests
    - User interactions

    Output:
    - CloudWatch Logs
    - Local logger fallback
    """

    entry = _build_entry(

        session_id=session_id,

        employee_id=employee_id,

        agent=agent,

        query=query,

        response_summary=response_summary,

        metadata=metadata
    )

    # =====================================================
    # CLOUDWATCH
    # =====================================================

    client = _get_client()

    if client:

        stream_name = (

            f"{datetime.utcnow().strftime('%Y/%m/%d')}"
            f"/{session_id[:8]}"
        )

        try:

            _ensure_log_group(
                client
            )

            _ensure_log_stream(

                client,
                stream_name
            )

            _send_to_cloudwatch(

                client,
                stream_name,
                entry
            )

            print(
                f"""
                ✅ Audit Logged

                Agent:
                {agent}

                Session:
                {session_id}
                """
            )

            return

        except Exception as e:

            print(
                f"""
                ⚠️ CloudWatch logging failed.

                Falling back to local logger.

                Error:
                {e}
                """
            )

    # =====================================================
    # LOCAL FALLBACK
    # =====================================================

    local_logger.info(

        "AUDIT | " +
        json.dumps(entry)
    )


# =========================================================
# QUICK TEST
# =========================================================

if __name__ == "__main__":

    log_agent_call(

        session_id="demo_session",

        employee_id="EMP001",

        agent="HR_AGENT",

        query="What is my leave balance?",

        response_summary="You have 4 sick leaves remaining.",

        metadata={
            "channel": "web"
        }
    )
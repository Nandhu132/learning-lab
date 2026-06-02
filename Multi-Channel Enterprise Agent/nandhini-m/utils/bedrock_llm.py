import boto3
import json
import os

from dotenv import load_dotenv

load_dotenv()

# =====================================================
# AWS CONFIG
# =====================================================

AWS_REGION = os.getenv(
    "AWS_REGION",
    "us-east-1"
)

# =====================================================
# SINGLE MODEL
# =====================================================

MODEL_ID = os.getenv(
    "BEDROCK_MODEL",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0"
)

# =====================================================
# BEDROCK CLIENT
# =====================================================

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION
)

# =====================================================
# CALL LLM
# =====================================================

def call_llm(
    prompt: str,
    system: str = ""
) -> str:

    body = {

        "anthropic_version":
        "bedrock-2023-05-31",

        "max_tokens":
        1024,

        "temperature":
        0.3,

        "messages": [

            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    # Optional system prompt
    if system:

        body["system"] = system

    try:

        response = bedrock_client.invoke_model(

            modelId=MODEL_ID,

            body=json.dumps(body)
        )

        result = json.loads(
            response["body"].read()
        )

        return (

            result["content"][0]["text"]
            .strip()
        )

    except Exception as e:

        print(
            f"""
❌ Bedrock Error

Model:
{MODEL_ID}

Error:
{str(e)}
"""
        )

        return (
            "I'm having trouble connecting right now. "
            "Please try again later."
        )
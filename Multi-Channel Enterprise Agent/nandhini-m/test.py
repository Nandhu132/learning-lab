import boto3
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()

# ==============================
# LOAD CSV
# ==============================

df = pd.read_csv("data/hr_data.csv")

# ==============================
# BEDROCK CLIENT
# ==============================

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION")
)

MODEL_ID = os.getenv("BEDROCK_MODEL")


# ==============================
# SEARCH EMPLOYEE DATA
# ==============================

def search_employee_data(question):

    question = question.lower()

    # Search by employee id
    for emp_id in df["employee_id"].values:

        if emp_id.lower() in question:

            employee = df[df["employee_id"] == emp_id]

            return employee.to_dict(orient="records")[0]

    return None


# ==============================
# GENERATE LLM RESPONSE
# ==============================

def ask_bedrock(question, employee_data):

    if employee_data:
        context = json.dumps(employee_data, indent=2)
    else:
        context = "No employee found"

    prompt = f"""
You are an HR assistant chatbot.

Use the employee data below to answer the question.

Employee Data:
{context}

Question:
{question}

Give a professional answer.
"""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body)
    )

    response_body = json.loads(response["body"].read())

    answer = response_body["content"][0]["text"]

    return answer


# ==============================
# CHAT LOOP
# ==============================

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    employee_data = search_employee_data(question)

    answer = ask_bedrock(question, employee_data)

    print("\nBot:", answer)
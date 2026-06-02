from fastapi import FastAPI
from pydantic import BaseModel

from main import run_agent


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Pharma Enterprise Agent API"
)

# =====================================================
# REQUEST MODEL
# =====================================================

class ChatRequest(BaseModel):

    session_id: str

    employee_id: str

    message: str

    channel: str = "web"


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")

def health():

    return {

        "status": "healthy"
    }


# =====================================================
# CHAT ENDPOINT
# =====================================================

@app.post("/chat")

def chat(
    request: ChatRequest
):

    result = run_agent(

        session_id=request.session_id,

        employee_id=request.employee_id,

        user_message=request.message,

        channel=request.channel
    )

    return result
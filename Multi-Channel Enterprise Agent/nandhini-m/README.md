# Pharma Multi-Channel Enterprise Agent

Enterprise AI assistant for pharmaceutical organizations using LangGraph, AWS Bedrock, FastAPI, Streamlit, FAISS, DynamoDB, CloudWatch, and WhatsApp integration.

---

#  Features

- Multi-Agent AI Architecture
- HR Assistant
- Policy Retrieval (RAG)
- Pharma Product Knowledge Assistant
- Incentive & Sales Queries
- Streamlit Web Chat
- FastAPI Backend
- WhatsApp Bot
- Session Memory
- CloudWatch Audit Logging

---

#  System Architecture

```text
                           ┌────────────────────┐
                           │     Employee       │
                           └─────────┬──────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
           ┌────────▼────────┐              ┌────────▼────────┐
           │  Streamlit UI   │              │ WhatsApp Bot    │
           └────────┬────────┘              └────────┬────────┘
                    │                                 │
                    └────────────────┬────────────────┘
                                     │
                           ┌─────────▼─────────┐
                           │     FastAPI       │
                           │    API Layer      │
                           └─────────┬─────────┘
                                     │
                           ┌─────────▼─────────┐
                           │    LangGraph      │
                           │   Orchestrator    │
                           └─────────┬─────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
 ┌────────▼────────┐       ┌────────▼────────┐       ┌────────▼────────┐
 │    HR Agent     │       │  Policy Agent   │       │ Knowledge Agent │
 └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
          │                         │                         │
          │                         │                         │
 ┌────────▼────────┐       ┌────────▼────────┐       ┌────────▼────────┐
 │ HR CSV Dataset  │       │  FAISS Vector   │       │ Pharma Product  │
 │                 │       │      Store      │       │ & Incentive CSV │
 └─────────────────┘       └─────────────────┘       └─────────────────┘
                                     │
                           ┌─────────▼─────────┐
                           │ AWS Bedrock LLM  │
                           │ Claude Haiku     │
                           └─────────┬─────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
          ┌─────────▼─────────┐             ┌────────▼────────┐
          │ DynamoDB Memory   │             │ CloudWatch Logs │
          └───────────────────┘             └─────────────────┘
```

---


#  Installation

## 1. Clone Repository

```bash
git clone <your_gitlab_repo>
cd pharma-enterprise-agent
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

#  Run Application

## Run Streamlit UI

```bash
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

---

## Run FastAPI

```bash
uvicorn api.web_api:app --reload --port 8001
```

Swagger UI:

```text
http://127.0.0.1:8001/docs
```

---

## Run WhatsApp Webhook

```bash
uvicorn api.whatsapp_webhook:app --reload --port 8002
```

---

#  Configure ngrok

```bash
ngrok http 8002
```

Example:

```text
https://relatable-laurel-doorbell.ngrok-free.dev/whatsapp
```

---

# 📱 Configure Twilio WhatsApp

Webhook URL:

```text
https://relatable-laurel-doorbell.ngrok-free.dev/whatsapp
```

Method:

```text
POST
```

---

# Application Flow

```text
User Query
    ↓
Streamlit / WhatsApp / API
    ↓
FastAPI Backend
    ↓
run_agent()
    ↓
Load Session + History
    ↓
LangGraph Orchestrator
    ↓
Intent Classification
    ↓
Route To Correct Agent
    ↓
Retrieve Enterprise Context
    ↓
AWS Bedrock Claude
    ↓
Generate Response
    ↓
Store Session + Logs
    ↓
Return Final Response
```

---

# Session Memory

Implemented using:

- DynamoDB
- Local Memory Fallback

Stores:

- Session Context
- Chat History
- Employee Context
- Multi-channel Conversations

---

# Audit Logging

CloudWatch logging includes:

- User Query
- Session ID
- Employee ID
- Agent Used
- Response Summary
- Timestamp

---

#  Policy RAG Pipeline

```text
Policy Documents
        ↓
Text Chunking
        ↓
Titan Embeddings
        ↓
FAISS Vector Store
        ↓
Similarity Search
        ↓
Claude LLM
        ↓
Final Response
```

---

#  Sample Queries

## HR Queries

```text
What is my leave balance?
Show my PF details
Who is my manager?
```

---

## Policy Queries

```text
What is the notice period?
Explain anti-bribery policy
What is work from home policy?
```

---

## Knowledge Queries

```text
Show top cardiac brands
What is my sales incentive?
List diabetes medicines
```

---

#  API Request Example

## Request

```json
{
  "session_id": "session_001",
  "employee_id": "EMP001",
  "message": "What is my leave balance?",
  "channel": "web"
}
```

---

## Response

```json
{
  "success": true,
  "data": {
    "session_id": "session_001",
    "employee_id": "EMP001",
    "channel": "web",
    "intent": "hr",
    "agent": "HR Agent",
    "question": "What is my leave balance?",
    "answer": "You currently have 4 casual leaves remaining."
  }
}
```

---

#  Local CLI Testing

```bash
python main.py
```

Example:

```text
You: What is my salary?
```

---

#  Enterprise Features

- Multi-Agent AI Routing
- RAG-based Policy Retrieval
- Enterprise Audit Logging
- Session Persistence
- WhatsApp Integration
- Cloud-native AWS Architecture
- Scalable API Design
- Multi-channel Support

---

#  Developed For

Enterprise Pharmaceutical AI Assistant Platform
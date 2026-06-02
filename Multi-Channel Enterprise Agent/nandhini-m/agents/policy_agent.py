import os

from utils.bedrock_llm import call_llm
from utils.cloudwatch_logger import log_agent_call

from langchain_community.vectorstores import FAISS

from langchain_aws import BedrockEmbeddings

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_core.documents import (
    Document
)


# =====================================================
# PATHS
# =====================================================

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data"
)

POLICIES_DIR = os.path.join(
    DATA_DIR,
    "policies"
)

# =====================================================
# BUILD VECTORSTORE
# =====================================================

def build_vectorstore():

    documents = []

    # =================================================
    # CHECK POLICY DIRECTORY
    # =================================================

    if not os.path.exists(
        POLICIES_DIR
    ):

        return None

    # =================================================
    # LOAD TXT FILES
    # =================================================

    for file_name in os.listdir(
        POLICIES_DIR
    ):

        if not file_name.endswith(".txt"):

            continue

        file_path = os.path.join(
            POLICIES_DIR,
            file_name
        )

        content = None

        for encoding in [

            "utf-8",
            "utf-8-sig",
            "latin-1"
        ]:

            try:

                with open(
                    file_path,
                    encoding=encoding
                ) as file:

                    content = file.read()

                break

            except UnicodeDecodeError:

                continue

        if content:

            documents.append(

                Document(

                    page_content=content,

                    metadata={
                        "source": file_name
                    }
                )
            )

    # =================================================
    # NO DOCUMENTS
    # =================================================

    if not documents:

        return None

    # =================================================
    # SPLIT DOCUMENTS
    # =================================================

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=700,

        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        documents
    )

    # =================================================
    # EMBEDDINGS
    # =================================================

    embeddings = BedrockEmbeddings(

        model_id=
        "amazon.titan-embed-text-v2:0",

        region_name=os.getenv(
            "AWS_REGION",
            "us-east-1"
        )
    )

    # =================================================
    # CREATE VECTORSTORE
    # =================================================

    vectorstore = FAISS.from_documents(

        chunks,

        embeddings
    )

    return vectorstore


# =====================================================
# LOAD VECTORSTORE
# =====================================================

VECTORSTORE = build_vectorstore()

# =====================================================
# POLICY AGENT
# =====================================================

def policy_agent(
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
    # VECTORSTORE CHECK
    # =================================================

    if VECTORSTORE is None:

        state["agent_response"] = (
            "Policy documents are unavailable."
        )

        state["agent_used"] = (
            "Policy Agent"
        )

        return state

    # =================================================
    # RETRIEVE POLICY CHUNKS
    # =================================================

    search_results = VECTORSTORE.similarity_search(

        user_message,

        k=5
    )

    # =================================================
    # BUILD POLICY CONTEXT
    # =================================================

    policy_context = "\n\n".join([

        f"""
Source:
{result.metadata.get('source', 'policy')}

Policy Content:
{result.page_content}
"""

        for result in search_results
    ])

    # =================================================
    # RESPONSE STYLE
    # =================================================

    if channel == "whatsapp":

        style_note = """
Give concise WhatsApp-style answers.
Use short bullet points.
"""

    else:

        style_note = """
Give professional compliance responses.
Use bullet points where useful.
Mention exact limits, rules, timelines, or approvals clearly.
Keep responses concise and business-friendly.
"""

    # =================================================
    # SYSTEM PROMPT
    # =================================================

    system = f"""
You are PharmAssist Policy Agent for PharmD India.

You answer:
- HR policies
- leave policies
- compliance rules
- resignation rules
- pharma regulations
- anti-bribery rules
- data privacy rules
- pharmacovigilance policies

Rules:
- Give direct business-style answers
- Use professional formatting
- Mention exact rules clearly
- Keep responses concise
- Never mention AI limitations
- Never say "based on provided context"
- Never say "I cannot hallucinate"

{style_note}
"""

    # =================================================
    # FINAL PROMPT
    # =================================================

    prompt = f"""
Policy Documents:
{policy_context}

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

            employee_id=session_context.get(
                "employee_id",
                "unknown"
            ),

            agent="POLICY_AGENT",

            query=user_message,

            response_summary=response[:300]
        )

    except Exception:

        pass

    # =================================================
    # UPDATE STATE
    # =================================================

    state["agent_response"] = response

    state["agent_used"] = "Policy Agent"

    return state
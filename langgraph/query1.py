import os
import json
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN)
def hf_llm_chat(prompt: str) -> str:
    """Send a text prompt to HuggingFace LLM and return plain answer."""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat_completion(messages=messages, max_tokens=512)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a knowledgeable assistant. Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
)

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    context: str
    answer: str

@tool
def human_assistance(query: str) -> str:
    """Ask a human for help when model cannot answer."""
    human_reply = interrupt({"query": query})
    return human_reply["data"]


tools = [human_assistance]
tool_node = ToolNode(tools=tools)


def question_node(state: GraphState) -> GraphState:
    """Extract latest user question."""
    last_user_msg = state["messages"][-1].content
    return {**state, "question": last_user_msg}


def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve relevant PDF content using simple FAISS similarity search."""
    query = state["question"]
    top_k = 5
    results = vectorstore.similarity_search(query, k=top_k)
    if not results:
        return {**state, "context": "NO_MATCH"}
    context = "\n\n".join(doc.page_content for doc in results)
    return {**state, "context": context}

def generate_node(state: GraphState) -> GraphState:
    """Generate answer using RAG or fallback."""
    if state["context"] == "NO_MATCH":
        return {**state, "answer": "This query doesn't match the document."}

    # Normal RAG generation
    prompt = prompt_template.format(context=state["context"], question=state["question"])
    answer = hf_llm_chat(prompt)
    return {**state, "answer": answer}


def response_node(state: GraphState) -> GraphState:
    """Append assistant’s last answer to message history."""
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": state["answer"]}]
    }


def tools_condition(state: GraphState):
    """Decide whether to call human assistance."""
    answer = state.get("answer", "").lower()

    if "this query doesn't match" in answer:
        return "continue" 

    if "need assistance" in answer:
        return "human_assistance"

    return "continue"

builder = StateGraph(GraphState)

builder.add_node("question_node", question_node)
builder.add_node("retrieve_node", retrieve_node)
builder.add_node("generate_node", generate_node)
builder.add_node("response_node", response_node)
builder.add_node("tool_node", tool_node)

builder.set_entry_point("question_node")

builder.add_edge(START, "question_node")
builder.add_edge("question_node", "retrieve_node")
builder.add_edge("retrieve_node", "generate_node")

builder.add_conditional_edges(
    "generate_node",
    tools_condition,
    {"continue": "response_node", "human_assistance": "tool_node"}
)

builder.add_edge("tool_node", "response_node")
builder.add_edge("response_node", END)

graph = builder.compile()

if __name__ == "__main__":
    print("\n Chatbot Ready. Type 'exit' to quit.\n")
    state = None

    while True:
        user_input = input("You: ")

        if user_input.lower() in {"exit", "quit"}:
            print(" Goodbye!")
            break

        msg = HumanMessage(content=user_input)
        if state is None:
            state = {"messages": [msg], "question": "", "context": "", "answer": ""}
        else:
            state["messages"].append(msg)
        events = graph.invoke(state)

        print(json.dumps({
            "question": events["question"],
            "answer": events["answer"]
        }, indent=2))

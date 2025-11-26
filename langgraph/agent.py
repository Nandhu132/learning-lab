import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq

load_dotenv()

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Update the document with provided content."""
    global document_content
    document_content = content
    return f"Document updated.\nCurrent content:\n{document_content}"


@tool
def save(filename: str) -> str:
    """Save document to a text file."""
    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as f:
            f.write(document_content)
        return f"Document saved to '{filename}'."
    except Exception as e:
        return f"Error saving file: {str(e)}"


tools = [update, save]

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
).bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
You are Drafter, a helpful writing assistant.

- If the user wants to update or change document content, call 'update'.
- If the user wants to save/finish, call 'save'.
- Always show the current document after updates.

Current document:
{document_content}
""")

    if not state["messages"]:
        user_input = input("How can I help with your document? ")
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = llm.invoke(all_messages)

    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f" USING TOOL: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState):
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and "saved" in msg.content.lower():
            return "end"
    return "continue"


def print_messages(messages):
    for m in messages[-3:]:
        if isinstance(m, ToolMessage):
            print(f"\n TOOL RESULT: {m.content}")

graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()

def run_document_agent():
    print("\n====== DRAFTER ======\n")
    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n====== FINISHED ======\n")


if __name__ == "__main__":
    run_document_agent()

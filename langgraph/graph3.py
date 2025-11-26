from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    number1: int
    operation: str
    number2: int
    finalNumber: int
    number3: int
    operation2: str
    number4: int
    finalNumber2: int
    message: str | None

def adder(state: AgentState) -> AgentState:
    """Adds number1 + number2"""
    print(" adder (first operation)")
    state["finalNumber"] = state["number1"] + state["number2"]
    print(f"Result of first operation: {state['finalNumber']}")
    return state

def subtractor(state: AgentState) -> AgentState:
    """Subtracts number1 - number2"""
    print(" subtractor (first operation)")
    state["finalNumber"] = state["number1"] - state["number2"]
    print(f"Result of first operation: {state['finalNumber']}")
    return state

def decide_next_node(state: AgentState):
    """Routes to addition or subtraction for first operation"""
    if state["operation"] == "+":
        return "addition_operation"
    elif state["operation"] == "-":
        print("Routing to subtraction_operation")
        return "subtraction_operation"
    else:
        raise ValueError("Invalid operation. Use '+' or '-'.")

def adder2(state: AgentState) -> AgentState:
    """Adds number3 + number4"""
    print(" adder2 (second operation)")
    state["finalNumber2"] = state["number3"] + state["number4"]
    print(f"Result of second operation: {state['finalNumber2']}")
    return state


def subtractor2(state: AgentState) -> AgentState:
    """Subtracts number3 - number4"""
    print(" subtractor2 (second operation)")
    state["finalNumber2"] = state["number3"] - state["number4"]
    print(f"Result of second operation: {state['finalNumber2']}")
    return state

def decide_next_node2(state: AgentState):
    """Routes to addition or subtraction for second operation"""
    if state["operation2"] == "+":
        print("Routing to addition_operation2")
        return "addition_operation2"
    elif state["operation2"] == "-":
        print("Routing to subtraction_operation2")
        return "subtraction_operation2"
    else:
        raise ValueError("Invalid operation2. Use '+' or '-'.")


def combine_node(state: AgentState) -> AgentState:
    """Combine both operation results into a readable message"""
    message = (
        f"Your first result is {state['finalNumber']} "
        f"and your second result is {state['finalNumber2']}."
    )
    print(" Combined Message:", message)
    state["message"] = message
    return state

graph = StateGraph(AgentState)

graph.add_node("add_node", adder)
graph.add_node("subtract_node", subtractor)
graph.add_node("add_node2", adder2)
graph.add_node("subtract_node2", subtractor2)
graph.add_node("combine", combine_node)

graph.add_node("router", lambda state: state)
graph.add_node("router2", lambda state: state)

graph.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "addition_operation": "add_node",
        "subtraction_operation": "subtract_node"
    }
)

graph.add_conditional_edges(
    "router2",
    decide_next_node2,
    {
        "addition_operation2": "add_node2",
        "subtraction_operation2": "subtract_node2"
    }
)

graph.add_edge(START, "router")
graph.add_edge("add_node", "router2")
graph.add_edge("subtract_node", "router2")
graph.add_edge("add_node2", "combine")
graph.add_edge("subtract_node2", "combine")
graph.add_edge("combine", END)


app = graph.compile()
app.get_graph().print_ascii()

initial_state = AgentState(
    number1=10,
    operation="-",
    number2=5,
    number3=7,
    number4=2,
    operation2="+",
    finalNumber=0,
    finalNumber2=0,
    message=None
)

final_state = app.invoke(initial_state)

print("\n Final State:", final_state)
print("\n Message:", final_state["message"])

from langgraph.graph import StateGraph, END
from typing import TypedDict

class InputState(TypedDict):
    name: str
    values: list[int]
    operation: str
    result: int | None = None

def compute_node(state: InputState):
    """Perform addition or multiplication based on user input."""
    name = state["name"]
    values = state["values"]
    operation = state["operation"]

    if operation == "+":
        result = sum(values)
    elif operation == "*":
        result = 1
        for v in values:
            result *= v
    else:
        raise ValueError("Invalid operation. Use '+' or '*'")

    state["result"] = result
    print(f"Hi {name}, your answer is: {result}")
    return state

graph = StateGraph(InputState)
graph.add_node("compute", compute_node)
graph.set_entry_point("compute")
graph.set_finish_point("compute")

app = graph.compile()
app.get_graph().print_ascii()

input_data = {"name": "Jack Sparrow", "values": [1, 2, 3, 4], "operation": "*"}
final_state = app.invoke(input_data)
print(final_state)
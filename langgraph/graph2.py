from langgraph.graph import StateGraph, END
from typing import TypedDict

class UserState(TypedDict):
    name: str
    age: int
    skills: list[str]
    result: str | None = None

def greet_node(state: UserState):
    """This node will greet the user"""
    state["result"] = f"{state['name']}, welcome to the system!"
    return state

def age_node(state: UserState):
    """This node will describe the user's age"""
    state["result"] += f" You are {state['age']} years old!"
    return state

def skills_node(state: UserState):
    """This node will list the user's skills in a formatted string"""
    skills = ", ".join(state["skills"])
    state["result"] += f" You have skills in: {skills}"
    return state


graph = StateGraph(UserState)

graph.add_node("greet", greet_node)
graph.add_node("describe_age", age_node)
graph.add_node("list_skills", skills_node)
graph.set_entry_point("greet")
graph.add_edge("greet", "describe_age")
graph.add_edge("describe_age", "list_skills")
graph.set_finish_point("list_skills")

app = graph.compile()

input_data = {"name": "Linda", "age": 31, "skills": ["Python", "Machine Learning", "LangGraph"]}
final_state = app.invoke(input_data)
print(final_state["result"])

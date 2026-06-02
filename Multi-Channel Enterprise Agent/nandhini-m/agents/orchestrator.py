from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    END
)

from utils.bedrock_llm import call_llm

from agents.hr_agent import hr_agent
from agents.policy_agent import policy_agent
from agents.knowledge_agent import knowledge_agent


# =====================================================
# STATE
# =====================================================

class AgentState(TypedDict):

    session_id: str

    user_message: str

    intent: str

    agent_response: str

    agent_used: str

    chat_history_str: str

    session_context: dict

    channel: str


# =====================================================
# LLM INTENT CLASSIFIER
# =====================================================

def classify_intent(
    message: str
):

    prompt = f"""
You are an enterprise AI intent classifier.

Classify the employee query into EXACTLY ONE category.

=================================================

1. hr

Examples:
- salary
- leave balance
- payroll
- reimbursements
- PF
- employee information
- allowances
- manager details
- joining date

=================================================

2. policy

Examples:
- notice period
- resignation rules
- compliance
- anti-bribery
- leave policy
- work from home policy
- pharmacovigilance
- company regulations
- data privacy
- grievance process

=================================================

3. knowledge

Examples:
- medicine brands
- pharma products
- drug compositions
- product details
- incentives
- sales incentives
- payout
- target achievement
- quarterly targets
- quarterly performance
- sales performance
- top pharma brands
- incentive details

=================================================

Employee Query:
{message}

=================================================

Rules:
- Return ONLY one word
- Do NOT explain
- Output must be exactly:
hr
policy
knowledge
"""

    try:

        result = (

            call_llm(prompt)

            .lower()

            .strip()

            .split()[0]
        )

        if result in [

            "hr",
            "policy",
            "knowledge"
        ]:

            return result

    except Exception:

        pass

    return "hr"


# =====================================================
# DETECT INTENT
# =====================================================

def detect_intent(
    state: AgentState
):

    message = (

        state["user_message"]

        .strip()
    )

    intent = classify_intent(
        message
    )

    state["intent"] = intent

    return state


# =====================================================
# ROUTER
# =====================================================

def route_agent(
    state: AgentState
):

    return state["intent"]


# =====================================================
# BUILD GRAPH
# =====================================================

def build_graph():

    graph = StateGraph(
        AgentState
    )

    # =================================================
    # NODES
    # =================================================

    graph.add_node(
        "detect_intent",
        detect_intent
    )

    graph.add_node(
        "hr",
        hr_agent
    )

    graph.add_node(
        "policy",
        policy_agent
    )

    graph.add_node(
        "knowledge",
        knowledge_agent
    )

    # =================================================
    # ENTRY
    # =================================================

    graph.set_entry_point(
        "detect_intent"
    )

    # =================================================
    # ROUTING
    # =================================================

    graph.add_conditional_edges(

        "detect_intent",

        route_agent,

        {

            "hr": "hr",

            "policy": "policy",

            "knowledge": "knowledge"
        }
    )

    # =================================================
    # END STATES
    # =================================================

    graph.add_edge(
        "hr",
        END
    )

    graph.add_edge(
        "policy",
        END
    )

    graph.add_edge(
        "knowledge",
        END
    )

    return graph.compile()


# =====================================================
# COMPILED GRAPH
# =====================================================

AGENT_GRAPH = build_graph()
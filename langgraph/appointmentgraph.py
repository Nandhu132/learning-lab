from langgraph.graph import StateGraph
from typing import TypedDict

class State(TypedDict):
    request_type: str = ""  # booking / cancel / reschedule

workflow = StateGraph(State)

workflow.add_node("UserRequest", lambda state: state)

# Booking nodes
workflow.add_node("CheckDoctorExist", lambda state: state)
workflow.add_node("CheckDoctorAvailability", lambda state: state)
workflow.add_node("GenerateSlots", lambda state: state)
workflow.add_node("AvailableSlots", lambda state: state)
workflow.add_node("CheckPatientExist", lambda state: state)
workflow.add_node("AddPatient", lambda state: state)
workflow.add_node("BookAppointment", lambda state: state)
workflow.add_node("SendEmail", lambda state: state)
workflow.add_node("FinalResponse", lambda state: state)

# Cancel nodes
workflow.add_node("CancelAppointment", lambda state: state)
workflow.add_node("FinalResponse_Cancel", lambda state: state)

# Reschedule nodes
workflow.add_node("RescheduleAppointment", lambda state: state)
workflow.add_node("SendEmail_Reschedule", lambda state: state)
workflow.add_node("FinalResponse_Reschedule", lambda state: state)

def router(state: State):
    if state.request_type == "booking":
        return "CheckDoctorExist"
    if state.request_type == "cancel":
        return "CancelAppointment"
    if state.request_type == "reschedule":
        return "RescheduleAppointment"
    return "FinalResponse"   

workflow.add_conditional_edges(
    "UserRequest",
    router,
    {
        "CheckDoctorExist": "CheckDoctorExist",
        "CancelAppointment": "CancelAppointment",
        "RescheduleAppointment": "RescheduleAppointment",
        "FinalResponse": "FinalResponse"
    }
)
workflow.add_edge("CheckDoctorExist", "CheckDoctorAvailability")
workflow.add_edge("CheckDoctorAvailability", "GenerateSlots")
workflow.add_edge("GenerateSlots", "AvailableSlots")
workflow.add_edge("AvailableSlots", "CheckPatientExist")

# Patient Decision Branch
workflow.add_edge("CheckPatientExist", "AddPatient")        # New patient
workflow.add_edge("CheckPatientExist", "BookAppointment")   # Existing patient
workflow.add_edge("AddPatient", "BookAppointment")

# Booking → Email
workflow.add_edge("BookAppointment", "SendEmail")
workflow.add_edge("SendEmail", "FinalResponse")

# POST-BOOKING OPTIONS
workflow.add_edge("FinalResponse", "CancelAppointment")
workflow.add_edge("FinalResponse", "RescheduleAppointment")

# CANCEL FLOW
workflow.add_edge("CancelAppointment", "FinalResponse_Cancel")

# RESCHEDULE FLOW
workflow.add_edge("RescheduleAppointment", "SendEmail_Reschedule")
workflow.add_edge("SendEmail_Reschedule", "FinalResponse_Reschedule")
workflow.set_entry_point("UserRequest")

workflow.set_finish_point("FinalResponse")
workflow.set_finish_point("FinalResponse_Cancel")
workflow.set_finish_point("FinalResponse_Reschedule")

app = workflow.compile()
graph = app.get_graph()
png_bytes = graph.draw_mermaid_png()

with open("workflow_graph.png", "wb") as f:
    f.write(png_bytes)

print("Saved workflow_graph.png")

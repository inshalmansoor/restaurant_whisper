from langgraph.graph import StateGraph, START, END

from nodes.booking_nodes import (
    start_node_booking,
    booking_complete_node, 
    booking_checker_node,
    booking_repeater_node,
    booking_cancel
)

from nodes.order_nodes import (
    start_node,
    order_complete_node,
    order_checker_node,
    order_repeater_node,
    order_cancel
)

from nodes.supervisor_nodes import (
    supervisor_node,
    order_supervisor_node,
    booking_supervisor_node,
    complaint_supervisor_node
)

from nodes.general_nodes import (
    information_node,
    customer_details_node,
    intent_detection_node,
    refine_node
)

from nodes.customer_nodes import (
    get_customer_node,
    customer_checker_node,
    insert_customer_in_db_node,
    get_customer_from_db_node,
    get_details_from_db_node
)

from nodes.complaint_nodes import (
    complaint_customer_check_node,
    complaint_classifier_node,
    complaint_update_node,
    complaint_save_node
)

from state import MyState

# Create workflow
workflow = StateGraph(MyState)

# ROUTING FUNCTION
def supervisor_routing_fn(state: MyState) -> str:
    """Extract the routing decision from state."""
    next_step = state.get("next", END)
    return END if next_step == "FINISH" else next_step

def general_routing_fn(state: MyState) -> str:
    """Routing function for order supervisor."""
    next_step = state.get("next", "supervisor")
    return next_step

# BUILD THE GRAPH
# Start Nodes
workflow.add_node("refine", refine_node)
workflow.add_node("intent_detection", intent_detection_node)

# Supervisor Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("order_supervisor", order_supervisor_node)
#workflow.add_node("customer_supervisor", customer_supervisor_node)
workflow.add_node("booking_supervisor", booking_supervisor_node)
workflow.add_node("compaint_supervisor", complaint_supervisor_node)

workflow.add_node("information", information_node)
workflow.add_node("customer_details", customer_details_node)

# Order Agent Nodes
workflow.add_node("order_start", start_node)
workflow.add_node("order_complete", order_complete_node)
workflow.add_node("order_checker", order_checker_node)
workflow.add_node("order_repeater", order_repeater_node)
workflow.add_node("order_cancel", order_cancel)

# Booking Agent Nodes
workflow.add_node("booking_start", start_node_booking)
workflow.add_node("booking_complete", booking_complete_node)
workflow.add_node("booking_checker", booking_checker_node)
workflow.add_node("booking_repeater", booking_repeater_node)
workflow.add_node("booking_cancel", booking_cancel)

#Complaint Agent Nodes
workflow.add_node("complaint_customer_check", complaint_customer_check_node)
workflow.add_node("complaint_classifier", complaint_classifier_node)
workflow.add_node("complaint_update", complaint_update_node)
workflow.add_node("complaint_save", complaint_save_node)

#Customer Agent Nodes
workflow.add_node("get_customer", get_customer_node)
workflow.add_node("customer_checker", customer_checker_node)
workflow.add_node("insert_customer_in_db", insert_customer_in_db_node)
workflow.add_node("get_customer_from_db", get_customer_from_db_node)
workflow.add_node("get_details_from_db", get_details_from_db_node)

# Define edges
workflow.add_edge(START, "refine")
workflow.add_edge("refine", "intent_detection")
workflow.add_edge("intent_detection", "supervisor")

# Order Edges
workflow.add_edge("order_start", "supervisor")
workflow.add_edge("order_supervisor", "customer_details")
workflow.add_edge("order_complete", "order_repeater")
workflow.add_edge("order_checker", "order_repeater")
workflow.add_edge("order_repeater", "supervisor")
workflow.add_edge("order_cancel", "supervisor")

# Booking Edges
workflow.add_edge("booking_start", "supervisor")
workflow.add_edge("booking_supervisor", "customer_details")
workflow.add_edge("booking_complete", "booking_repeater")
workflow.add_edge("booking_repeater", "supervisor")
workflow.add_edge("booking_cancel", "supervisor")

workflow.add_edge("information", "supervisor")

# Customer Edges
#workflow.add_edge("customer_supervisor", "customer_checker")
workflow.add_edge("get_customer", "supervisor")
workflow.add_edge("insert_customer_in_db", "supervisor")
workflow.add_edge("get_details_from_db", "supervisor")


# Conditional routing from supervisor
workflow.add_conditional_edges(
    "supervisor",
    supervisor_routing_fn,
    ["information", "order_supervisor", "customer_checker", "booking_supervisor", END]  # Changed "order" to "order_supervisor"
)

# Conditional routing from order_supervisor
workflow.add_conditional_edges(
    "customer_details",
    general_routing_fn,
    {
        "order_start": "order_start",
        "order_complete": "order_complete",
        "order_checker": "order_checker",  # ADD
        "order_repeater": "order_repeater",  # ADD
        "order_cancel": "order_cancel",
        "booking_start": "booking_start",
        "booking_complete": "booking_complete",
        "booking_checker": "booking_checker",  # ADD
        "booking_repeater": "booking_repeater",  # ADD
        "booking_cancel": "booking_cancel",
        "get_customer": "get_customer",
        "supervisor": "supervisor"
    }
)

#Conditional routing from customer_checker
workflow.add_conditional_edges(
    "customer_checker",
    general_routing_fn,
    {
        "get_customer_from_db":"get_customer_from_db",
        "supervisor": "supervisor"
    }
)

#Conditional routing from customer_checker
workflow.add_conditional_edges(
    "booking_checker",
    general_routing_fn,
    {
        "supervisor":"supervisor",
        "booking_repeater": "booking_repeater"
    }
)

workflow.add_conditional_edges(
    "get_customer_from_db",
     general_routing_fn,
    {
        "get_details_from_db": "get_details_from_db",
        "insert_customer_in_db": "insert_customer_in_db"
    }
)

# Compile
compiled = workflow.compile()
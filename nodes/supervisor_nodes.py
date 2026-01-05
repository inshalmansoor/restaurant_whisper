from chat_model import chat_model
from state import MyState
from typing import Dict, Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage

class SupervisorDecision(BaseModel):
    next_node: Literal[
        "information",
        "order_supervisor",
        "customer_checker",
        "booking_supervisor",
        "complaint_supervisor",
    ]

def supervisor_node(state: MyState) -> Dict:
    """
    Supervisor that asks the LLM to return a single JSON object:
      {"next_node": "<one_of_allowed_nodes>"}

    The LLM decides the next node without any hard-coded if/else selection logic.
    """
    print("\n" + "="*80)
    print("👔 SUPERVISOR NODE")
    print("="*80)

    query_list = state["query_list"]
    processed_queries = state["processed_queries"]
    query_responses = state["query_responses"]

    print(f"Progress: {len(processed_queries)}/{len(query_list)} queries processed")

    # Check if all queries are processed
    if len(processed_queries) >= len(query_list):
        print("\n✅ All queries completed!")
        final_response = " ".join(query_responses)
        print(f"FINAL RESPONSE: {final_response}\n")
        return {"messages": state["messages"] + [SystemMessage(content=final_response)], "next": "FINISH"}

    # Get next query to process
    current_query = query_list[len(processed_queries)]
    print(f"\n📋 Query #{len(processed_queries) + 1}: {current_query}")

    # System prompt: request a JSON object ONLY with next_node
    supervisor_prompt = f"""You are a restaurant supervisor deciding which specialist node should handle a customer query.

                            Available nodes:
                            - information
                            - order_supervisor
                            - customer_checker
                            - booking_supervisor
                            - complaint_supervisor

                            Current situation:
                            - Customer query: "{current_query}"
                            - Progress: {len(processed_queries)}/{len(query_list)} completed
                            - Remaining: {len(query_list) - len(processed_queries)}

                            Your task:
                            Choose exactly one node from the list above to handle this query.

                            **REPLY WITH ONLY A SINGLE JSON OBJECT** (no extra text, no explanation), for example:
                            {{"next_node": "information"}}
                            Make sure the value is one of the allowed nodes exactly.
                          """

    # Single LLM call with the supervisor prompt
    response = chat_model.invoke([SystemMessage(content=supervisor_prompt)])

    raw = response.content.strip()
    print(f"LLM raw output: {raw}")

    # Try to parse with Pydantic
    try:
        decision = SupervisorDecision.parse_raw(raw)
        next_node = decision.next_node
    except ValidationError as e:
        # Minimal, explicit fallback — keeps behaviour predictable while remaining simple.
        print("⚠️ Supervisor parsing failed. Falling back to 'information'. Validation error:")
        print(e)
        next_node = "information"

    print(f"🎯 Supervisor routes to: {next_node}\n")

    # Return chosen node (and keep the original 'input' key as before)
    return {"input": current_query, "next": next_node}

class OrderSupervisorDecision(BaseModel):
    next_node: Literal["order_checker", "order_repeater", "order_start", "order_complete", "order_cancel"]

def order_supervisor_node(state: Dict) -> Dict:
    """
    Sub-supervisor that routes order-related queries to specific order handlers.
    Uses the LLM to return a JSON object: {"next_node": "<handler>"} and parses it with Pydantic.
    """
    print("\n" + "="*80)
    print("🍽️  ORDER SUPERVISOR NODE")
    print("="*80)

    current_query = state.get("input")

    print(f"ANALYZING ORDER QUERY: {current_query}")

    order_supervisor_prompt = f"""You are an order management supervisor. Choose exactly one handler for the query.

                              Available handlers:
                              - order_start
                              - order_complete
                              - order_checker
                              - order_repeater
                              - order_cancel

                              Current order query: "{current_query}"

                              Your task:
                              Reply with ONLY a single JSON object (no extra text or explanation), for example:
                              {{"next_node": "order_checker"}}

                              Make sure the value is exactly one of the allowed handlers.
                              """

    # Single LLM call
    response = chat_model.invoke([SystemMessage(content=order_supervisor_prompt)])
    raw = response.content.strip()
    print(f"LLM raw output: {raw}")

    # Parse with Pydantic
    try:
        decision = OrderSupervisorDecision.parse_raw(raw)
        next_node = decision.next_node
    except ValidationError as e:
        print("⚠️ Parsing failed. Falling back to 'order_checker'. Validation error:")
        print(e)
        next_node = "order_checker"

    print(f"🎯 Order Supervisor routes to: {next_node}\n")
    return {"next": next_node}

class BookingSupervisorDecision(BaseModel):
    next_node: Literal["booking_checker", "booking_repeater", "booking_start", "booking_complete", "booking_cancel"]

def booking_supervisor_node(state: Dict) -> Dict:
    """
    Sub-supervisor that routes booking-related queries to specific order handlers.
    Uses the LLM to return a JSON object: {"next_node": "<handler>"} and parses it with Pydantic.
    """
    print("\n" + "="*80)
    print("🍽️  BOOKING SUPERVISOR NODE")
    print("="*80)

    current_query = state.get("input")

    print(f"ANALYZING BOOKING QUERY: {current_query}")

    order_supervisor_prompt = f"""You are an booking management supervisor. Choose exactly one handler for the query.

                              Available handlers:
                              - booking_start
                              - booking_complete
                              - booking_checker
                              - booking_repeater
                              - booking_cancel

                              Current order query: "{current_query}"

                              Your task:
                              Reply with ONLY a single JSON object (no extra text or explanation), for example:
                              {{"next_node": "booking_checker"}}

                              Make sure the value is exactly one of the allowed handlers.
                              """

    # Single LLM call
    response = chat_model.invoke([SystemMessage(content=order_supervisor_prompt)])
    raw = response.content.strip()
    print(f"LLM raw output: {raw}")

    # Parse with Pydantic
    try:
        decision = BookingSupervisorDecision.parse_raw(raw)
        next_node = decision.next_node
    except ValidationError as e:
        print("⚠️ Parsing failed. Falling back to 'booking_checker'. Validation error:")
        print(e)
        next_node = "booking_checker"

    print(f"🎯 Booking Supervisor routes to: {next_node}\n")
    return {"next": next_node}

class ComplaintSupervisorDecision(BaseModel):
    next_node: Literal[
        "complaint_customer_check",
        "complaint_classifier",
        "complaint_save"
    ]

def complaint_supervisor_node(state: MyState) -> Dict:
    print("\n" + "="*80)
    print("📣 COMPLAINT SUPERVISOR")
    print("="*80)

    current_query = state["input"]

    prompt = f"""
    You are a complaint supervisor.

    Decide the next step:
    - complaint_customer_check → if customer info may be required
    - complaint_classifier → if complaint text is being discussed
    - complaint_save → if complaint seems complete

    Query: "{current_query}"

    Reply ONLY JSON:
    {{ "next_node": "complaint_customer_check" }}
    """

    response = chat_model.invoke([SystemMessage(content=prompt)])
    raw = response.content.strip()
    print(f"LLM raw output: {raw}")

    try:
        decision = ComplaintSupervisorDecision.parse_raw(raw)
        next_node = decision.next_node
    except ValidationError:
        next_node = "complaint_customer_check"

    return {"next": next_node}

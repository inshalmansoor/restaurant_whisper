from chat_model import chat_model
from state import MyState
from typing import Dict, Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage
from utils import get_conversation_context
from pydantic import ValidationError
from output_schema import (
    SupervisorDecision,
    OrderSupervisorDecision,
    BookingSupervisorDecision,
    ComplaintSupervisorDecision
)

def supervisor_node(state: Dict) -> Dict:
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
        final_response = " ".join(
            r.content if hasattr(r, "content") else str(r) for r in query_responses
        )
        for r in query_responses:
            print(f"response: {r}")
        print(f"FINAL RESPONSE: {final_response}\n")
        return {"messages": state["messages"] + [SystemMessage(content=final_response)], "next": "FINISH"}


    # Get next query to process
    current_query = query_list[len(processed_queries)]
    print(f"\n📋 Query #{len(processed_queries) + 1}: {current_query}")

    conversation_history = get_conversation_context(state["messages"])

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
                            - Conversation history: "{conversation_history}"
                            - Progress: {len(processed_queries)}/{len(query_list)} completed
                            - Remaining: {len(query_list) - len(processed_queries)}

                            Your task:
                            Choose exactly one node from the list above to handle this query and return ONLY a single JSON object with the key "next_node" whose value is exactly one of the allowed nodes. No extra text, no explanation, no logs.

                            Selection rules (apply exactly):
                            1) INTENT CLASSIFICATION: Examine both the current_query and the conversation_history to identify the user's primary intent. Determine whether the intent is best handled as:
                              - information: factual or clarifying requests (menu, hours, location, services, discounts, ambiguous questions that need clarification),
                              - order_supervisor: any order-related action (create, modify, add, remove, cancel items, place orders, reorder, request order status),
                              - customer_checker: creation or update of customer personal data (name, phone, email, delivery address, profile update),
                              - booking_supervisor: reservation/booking actions (create, modify, cancel reservations, change party size, reschedule),
                              - complaint_supervisor: complaints, feedback, refunds, service/food quality issues, or escalation requests.

                            2) AMBIGUITY & FALLBACK:
                              - If you cannot reliably classify the primary intent (unclear/ambiguous language or equal-weight multiple intents), route to information.
                              - Do NOT guess or invent missing details to force a classification.

                            3) DECISION CONSTRAINTS:
                              - Use both current_query and conversation_history to make the decision.
                              - Do NOT perform clarifications here — classification only.
                              - Do NOT include any text other than the single JSON object required below.

                            Return the single JSON object only.
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

def order_supervisor_node(state: Dict) -> Dict:
    """
    Sub-supervisor that routes order-related queries to specific order handlers.
    Uses the LLM to return a JSON object: {"next_node": "<handler>"} and parses it with Pydantic.
    """
    print("\n" + "="*80)
    print("🍽️  ORDER SUPERVISOR NODE")
    print("="*80)

    current_query = state.get("input")

    conversation_history = get_conversation_context(state["messages"])

    print(f"ANALYZING ORDER QUERY: {current_query}")

    order_supervisor_prompt = f"""You are an order management supervisor. Choose exactly one handler for the query.

                                  Available handlers:
                                  - order_start
                                  - order_complete
                                  - order_checker
                                  - order_repeater
                                  - order_cancel

                                  Current order query: "{current_query}"
                                  Conversation history: "{conversation_history}"

                                  Your task:
                                  Return ONLY a single JSON object and nothing else. The object must contain exactly one key: "next_node", and its value MUST be exactly one of the allowed handlers. No extra text, no explanation, no examples, no logs.

                                  Selection rules (apply these precisely, using the priority order listed to resolve overlaps):
                                  1) order_cancel — choose when the user intent is to cancel or stop an existing order (cancellation intent found in the current query or conversation history). Highest priority.
                                  2) order_complete — choose when the user explicitly confirms the order is finished/placed/confirmed or when conversation history shows a pending order and the current query indicates final confirmation.
                                  3) order_repeater — choose ONLY when the user explicitly requests the assistant to *repeat, read back, or recite* the current or most-recent order details (e.g., user asks "repeat my order", "what did I order", "tell me my order again", or "read back my order").
                                    - DO NOT route requests to resend receipts, place the same order again ("reorder"), request an extra item, or any actionable modification to this node. Those are NOT repetition intents.
                                  4) order_checker — choose when the user creates or modifies an order: adding items, removing items, updating quantities/modifiers, placing a new order, or when the user intends to place the same items again ("reorder") or requests a receipt/resend. Treat these actionable intents as order_checker.
                                  5) order_start — choose only when none of the above apply, i.e., the query is ambiguous regarding action or explicitly requests to start/begin an order but contains no executable item details.

                                  Decision constraints:
                                  - Use the priority order above to pick a single handler when multiple intents overlap (always select the highest-priority applicable handler).
                                  - Base the decision on both the current_query and the conversation_history.
                                  - Do NOT invent missing details, perform clarifications, or append clarifying text to the selected handler decision — selection is purely classification.
                                  - The output must be exactly one JSON object with a single "next_node" key and a single allowed handler value.

                                  Produce the JSON object only.
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

def booking_supervisor_node(state: Dict) -> Dict:
    """
    Sub-supervisor that routes booking-related queries to specific order handlers.
    Uses the LLM to return a JSON object: {"next_node": "<handler>"} and parses it with Pydantic.
    """
    print("\n" + "="*80)
    print("🍽️  BOOKING SUPERVISOR NODE")
    print("="*80)

    current_query = state.get("input")

    conversation_history = get_conversation_context(state["messages"])

    print(f"ANALYZING BOOKING QUERY: {current_query}")

    booking_supervisor_prompt = f"""You are a booking management supervisor. Choose exactly one handler for the query.

                                    Available handlers:
                                    - booking_start
                                    - booking_complete
                                    - booking_checker
                                    - booking_repeater
                                    - booking_cancel

                                    Current booking query: "{current_query}"
                                    Conversation history: "{conversation_history}"

                                    Your task:
                                    Return ONLY a single JSON object and nothing else. The object must contain exactly one key: "next_node", and its value MUST be exactly one of the allowed handlers. No extra text, no explanation, no examples, no logs.

                                    Selection rules (apply these precisely, using the priority order listed to resolve overlaps):
                                    1) booking_cancel — choose when the user intent is to cancel or stop an existing booking/reservation (cancellation intent found in the current query or conversation history). Highest priority.
                                    2) booking_complete — choose when the user explicitly confirms the booking is finished/confirmed/placed (e.g., "yes confirm", "that's all, confirm booking") or when conversation history shows a pending booking and the current query indicates final confirmation.
                                    3) booking_repeater — choose ONLY when the user explicitly requests the assistant to *repeat, read back, or recite* the current or most-recent booking details (e.g., "repeat my booking", "what is my reservation", "tell me my booking details again").
                                      - DO NOT route requests to rebook, reschedule, place the same booking again ("rebook"), or perform any actionable modification to this node. Those are NOT repetition intents.
                                    4) booking_checker — choose when the user creates or modifies a booking: making a new reservation, changing date/time/party size/location, adding or removing guests, rescheduling, or any actionable booking request (including "book a table for 4", "change my reservation to 8pm"). Treat actionable intents such as rebook/reschedule/modify as booking_checker.
                                    5) booking_start — choose only when none of the above apply, i.e., the query is ambiguous regarding action or explicitly requests to start/begin a booking but contains no executable details required to perform the booking.

                                    Decision constraints:
                                    - Use the priority order above to pick a single handler when multiple intents overlap (always select the highest-priority applicable handler).
                                    - Base the decision on both the current_booking query and the conversation_history.
                                    - Do NOT invent missing details, perform clarifications, or append clarifying text to the selected handler decision — selection is purely classification.
                                    - The output must be exactly one JSON object with a single "next_node" key and a single allowed handler value.

                                    Produce the JSON object only.
                                    """


    # Single LLM call
    response = chat_model.invoke([SystemMessage(content=booking_supervisor_prompt)])
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

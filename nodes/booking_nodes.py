from state import MyState
from prompt_templates import StartBooking, BookingDetailManager, BookingRepeater 
from models import Customer, BookingDoc
from utils import safe_int, get_conversation_context
from typing import Dict
from datetime import datetime

def start_node_booking(state: MyState) -> Dict:
    """Handles booking initiation or other ambiguious booking related queries"""
    print("\n" + "="*80)
    print("🔍 Booking Start NODE")
    print("="*80)

    current_query = state["input"]
    print(f"PROCESSING: {current_query}")

    booking = state['booking']

    conversation_history = get_conversation_context(state["messages"])

    start_booking = StartBooking()
    response = start_booking.execute_query(
        query = current_query,
        booking = booking,
        conversation_history=conversation_history
    )

    print(f"RESPONSE: {response}\n")
    print("ROUTING BACK TO: supervisor")

    return {
        "processed_queries": state["processed_queries"] + [current_query],
        "query_responses": state["query_responses"] + [str(response)],
        "next": "supervisor"
    }

def booking_complete_node(state: dict) -> Dict:
    """Handles booking completion: update an existing booking (if booking_id present)
    or create a new booking. Updates state['booking']['booking_id'].
    """
    print("\n" + "=" * 80)
    print("🔍 BOOKING COMPLETE NODE")
    print("=" * 80)

    current_query = state.get("input")
    print(f"PROCESSING: {current_query}")

    booking = state.get("booking", {}) or {}
    customer = state.get("customer", {}) or {}
    customer_id = customer.get("customer_id")
    booking_id = booking.get("booking_id")

    # Extract booking fields from state['booking']
    b_date = booking.get("date", "")        # YYYY-MM-DD
    b_time = booking.get("time", "")        # HH:MM
    b_location = booking.get("location", "") or ""
    b_guests = safe_int(booking.get("guests"))
    b_special = booking.get("special_requests", "") or ""

    # Resolve reservation_date: combine date and time if present, else use utcnow()
    reservation_date = None
    if b_date and b_time:
        try:
            reservation_date = datetime.strptime(f"{b_date} {b_time}", "%Y-%m-%d %H:%M")
        except Exception:
            reservation_date = datetime.utcnow()
    elif b_date:
        try:
            reservation_date = datetime.strptime(b_date, "%Y-%m-%d")
        except Exception:
            reservation_date = datetime.utcnow()
    else:
        reservation_date = datetime.utcnow()

    # fetch customer document (expecting it to exist)
    cust_doc = None
    if customer_id:
        cust_doc = Customer.objects(customer_id=customer_id).first()

    # Update existing booking if booking_id present
    booking_doc = None
    if booking_id:
        booking_doc = BookingDoc.objects(booking_id=booking_id).first()
        if booking_doc:
            # update simple fields
            booking_doc.location = b_location
            booking_doc.reservation_date = reservation_date
            if b_guests is not None:
                booking_doc.guests = b_guests
            # (we store special_requests in booking_doc.location or a separate field if you add one;
            # for now we keep it out since model doesn't have special_requests)
            booking_doc.save()

    # If no existing booking_doc, create new one
    if not booking_id or not booking_doc:
        booking_doc = BookingDoc(
            customer=cust_doc,
            location=b_location,
            reservation_date=reservation_date,
            guests=(b_guests if b_guests is not None else None),
            booking_placement_date=datetime.utcnow()
        )
        booking_doc.save()
        # booking_doc.booking_id is available after save
        booking_id = booking_doc.booking_id
        booking["booking_id"] = booking_id

    print("Booking saved/updated in DB (booking_id={})".format(booking_id))
    print("ROUTING TO: booking_repeater\n")

    return {
        "processed_queries": state.get("processed_queries", []) + [current_query],
        "query_responses": state.get("query_responses", []) + ["Your booking has been noted."],
        "next": "booking_repeater",
        "customer": customer,
        "booking": booking
    }


def booking_checker_node(state: MyState) -> Dict:
    """Handles booking checking: use the BookingDetailManager to extract/merge booking fields."""
    print("\n" + "="*80)
    print("🔍 BOOKING CHECKER NODE")
    print("="*80)

    current_query = state.get("input")
    print(f"PROCESSING: {current_query}")

    conversation_history = get_conversation_context(state.get("messages", []))

    booking = state.get("booking", {}) or {}

    # Run the BookingDetailManager to extract/merge booking fields
    booking_manager = BookingDetailManager()
    result = booking_manager.execute_query(
        query=current_query,
        booking=booking,
        conversation_history=conversation_history
    )

    # result should follow BookingDetailOutput: { "details": {...}, "response": "...", "info_status": "complete"/"incomplete" }
    details = result.get("details", {}) if isinstance(result, dict) else {}
    response_text = result.get("response", "") if isinstance(result, dict) else ""
    info_status = result.get("response", "") if isinstance(result, dict) else "incomplete"

    # If details is a pydantic model, convert to dict
    if hasattr(details, "dict"):
        details = details.dict()

    # Update state booking with merged details
    state["booking"].update(details)

    print(f"RESPONSE: {response_text}\n")
    print("ROUTING BACK TO: booking_repeater")

    if info_status == "incomplete":
        next = "supervisor"
    elif info_status == "complete":
        next = ""

    return {
        "processed_queries": state.get("processed_queries", []),
        "query_responses": state.get("query_responses", []) + [response_text],
        "booking": state["booking"],
        "next": next
    }

def booking_cancel(state: MyState) -> Dict:
    """Handles order cancellation - cancels current order."""
    print("\n" + "=" * 80)
    print("📋 ORDER CANCEL NODE")
    print("=" * 80)

    booking = state.get("booking", {}) or {}
    current_query = state.get("input")
    print(f"PROCESSING: {current_query}")

    # try to get order_id from state.order first, then from top-level state
    booking_id = booking.get("booking_id") if isinstance(booking, dict) else None
    if not booking_id:
        response = "Nothing to delete."
    else:
        # find the order doc and delete order items + order
        booking_doc = BookingDoc.objects(booking_id=booking_id).delete()
        response = f"Deleted booking and its items for booking_id={booking_id}"

    booking = {}

    print(response)
    print("ROUTING BACK TO: order_repeater")

    return {
        "processed_queries": state.get("processed_queries", []) + [current_query],
        "query_responses": state.get("query_responses", []) + [response],
        "next": "supervisor",
        "booking": booking
    }


def booking_repeater_node(state: MyState) -> Dict:
    """Handles booking repetition - summarizes current booking."""
    print("\n" + "="*80)
    print("📋 BOOKING REPEATER NODE")
    print("="*80)

    booking = state.get("booking")

    current_query = state["input"]
    print(f"PROCESSING: {current_query}")

    booking_repeater = BookingRepeater()
    response = booking_repeater.execute_query(
        "", [], booking=booking
    )

    print(f"RESPONSE: {response}\n")
    print("ROUTING BACK TO: supervisor")

    return {
        "processed_queries": state["processed_queries"] + [current_query],
        "query_responses": state["query_responses"] + [str(response)],
        "next": "supervisor"
    }

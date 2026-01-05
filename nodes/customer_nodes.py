from state import MyState
from prompt_templates import CustomerDetailManager
from models import Customer, OrderDoc, OrderItem, BookingDoc
from datetime import datetime
from typing import Dict
from utils import get_conversation_context

def get_customer_node(state: MyState):
    """Prompts user to provide customer details"""
    print("\n" + "="*80)
    print("🔍 Customer get NODE")
    print("="*80)

    current_query = state["input"]
    print(f"PROCESSING: {current_query}")

    print("ROUTING BACK TO: supervisor")

    return {
        "processed_queries": state["processed_queries"] + [current_query],
        "query_responses": state["query_responses"] + [str("Could you please provide your name, phone number, email and delivery address first?")],
        "next": "supervisor"
    }

def customer_checker_node(state: MyState) -> Dict:
    """Handles customer details checking - validates customer details."""
    print("\n" + "="*80)
    print("🔍 CUSTOMER CHECKER NODE")
    print("="*80)

    current_query = state["input"]
    customer = state.get("customer")
    print(f"PROCESSING: {current_query}")
    print(f"Customer: {customer}")

    conversation_history = get_conversation_context(state["messages"])
    customer_details = state['customer']

    customer_checker = CustomerDetailManager()
    response = customer_checker.execute_query(
        query=current_query,
        conversation_history=conversation_history,
        customer_details = customer
    )

    customer_details['name'] = response.details.name if response.details.name else customer_details.get('name', None)
    customer_details['phone_number'] = response.details.phone_number if response.details.phone_number else customer_details.get('phone_number', None)
    customer_details['address'] = response.details.address if response.details.address else customer_details.get('address', None)
    customer_details['additional_details'] = response.details.additional_details if response.details.additional_details else customer_details.get('additional_details', None)

    print(f"RESPONSE: {response}\n")


    if customer_details.get("phone_number"):
        print("ROUTING BACK: get_customer_from_db")
        next_node = "get_customer_from_db"
    else:
        print("ROUTING BACK: supervisor")
        next_node = "supervisor"

    return {
        "processed_queries": state["processed_queries"] + [current_query],
        "query_responses": state["query_responses"] + [str(response.response)],
        "customer": customer_details,
        "next": next_node
    }

def insert_customer_in_db_node(state: dict) -> Dict:
    """Check if Customer exists in db by phone_number. If not, create one.
    Update state['customer']['customer_id'] and return it in the 'customer' key.
    """
    print("\n" + "="*80)
    print("🔍 CUSTOMER INSERT NODE")
    print("="*80)

    customer = state["customer"]
    phone_number = customer.get("phone_number")
    name = customer.get("name", "")
    address = customer.get("address", "")

    # create and save a new customer (customer_id will be set by model default)
    new_customer = Customer(name=name, phone_number=phone_number, address=address)
    new_customer.save()
    customer_id = new_customer.customer_id

    # store customer_id back into the state customer dict
    customer["customer_id"] = customer_id

    print("ROUTING BACK: supervisor")

    return {
        "processed_queries": state.get("processed_queries"),
        "query_responses": state.get("query_responses"),
        "next": "supervisor",
        "customer": customer
    }

def get_customer_from_db_node(state: dict) -> Dict:
    """Check if Customer exists in db by phone_number. If not, create one.
    Update state['customer']['customer_id'] and return it in the 'customer' key.
    """
    print("\n" + "="*80)
    print("🔍 CUSTOMER Retrieval NODE")
    print("="*80)

    customer = state["customer"]
    phone_number = customer.get("phone_number")
    name = customer.get("name", "")
    address = customer.get("address", "")
    queries = state.get("query_responses", [])

    # Try to find existing customer by phone_number
    existing = Customer.objects(phone_number=phone_number).first()

    if existing:
        if name:
            existing.name = name
        if address:
            existing.address = address
        existing.save()
        customer_id = existing.customer_id
        customer["customer_id"] = customer_id
        queries.pop()
        queries.append("Your details are noted!!!")

        next_node = "get_details_from_db"
        print("ROUTING BACK: get_detais_from_db")
    else:
        next_node = "insert_customer_in_db"
        print("ROUTING BACK: insert_customer_in_db")

    return {
        "query_responses": queries,
        "next": next_node,
        "customer": customer
    }

def get_details_from_db_node(state: dict) -> Dict:
    """
    Retrieve the customer's incomplete order (if any) and their latest booking (if any).
    Returns both in `order` and `booking` keys respectively.
    """
    print("\n" + "=" * 80)
    print("🔍 ORDER & BOOKING Retrieval NODE")
    print("=" * 80)

    customer = state.get("customer", {})
    customer_id = customer.get("customer_id")

    cust_doc = None
    if customer_id:
        cust_doc = Customer.objects(customer_id=customer_id).first()

    # Default outputs
    order_dict = state.get("order")
    booking_dict = state.get("booking")

    # --- Order retrieval (same behaviour as before) ---
    order_doc = None
    if cust_doc:
        order_doc = OrderDoc.objects(customer=cust_doc, status="incomplete").first()

    if order_doc:
        items_qs = OrderItem.objects(order=order_doc)
        items = {}
        for it in items_qs:
            # item name : [quantity, price]
            items[it.item] = [it.quantity, it.price]
        order_dict['items'] = {} if not order_dict.get('items', {}) else order_dict['items']
        order_dict['items'].update(items)
        order_dict.update({
            "order_id": order_doc.order_id,
            "order_total": order_doc.total
        })
        # optional summarization — keep it simple
        order_message = f"You have an incomplete order (id: {order_doc.order_id})."
    else:
        order_message = None

    # --- Booking retrieval (fetch most-recent booking for this customer) ---
    booking_doc = None
    if cust_doc:
        # model does not have a status field in your provided model,
        # so simply take the most recent booking (if any)
        booking_doc = BookingDoc.objects(customer=cust_doc).order_by('-booking_placement_date').first()

    if booking_doc:
        # convert reservation_date to date/time strings if available
        rd = booking_doc.reservation_date
        if isinstance(rd, datetime):
            date_str = rd.strftime("%Y-%m-%d")
            time_str = rd.strftime("%H:%M")
        else:
            date_str = ""
            time_str = ""

        booking_dict.update({
            "booking_id": booking_doc.booking_id,
            "location": getattr(booking_doc, "location", "") or "",
            "date": date_str,
            "time": time_str,
            "guests": getattr(booking_doc, "guests", None),
            "booking_placement_date": booking_doc.booking_placement_date.isoformat() if getattr(booking_doc, "booking_placement_date", None) else ""
        })

        booking_message = (
            f"You have an existing booking (id: {booking_doc.booking_id}) "
            f"for {booking_dict.get('guests') or 'N/A'} guests on {booking_dict.get('date') or 'N/A'} "
            f"at {booking_dict.get('time') or 'N/A'} at {booking_dict.get('location') or 'N/A'}."
        )
    print("ROUTING back to supervisor")
    return {
        "next": "supervisor",
        "order": order_dict,
        "booking": booking_dict
    }
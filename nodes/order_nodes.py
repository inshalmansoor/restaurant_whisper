from state import MyState
from prompt_templates import StartOrder, OrderItemChecker, OrderRepeater
from models import Customer, OrderDoc, OrderItem
from utils import safe_int, safe_float, get_conversation_context
from typing import Dict
from nodes.general_nodes import get_context

def start_node(state: MyState) -> Dict:
    """Handles order initiation or other ambiguious order related queries"""
    print("\n" + "="*80)
    print("🔍 ORDER Start NODE")
    print("="*80)

    current_query = state["input"]
    print(f"PROCESSING: {current_query}")

    order = state['order']

    conversation_history = get_conversation_context(state["messages"])

    start_order = StartOrder()
    response = start_order.execute_query(
        query = current_query,
        order = order,
        conversation_history=conversation_history
    )

    print(f"RESPONSE: {response}\n")
    print("ROUTING BACK TO: supervisor")

    return {
        "processed_queries": state["processed_queries"] + [current_query],
        "query_responses": state["query_responses"] + [str(response)],
        "next": "supervisor"
    }

def order_complete_node(state: dict) -> Dict:
    """Handles order completion: update an existing order (if order_id present)
    or create a new order and its items. Updates state['order']['order_id'].
    """
    print("\n" + "=" * 80)
    print("🔍 ORDER Complete NODE")
    print("=" * 80)

    current_query = state.get("input")
    print(f"PROCESSING: {current_query}")

    order = state.get("order", {})
    customer = state.get("customer", {})
    customer_id = customer.get("customer_id")
    order_id = order.get("order_id")
    order_items = order.get("items", {})  # expected format: {item_name: [quantity, price], ...}
    # accept either "total" or "order_total" keys
    order_total = order.get("total")

    cust_doc = Customer.objects(customer_id=customer_id).first()

    order_doc = None

    if order_id:
        # try to update existing order
        order_doc = OrderDoc.objects(order_id=order_id).first()
        if order_doc:
            # update total if provided
            if order_total is not None:
                order_doc.total = safe_int(order_total)
            order_doc.save()

            # update/create order items
            for item_name, vals in (order_items or {}).items():
                # expect vals like [quantity, price]
                qty = vals[0] if len(vals) > 0 else None
                price = vals[1] if len(vals) > 1 else None
                qty = safe_int(qty)
                price = safe_int(price)

                existing_item = OrderItem.objects(order=order_doc, item=item_name).first()
                if existing_item:
                    if qty is not None:
                        existing_item.quantity = qty
                    if price is not None:
                        existing_item.price = price
                    existing_item.save()
                else:
                    # create new item (require quantity, but keep simple)
                    OrderItem(order=order_doc, item=item_name,
                              quantity=(qty if qty is not None else 1),
                              price=price).save()


    if not order_id or not order_doc:
        # create new order
        order_doc = OrderDoc(customer=cust_doc,
                              status="incomplete",
                              total=safe_int(order_total)).save()
        # retrieve saved doc
        order_doc = OrderDoc.objects(id=order_doc.id).first()
        for item_name, vals in (order_items or {}).items():
            qty = vals[0] if len(vals) > 0 else None
            price = vals[1] if len(vals) > 1 else None
            qty = safe_int(qty)
            price = safe_int(price)
            OrderItem(order=order_doc, item=item_name,
                      quantity=(qty if qty is not None else 1),
                      price=price).save()

        order_id = order_doc.order_id
        order["order_id"] = order_id

    print("Order added to db")
    print("ROUTING TO: order_repeater")

    return {
        "processed_queries": state.get("processed_queries", []) + [current_query],
        "query_responses": state.get("query_responses", []) + [str("Your order has been noted.")],
        "next": "order_repeater",
        "customer": customer,
        "order": order
    }


def order_checker_node(state: MyState) -> Dict:
    """Handles order item checking - validates and categorizes items to add/remove."""
    print("\n" + "="*80)
    print("🔍 ORDER CHECKER NODE")
    print("="*80)

    current_query = state["input"]
    print(f"PROCESSING: {current_query}")

    conversation_history = get_conversation_context(state["messages"])

    order = state['order']

    # Get menu context (assuming you have this available)
    menu_context = get_context(current_query)  # Replace with your actual menu retrieval

    order_checker = OrderItemChecker()
    response = order_checker.execute_query(
        query=current_query,
        context=menu_context,
        order= order,
        conversation_history=conversation_history
    )

    remove = response.remove
    add = response.add
    # ensure items dict exists
    order_items = order.setdefault("items", {})

    # handle removals: decrease qty, delete if <= 0
    if remove:
        for item_name, vals in remove.items():
            rem_qty = safe_float(vals[0]) if vals else None
            if rem_qty is None:
                continue
            if item_name in order_items:
                existing_vals = order_items[item_name]
                existing_qty = safe_float(existing_vals[0]) or 0.0
                existing_price = safe_float(existing_vals[1]) or 0.0
                new_qty = existing_qty - rem_qty
                if new_qty <= 0:
                    del order_items[item_name]
                else:
                    order_items[item_name] = [new_qty, existing_price]

    # handle additions: increase qty or add new item
    if add:
        for item_name, vals in add.items():
            add_qty = safe_float(vals[0]) if vals else None
            add_price = safe_float(vals[1]) if vals and len(vals) > 1 else None
            if add_qty is None:
                continue
            if item_name in order_items:
                existing_vals = order_items[item_name]
                existing_qty = safe_float(existing_vals[0]) or 0.0
                existing_price = safe_float(existing_vals[1]) or 0.0
                new_qty = existing_qty + add_qty
                price_to_set = add_price if add_price is not None else existing_price
                order_items[item_name] = [new_qty, price_to_set]
            else:
                price_to_set = add_price if add_price is not None else 0.0
                order_items[item_name] = [add_qty, price_to_set]

    # recalculate total
    total = 0.0
    for qty_price in order_items.values():
        qty = safe_float(qty_price[0]) or 0.0
        price = safe_float(qty_price[1]) or 0.0
        total += qty * price

    # store total (as int if whole number, else float)
    order["total"] = int(total) if float(total).is_integer() else total

    print(f"RESPONSE: {response}\n")
    print("ROUTING BACK TO: order_supervisor")

    return {
        "processed_queries": state["processed_queries"],
        "query_responses": state["query_responses"],
        "order": order,
        "next": "order_repeater"
    }

def order_cancel(state: MyState) -> Dict:
    """Handles order cancellation - cancels current order."""
    print("\n" + "=" * 80)
    print("📋 ORDER CANCEL NODE")
    print("=" * 80)

    order = state.get("order", {}) or {}
    current_query = state.get("input")
    print(f"PROCESSING: {current_query}")

    # try to get order_id from state.order first, then from top-level state
    order_id = order.get("order_id") if isinstance(order, dict) else None
    if not order_id:
        response = "No order_id provided; nothing to delete."
    else:
        # find the order doc and delete order items + order
        order_doc = OrderDoc.objects(order_id=order_id).first()
        if order_doc:
            # delete order items that belong to this order
            OrderItem.objects(order=order_doc).delete()
            OrderDoc.objects(order_id=order_id).delete()
            # delete the order itself
            response = f"Deleted order and its items for order_id={order_id}"

        else:
            # if order_id provided but no DB row found, just clear local order
            response = f"No DB order found for order_id={order_id}; local order cleared."

    order = {}

    print(response)
    print("ROUTING BACK TO: order_repeater")

    return {
        "processed_queries": state.get("processed_queries", []) + [current_query],
        "query_responses": state.get("query_responses", []) + [response],
        "next": "supervisor",
        "order": order
    }


def order_repeater_node(state: MyState) -> Dict:
    """Handles order repetition - summarizes current order."""
    print("\n" + "="*80)
    print("📋 ORDER REPEATER NODE")
    print("="*80)

    order = state.get("order")

    current_query = state["input"]
    print(f"PROCESSING: {current_query}")

    order_repeater = OrderRepeater()
    response = order_repeater.execute_query(
        "", [], order=order
    )

    print(f"RESPONSE: {response}\n")
    print("ROUTING BACK TO: order_supervisor")

    return {
        "processed_queries": state["processed_queries"] + [current_query],
        "query_responses": state["query_responses"] + [str(response)],
        "next": "supervisor"
    }

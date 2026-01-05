from state import MyState
from prompt_templates import ComplaintClassifier, UpdateComplain
from models import Customer, Complaint
from typing import Dict
from utils import get_conversation_context

def complaint_customer_check_node(state: MyState) -> Dict:
    print("\n" + "="*80)
    print("👤 COMPLAINT CUSTOMER CHECK")
    print("="*80)

    customer = state.get("customer", {})
    customer_id = customer.get("customer_id")

    if customer_id:
        return {"next": "complaint_classifier"}
    else:
        return {"next": "get_customer"}  # reuse your existing node

def complaint_classifier_node(state: MyState) -> Dict:
    print("\n" + "="*80)
    print("🧠 COMPLAINT CLASSIFIER")
    print("="*80)

    classifier = ComplaintClassifier()
    result = classifier.execute_query(
        query=state["input"],
        conversation_history=get_conversation_context(state["messages"])
    )

    if result.classification == "Finish":
        return {"next": "complaint_save"}
    else:
        return {"next": "complaint_update"}

def complaint_update_node(state: MyState) -> Dict:
    print("\n" + "="*80)
    print("✍️ COMPLAINT UPDATE")
    print("="*80)

    updater = UpdateComplain()
    result = updater.execute_query(
        current_complain=state.get("complaint", ""),
        query=state["input"]
    )

    return {
        "complaint": result.complain,
        "processed_queries": state["processed_queries"] + [state["input"]],
        "query_responses": state["query_responses"] + [result.response],
        "next": "supervisor"
    }

def complaint_save_node(state: MyState) -> Dict:
    print("\n" + "="*80)
    print("💾 SAVING COMPLAINT")
    print("="*80)

    customer_id = state["customer"]["customer_id"]
    complaint_text = state["complaint"]

    cust = Customer.objects(customer_id=customer_id).first()

    Complaint(
        customer=cust,
        complaint_text=complaint_text
    ).save()

    return {
        "processed_queries": state["processed_queries"] + [state["input"]],
        "query_responses": state["query_responses"] + [
            "✅ Your complaint has been recorded. Our team will get back to you soon."
        ],
        "complaint": "",
        "next": "supervisor"
    }
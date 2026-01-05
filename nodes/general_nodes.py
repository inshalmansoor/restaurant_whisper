from state import MyState
from prompt_templates import Refine_Query, MultiIntentDetector, Information_Retrieval
from knowledge_base import retriever
from utils import get_conversation_context
from typing import Dict


# NODE DEFINITIONS
def refine_node(state: MyState) -> Dict:
    """Refines the user query for better understanding."""
    print("\n" + "="*80)
    print("🔧 REFINE NODE")
    print("="*80)

    user_input = state["input"]
    print(f"INPUT: {user_input}")

    conversation_history = get_conversation_context(state["messages"])

    refiner = Refine_Query()

    refined = refiner.execute_query(
        query=user_input,
        conversation_history=conversation_history
    )

    print(f"OUTPUT: {refined}\n")

    return {"input": refined, "next": "intent_detection"}


def intent_detection_node(state: MyState) -> Dict:
    """Breaks down complex queries into individual tasks."""
    print("\n" + "="*80)
    print("🎯 INTENT DETECTION NODE")
    print("="*80)

    refined_query = state["input"]


    print(f"INPUT: {refined_query}")

    conversation_history = get_conversation_context(state["messages"])

    intent_detector = MultiIntentDetector()

    result = intent_detector.execute_query(
        query=refined_query,
        conversation_history=conversation_history
    )

    query_list = result.queries if result.queries else [refined_query]

    print(f"OUTPUT: {len(query_list)} queries detected")
    for i, q in enumerate(query_list, 1):
        print(f"  {i}. {q}")
    print()

    return {
        "query_list": query_list,
        "processed_queries": [],
        "query_responses": [],
        "next": "supervisor"
    }

def information_node(state: MyState) -> Dict:
    """Handles information-related queries."""
    print("\n" + "="*80)
    print("ℹ️  INFORMATION NODE")
    print("="*80)

    current_query = state["input"]
    print(f"PROCESSING: {current_query}")

    conversation_history = get_conversation_context(state["messages"])

    information_processor = Information_Retrieval()

    context = get_context(current_query)

    response = information_processor.execute_query(
        query=current_query,
        conversation_history=conversation_history,
        context=context
    )

    print(f"RESPONSE: {response}\n")

    return {
        "processed_queries": state["processed_queries"] + [current_query],
        "query_responses": state["query_responses"] + [str(response)],
        "next": "supervisor"
    }

def get_context(current_query):
    docs = retriever.invoke(current_query)
    context = "\n".join(doc.page_content for doc in docs)
    return context

def customer_details_node(state: MyState) -> Dict:
    """Checks whether customer details, booking details or order details exist or not?"""
    print("\n" + "="*80)
    print("🔍 CUSOTMER DETAILS CHECK NODE")
    print("="*80)

    customer = state.get("customer", None)
    customer_id = customer.get("customer_id")

    if customer_id:
        return {
            "next": state.get("next")
        }
    else:
        return {
            "next": "get_customer"
        }
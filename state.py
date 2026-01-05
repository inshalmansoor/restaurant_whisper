from typing import Dict, List, TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage

class MyState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    input: str
    query_list: List[str]
    processed_queries: List[str]
    query_responses: List[str]
    order: Dict
    customer: Dict
    booking: Dict
    next: str
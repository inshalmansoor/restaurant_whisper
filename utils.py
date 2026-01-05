from typing import List
# utils.py (append these helpers or create if not present)
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from typing import Dict, Any
from collections.abc import Mapping, Sequence, Set

def _is_primitive(o: Any) -> bool:
    return o is None or isinstance(o, (str, int, float, bool))

def sanitize_state(obj: Any, max_depth: int = 6, _depth: int = 0, _seen: set | None = None) -> Any:
    """
    Convert `obj` into a JSON-safe structure:
      - primitives unchanged
      - dict -> sanitized dict (keys cast to str)
      - list/tuple/set -> list of sanitized items
      - unknown objects -> replaced by "<Unserializable: ClassName>"
    Avoids infinite recursion via _seen set.
    """
    if _seen is None:
        _seen = set()

    try:
        obj_id = id(obj)
    except Exception:
        obj_id = None

    if obj_id is not None:
        if obj_id in _seen:
            return "<circular>"
        _seen.add(obj_id)

    if _depth >= max_depth:
        return f"<max_depth_{type(obj).__name__}>"

    # primitives
    if _is_primitive(obj):
        return obj

    # mappings/dicts
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            try:
                key = str(k)
            except Exception:
                key = "<unhashable-key>"
            out[key] = sanitize_state(v, max_depth, _depth + 1, _seen)
        return out

    # sequences (but not str)
    if isinstance(obj, (list, tuple, Set)):
        return [sanitize_state(i, max_depth, _depth + 1, _seen) for i in obj]

    # fallback for other iterables (generator, etc.)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [sanitize_state(i, max_depth, _depth + 1, _seen) for i in obj]

    # everything else -> replace with class name (no heavy repr)
    return f"<Unserializable: {type(obj).__name__}>"


def create_initial_state() -> Dict[str, Any]:
    """
    Create the initial state object the graph expects.
    Adjust keys/structure to match your graph / MyState type.
    """
    initial_state = {
        "messages": [],
        "query_list": [],
        "processed_queries": [],
        "query_responses": [],
        "order": {},
        "customer": {},
        "booking": {},
        "next": "refine",
    }
    return initial_state


def get_conversation_context(messages: List[BaseMessage], exclude_last: bool = True) -> str:
    """Extract relevant conversation history."""
    filtered = [m for m in messages if hasattr(m, 'type') and m.type in ['human', 'ai']]

    if exclude_last and filtered and hasattr(filtered[-1], 'type') and filtered[-1].type == 'human':
        filtered = filtered[:-1]

    return "\n".join(
        f"{m.type.capitalize()}: {m.content}" for m in filtered
    ) if filtered else "No previous conversation."

def safe_float(v):
    try:
        return float(v) if v is not None else None
    except Exception:
        return None
    
def safe_int(v):
    try:
        return int(v) if v is not None else None
    except Exception:
        return None
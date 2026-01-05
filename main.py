# main.py
import uuid
import asyncio
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from state import MyState
from knowledge_base import setup_retriever, load_documents
from get_models import prepare_and_load_whisper
from langchain_core.messages import HumanMessage, SystemMessage
from graph_builder import compiled
from pathlib import Path
# server_tts_ws.py (FastAPI)
import asyncio
from io import BytesIO
# add imports near the top of main.py
import json

import tempfile
import traceback
import whisper
from gtts import gTTS

from utils import create_initial_state, sanitize_state
from fastapi.middleware.cors import CORSMiddleware

# if you only want to allow your local frontend during development:
origins = ['*']

app = FastAPI(title="restaurant_whisper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # use ["*"] only for quick dev; prefer explicit origins
    allow_credentials=True,       # if you send cookies / auth from the browser
    allow_methods=["*"],          # GET, POST, OPTIONS, etc.
    allow_headers=["*"],          # allow custom headers
)

# In-memory session store (session_id -> state dict)
# For production replace with Redis or another persistent store
_sessions: Dict[str, Dict[str, Any]] = {}
_session_lock = asyncio.Lock()  # protects the _sessions dict during create/delete/update

# Expose heavy objects via app.state so they're globally available
# They will be set in startup event
# app.state.docs
# app.state.retriever
# app.state.compiled


class CreateSessionResponse(BaseModel):
    session_id: str


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    result: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """
    Load everything once when the server starts:
      - documents
      - retriever
      - compiled graph (already imported above as compiled)
    This keeps model/KB warm for all incoming requests.
    """
    # Load documents and retriever (blocking I/O — run in thread)
    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(None, load_documents, "knowledge_base")
    retriever = await loop.run_in_executor(None, setup_retriever, docs)

    whisper_model = await loop.run_in_executor(
        None,
        prepare_and_load_whisper,
        "large-v3",                  # model_name
        Path("models/whisper"),   # target_root
        False                     # force (set True to force re-download)
    )
    app.state.whisper_model = whisper_model
    print("[i] Whisper model available on app.state.whisper_model")

    # Attach to app state
    app.state.docs = docs
    app.state.retriever = retriever
    app.state.compiled = compiled  # the graph object you already have

    # Optional: create a default session so app is "warm"
    default_state = create_initial_state()
    default_state["retriever"] = retriever
    async with _session_lock:
        default_sid = str(uuid.uuid4())
        _sessions[default_sid] = default_state
    app.state.default_session_id = default_sid
    app.logger = app.logger if hasattr(app, "logger") else None
    print(f"Startup complete — preloaded docs/retriever. default_session: {default_sid}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/sessions", response_model=CreateSessionResponse)
async def create_session():
    sid = str(uuid.uuid4())
    state = create_initial_state()
    # Keep retriever in app.state only (do not store it into the session dict permanently)
    # If you need the retriever at runtime, attach it transiently when invoking the graph.

    async with _session_lock:
        _sessions[sid] = state
        return CreateSessionResponse(session_id=sid)
    return CreateSessionResponse(session_id=sid)


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    async with _session_lock:
        state = _sessions.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")
    return sanitize_state(state)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    async with _session_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            return {"deleted": True}
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/sessions/{session_id}/query", response_model=QueryResponse)
async def run_query(session_id: str, payload: QueryRequest):
    """
    Runs the graph/workflow for a session with the provided query.
    The compiled.invoke call is run in a threadpool to avoid blocking the event loop.
    After run, session state is updated with the result.
    """
    async with _session_lock:
        state = _sessions.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")

    # update the session state with the new input query
    state["input"] = payload.query

    # ensure retriever reference is present for the graph/workflow (some graphs expect it)
    state.setdefault("retriever", app.state.retriever)

    # run compiled.invoke in threadpool (in case it's CPU / blocking IO)
    loop = asyncio.get_running_loop()
    compiled_obj = app.state.compiled

    try:
        result = await loop.run_in_executor(None, compiled_obj.invoke, state)
    except Exception as e:
        # bubble up something useful
        raise HTTPException(status_code=500, detail=f"Graph invocation error: {e}")

    # Update session state atomically
    async with _session_lock:
        _sessions[session_id] = result

    return QueryResponse(result=result)



@app.get("/sessions/{session_id}/current")
async def get_current_session_data(session_id: str) -> Dict[str, Any]:
    """
    Return a small summary of the current session containing:
      {
        "order": <order details from state or {}>,
        "customer": <customer details from state or {}>,
        "bookings": <booking details from one of possible keys or []>
      }
    This endpoint is tolerant of different key names for bookings.
    """
    async with _session_lock:
        state = _sessions.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")

    # pull known keys or sensible defaults
    order = state.get("order", {})         # preserves whatever shape you store order in
    customer = state.get("customer", {})   # preserves whatever shape you store customer in

    # booking data might be stored under different keys depending on your graph
    bookings = (
        state.get("bookings")
        or state.get("booking")
        or state.get("reservations")
        or []
    )

    return {"order": order, "customer": customer, "bookings": bookings}

# synthesize helper (in-memory MP3)
def synthesize_mp3_bytes(text: str, lang: str = "en") -> bytes:
    fp = BytesIO()
    tts = gTTS(text or " ", lang=lang)
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

def extract_final_text(result: Any) -> str:
    """
    Try to find a reasonable reply string inside result/state.
    This inspects a few common keys and list structures, and falls back to str(result).
    """
    if not result:
        return ""
    if isinstance(result, dict):
        # common single-key names
        for k in ("final_response", "response", "answer", "output", "text", "result"):
            v = result.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # lists of responses
        for key in ("query_responses", "responses", "answers", "outputs"):
            lst = result.get(key)
            if isinstance(lst, list) and len(lst) > 0:
                last = lst[-1]
                if isinstance(last, str) and last.strip():
                    return last.strip()
                if isinstance(last, dict):
                    for subk in ("text", "response", "answer", "content"):
                        sv = last.get(subk)
                        if isinstance(sv, str) and sv.strip():
                            return sv.strip()
        # messages list (langchain-like)
        msgs = result.get("messages")
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if isinstance(last, dict):
                cont = last.get("content")
                if isinstance(cont, str) and cont.strip():
                    return cont.strip()
    # fallback to string representation
    try:
        s = str(result)
        return s[:2000]  # don't return insanely large strings
    except Exception:
        return ""

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """
    Integrated websocket:
     - receives binary audio frames into a buffer between "start" and "end" events
     - on "end": transcribe with Whisper, put transcript into session state['input'],
       run compiled.invoke in executor (same as run_query), update session, synthesize final
       reply to MP3 via gTTS and send binary MP3 back to client.
    """
    await websocket.accept()
    buffer = bytearray()
    try:
        while True:
            msg = await websocket.receive()

            # binary frame (append to buffer)
            if msg.get("bytes") is not None:
                buffer.extend(msg["bytes"])
                continue

            # text frame (JSON control)
            text = msg.get("text")
            if text is None:
                continue

            try:
                data = json.loads(text)
            except Exception:
                # ignore non-JSON text messages
                continue

            evt = data.get("event")
            if evt == "start":
                buffer = bytearray()
                await websocket.send_json({"event": "ready_to_receive"})
                continue

            if evt == "end":
                # client finished sending; begin processing pipeline
                await websocket.send_json({"event": "processing"})
                loop = asyncio.get_running_loop()

                # write received audio to a temporary file
                temp_path = None
                try:
                    if buffer:
                        tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
                        tmp.write(bytes(buffer))
                        tmp.flush()
                        tmp.close()
                        temp_path = tmp.name
                    else:
                        temp_path = None

                    # 1) Transcribe with Whisper (off the event loop)
                    transcript = ""
                    try:
                        whisper_model = getattr(app.state, "whisper_model", None)
                        if whisper_model is None:
                            # load a default whisper model in executor if not present
                            def _load_default():
                                return whisper.load_model("base")
                            whisper_model = await loop.run_in_executor(None, _load_default)
                            app.state.whisper_model = whisper_model

                        if temp_path:
                            # transcribe file path (blocking) inside executor
                            def _transcribe(path, model):
                                # model.transcribe accepts a filename
                                return model.transcribe(path)
                            trans_res = await loop.run_in_executor(None, _transcribe, temp_path, whisper_model)
                            transcript = trans_res.get("text", "").strip() if isinstance(trans_res, dict) else ""
                        else:
                            transcript = ""
                    except Exception as e:
                        # transcription failed; notify client and continue pipeline with empty transcript
                        await websocket.send_json({"event": "error", "message": f"transcription failed: {e}"})
                        transcript = ""

                    # 2) Insert transcript into session state and run compiled.invoke (same as run_query)
                    session_id = data.get("session_id") or getattr(app.state, "default_session_id", None)
                    if not session_id:
                        await websocket.send_json({"event": "error", "message": "no session_id provided and no default available"})
                        continue

                    async with _session_lock:
                        state = _sessions.get(session_id)
                        if state is None:
                            await websocket.send_json({"event": "error", "message": "session not found"})
                            continue

                    # update state with transcript
                    state["input"] = transcript
                    state.setdefault("retriever", app.state.retriever)

                    compiled_obj = app.state.compiled

                    try:
                        result = await loop.run_in_executor(None, compiled_obj.invoke, state)
                    except Exception as e:
                        await websocket.send_json({"event": "error", "message": f"Graph invocation error: {e}"})
                        continue

                    # save updated session state
                    async with _session_lock:
                        _sessions[session_id] = result

                    # 3) Determine final textual response
                    final_text = extract_final_text(result)
                    if not final_text:
                        # if none found, try some fallbacks
                        final_text = extract_final_text(state) or "Sorry, I have no response."

                    # 4) Synthesize final_text to MP3 bytes (off the event loop)
                    try:
                        audio_bytes = await loop.run_in_executor(None, synthesize_mp3_bytes, final_text, "en")
                    except Exception as e:
                        await websocket.send_json({"event": "error", "message": f"TTS failed: {e}"})
                        continue

                    # 5) Send speaking event + binary audio, then done
                    await websocket.send_json({"event": "speaking"})
                    await websocket.send_bytes(audio_bytes)
                    await websocket.send_json({"event": "done", "text": final_text})

                finally:
                    # cleanup temp file if any
                    if temp_path:
                        try:
                            os.unlink(temp_path)
                        except Exception:
                            pass

            elif evt == "stop":
                # client asks to close
                try:
                    await websocket.close()
                except Exception:
                    pass
                break

    except WebSocketDisconnect:
        # client disconnected
        return
    except Exception as e:
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
            await websocket.close()
        except Exception:
            pass
        return

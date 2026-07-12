"""Free, API-key-free realtime voice interface for MOSAIC-CRS.

Transport and inference stack:
- Browser microphone -> 16 kHz PCM over WebSocket
- Browser energy VAD for endpointing and immediate barge-in
- faster-whisper for local speech-to-text
- LangChain StructuredTool -> local MOSAIC-CRS checkpoint
- Optional local Ollama model for natural wording
- Browser SpeechSynthesis for text-to-speech

No paid cloud service and no API key are required.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from voice_ai.local_stt import LocalWhisperSTT
from voice_ai.mosaic_adapter import MosaicRecommendationAdapter
from voice_ai.realtime_langchain import MosaicRealtimeToolChain


load_dotenv()


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)


@dataclass
class SocketConversation:
    session_id: str
    history: List[Dict[str, str]] = field(default_factory=list)
    audio: bytearray = field(default_factory=bytearray)
    capturing: bool = False
    response_generation: int = 0
    response_task: Optional[asyncio.Task[Any]] = None
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def _safe_send(websocket: WebSocket, state: SocketConversation, payload: Dict[str, Any]) -> None:
    async with state.send_lock:
        await websocket.send_json(payload)


def build_app(args: argparse.Namespace) -> FastAPI:
    root = Path(__file__).resolve().parent
    adapter = MosaicRecommendationAdapter(
        project_root=root,
        config_path=args.config,
        checkpoint=args.checkpoint,
        dataset="ReDial",
        train_data_path=args.train_data,
        catalog_path=args.catalog,
        auto_build_catalog=not args.no_auto_catalog,
        candidate_pool_size=args.candidate_pool_size,
        top_k=args.top_k,
    )
    stt = LocalWhisperSTT(
        model_name=args.stt_model,
        device=args.stt_device,
        compute_type=args.stt_compute_type,
        language=args.stt_language,
    )
    tool_chain = MosaicRealtimeToolChain(
        adapter=adapter,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        use_ollama=not args.no_ollama,
    )

    app = FastAPI(title="MOSAIC-CRS Local Realtime Voice", version="3.0")
    app.state.adapter = adapter
    app.state.stt = stt
    app.state.tool_chain = tool_chain
    app.state.args = args

    @app.get("/api/status")
    async def status() -> Dict[str, Any]:
        return {
            "service": "MOSAIC-CRS Local Realtime Voice",
            "api_key_required": False,
            "transport": "WebSocket PCM16 at 16 kHz",
            "voice_activity_detection": "browser energy VAD",
            "text_to_speech": "browser SpeechSynthesis",
            "speech_to_text": stt.status(),
            "language_model": tool_chain.status(),
            "adapter": adapter.status(),
        }

    @app.post("/api/warmup")
    async def warmup() -> Dict[str, Any]:
        try:
            await asyncio.gather(
                asyncio.to_thread(adapter.ensure_loaded),
                asyncio.to_thread(stt.ensure_loaded),
            )
            return {
                "ok": True,
                "adapter": adapter.status(),
                "speech_to_text": stt.status(),
                "language_model": tool_chain.status(),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/reset")
    async def reset(request: ResetRequest) -> Dict[str, bool]:
        adapter.reset_session(request.session_id)
        return {"ok": True}

    @app.websocket("/ws/call/{session_id}")
    async def call_socket(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        state = SocketConversation(session_id=session_id)
        max_audio_bytes = int(args.max_utterance_seconds * 16000 * 2)

        async def cancel_response(notify: bool = True) -> None:
            state.response_generation += 1
            task = state.response_task
            state.response_task = None
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            if notify:
                with contextlib.suppress(Exception):
                    await _safe_send(
                        websocket,
                        state,
                        {"type": "assistant_interrupted", "generation": state.response_generation},
                    )

        async def generate_response(query: str, source: str, generation: int) -> None:
            try:
                if generation != state.response_generation:
                    return
                await _safe_send(
                    websocket,
                    state,
                    {"type": "assistant_start", "source": source, "generation": generation},
                )
                chunks: List[str] = []
                async for delta in tool_chain.astream_response(
                    query=query,
                    session_id=state.session_id,
                    history=state.history,
                ):
                    if generation != state.response_generation:
                        return
                    chunks.append(delta)
                    await _safe_send(
                        websocket,
                        state,
                        {"type": "assistant_delta", "delta": delta, "generation": generation},
                    )

                if generation != state.response_generation:
                    return
                answer = "".join(chunks).strip()
                if not answer:
                    answer = "I could not generate a response. Please try saying that again."
                state.history.append({"role": "user", "content": query})
                state.history.append({"role": "assistant", "content": answer})
                state.history = state.history[-16:]
                adapter.record_assistant(state.session_id, answer)
                await _safe_send(
                    websocket,
                    state,
                    {"type": "assistant_done", "text": answer, "generation": generation},
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if generation == state.response_generation:
                    await _safe_send(
                        websocket,
                        state,
                        {"type": "error", "message": str(exc), "generation": generation},
                    )

        async def handle_text(query: str, source: str) -> None:
            query = " ".join(str(query or "").split()).strip()
            if not query:
                return
            await cancel_response(notify=False)
            generation = state.response_generation
            await _safe_send(
                websocket,
                state,
                {"type": "thinking", "source": source, "generation": generation},
            )
            state.response_task = asyncio.create_task(generate_response(query, source, generation))

        async def transcribe_and_respond(audio_bytes: bytes) -> None:
            await cancel_response(notify=False)
            generation = state.response_generation
            try:
                await _safe_send(
                    websocket,
                    state,
                    {"type": "transcribing", "generation": generation},
                )
                transcript = await asyncio.to_thread(stt.transcribe_pcm16, audio_bytes, 16000)
                if generation != state.response_generation:
                    return
                if not transcript:
                    await _safe_send(
                        websocket,
                        state,
                        {"type": "empty_transcript", "generation": generation},
                    )
                    return
                await _safe_send(
                    websocket,
                    state,
                    {"type": "user_transcript", "text": transcript, "generation": generation},
                )
                await _safe_send(
                    websocket,
                    state,
                    {"type": "thinking", "source": "voice", "generation": generation},
                )
                state.response_task = asyncio.create_task(
                    generate_response(transcript, "voice", generation)
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if generation == state.response_generation:
                    await _safe_send(
                        websocket,
                        state,
                        {"type": "error", "message": str(exc), "generation": generation},
                    )

        await _safe_send(
            websocket,
            state,
            {
                "type": "ready",
                "session_id": session_id,
                "sample_rate": 16000,
                "api_key_required": False,
            },
        )

        try:
            while True:
                message = await websocket.receive()
                if message.get("bytes") is not None:
                    if state.capturing:
                        state.audio.extend(message["bytes"])
                        if len(state.audio) > max_audio_bytes:
                            state.audio = state.audio[-max_audio_bytes:]
                    continue

                raw_text = message.get("text")
                if raw_text is None:
                    continue
                try:
                    event = json.loads(raw_text)
                except json.JSONDecodeError:
                    await _safe_send(
                        websocket,
                        state,
                        {"type": "error", "message": "Invalid WebSocket event."},
                    )
                    continue

                event_type = event.get("type")
                if event_type == "speech_start":
                    await cancel_response(notify=True)
                    state.audio.clear()
                    state.capturing = True
                    await _safe_send(websocket, state, {"type": "listening"})
                elif event_type == "speech_end":
                    if not state.capturing:
                        continue
                    state.capturing = False
                    utterance = bytes(state.audio)
                    state.audio.clear()
                    if utterance:
                        state.response_task = asyncio.create_task(transcribe_and_respond(utterance))
                elif event_type == "typed_message":
                    state.capturing = False
                    state.audio.clear()
                    query = str(event.get("text", ""))[:4000]
                    await handle_text(query, "typed")
                elif event_type == "interrupt":
                    state.capturing = False
                    state.audio.clear()
                    await cancel_response(notify=True)
                elif event_type == "reset":
                    state.capturing = False
                    state.audio.clear()
                    await cancel_response(notify=False)
                    state.history.clear()
                    adapter.reset_session(state.session_id)
                    await _safe_send(websocket, state, {"type": "reset_done"})
                elif event_type == "ping":
                    await _safe_send(websocket, state, {"type": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            with contextlib.suppress(Exception):
                await cancel_response(notify=False)

    static_dir = root / "static"

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the free local MOSAIC-CRS realtime voice interface"
    )
    parser.add_argument("--config", default="config_redial.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/ReDial/best_rl_model.pt")
    parser.add_argument("--train-data", default="data/ReDial/train_data.json")
    parser.add_argument(
        "--catalog",
        default="data/ReDial/item_catalog.json",
        help="Original ReDial item catalog. Recommended for readable titles.",
    )
    parser.add_argument("--no-auto-catalog", action="store_true")
    parser.add_argument("--candidate-pool-size", type=int, default=180)
    parser.add_argument("--top-k", type=int, default=5)

    parser.add_argument("--stt-model", default=os.getenv("STT_MODEL", "base.en"))
    parser.add_argument("--stt-device", default=os.getenv("STT_DEVICE", "cpu"))
    parser.add_argument("--stt-compute-type", default=os.getenv("STT_COMPUTE_TYPE", "int8"))
    parser.add_argument("--stt-language", default=os.getenv("STT_LANGUAGE", "en"))

    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    parser.add_argument(
        "--ollama-base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--no-ollama", action="store_true")
    parser.add_argument("--max-utterance-seconds", type=float, default=35.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    application = build_app(arguments)
    uvicorn.run(application, host=arguments.host, port=arguments.port, reload=arguments.reload)

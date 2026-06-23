import io
import os
import wave
import base64
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from langgraph_sdk import get_client
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import json
from openai import OpenAI
import numpy as np
import sounddevice as sd
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from langsmith.middleware import TracingMiddleware
from sst_stream import record_until_silence
from websockets.asyncio.client import connect


load_dotenv()

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None  # works on Python 3.8+

app = FastAPI()
app.add_middleware(TracingMiddleware)
# init client
client = get_client(url=os.getenv("LANGGRAPH_URL", "http://localhost:2024"))
openai_client = OpenAI()
# init thread pool
executor = ThreadPoolExecutor(max_workers=2)
logger = logging.getLogger("uvicorn")

@app.post("/chat")
async def chat(req: ChatRequest):
    # reuse thread if provided, else create a new one
    if req.thread_id:
        thread_id = req.thread_id
    else:
        thread = await client.threads.create()
        thread_id = thread["thread_id"]

    final_response = None

    async for chunk in client.runs.stream(
        thread_id,
        "assistant",
        input={"messages": [{"role": "human", "content": req.message}]},
        stream_mode="values",
    ):
        if chunk.data and "messages" in chunk.data:
            final_response = chunk.data["messages"][-1]

    return {
        "thread_id": thread_id,   # return so client can continue the conversation
        "reply": final_response["content"] if final_response else None,
    }

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if req.thread_id:
        thread_id = req.thread_id
    else:
        thread = await client.threads.create()
        thread_id = thread["thread_id"]

    async def generate():
        # yield thread_id first so client has it immediately
        yield f"data: {json.dumps({'thread_id': thread_id})}\n\n"
        yield f"data: {json.dumps({'initial question': req.message})}\n\n"

        async for chunk in client.runs.stream(
            thread_id,
            "calculator",
            input={"messages": [{"role": "human", "content": req.message}]},
            stream_mode="messages",   # "messages" gives token-by-token, "values" gives full state
        ):
            if chunk.data and chunk.event == "messages/partial":
                yield f"data: {json.dumps({'type':chunk.event, 'token': chunk.data})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/chat/dummyaudio")
async def chat_tts():
    def generate_tone(frequency=440, duration=1.0, samplerate=22050):
        t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
        samples = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        return samples, samplerate


    samples, samplerate = generate_tone(440, 1.0, 22050)
    sd.play(samples, samplerate)
    sd.wait() #block until finished

    return {"status": "played"}


recording_sessions = {}

def to_wav_bytes(audio, samplerate):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    buffer.seek(0)
    buffer.name = "audio.wav"
    return buffer

@app.post("/chat/transcribe")
async def transcribe():
    def generate_speech():
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=final_response,
            response_format="wav"
        )
        return response.content

    def play_on_server():
        import sounddevice as sd
        import numpy as np
        import io, wave

        with wave.open(io.BytesIO(audio_bytes)) as wf:
            samplerate = wf.getframerate()
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            samples = samples.astype(np.float32) / 2**15

        sd.play(samples, samplerate)
        sd.wait()
        print("played sound")

    # one thread per user session for the langgraph app
    thread = await client.threads.create(if_exists="do_nothing")
    thread_id = thread["thread_id"]

    loop = asyncio.get_event_loop()

    # 1. recording
    samples, samplerate = await loop.run_in_executor(executor, record_until_silence)
    wav_buffer = to_wav_bytes(samples, samplerate)
    logger.info(f"samplerate: {samplerate}")

    # 2. transcribe
    transcript = openai_client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=("audio.wav", wav_buffer, "audio/wav"),
        response_format="text"
    )

    # 3. generate response
    final_response = None
    async for chunk in client.runs.stream(
        thread_id,
        "assistant",
        input={"messages": [{"role": "human", "content": transcript}]},
        stream_mode="values",
    ):
        if chunk.data and "messages" in chunk.data:
            final_response = chunk.data["messages"][-1]

    final_response = final_response["content"]

    print(f"final response: {final_response}, {type(final_response)}")
    audio_bytes = await loop.run_in_executor(executor, generate_speech)

    # 4. play on server
    await loop.run_in_executor(executor, play_on_server)
    return {"status": "task completed", "content": transcript}

@app.websocket("/chat/ws/text")
async def transcribe_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text  = await websocket.receive_text()
            await asyncio.sleep(10) # simulate the agent processing
            await websocket.send_text(f"Completed processing of {text}")
    except Exception as e:
        print(e)

def generate_dummy_audio():
    file_path = "assets/speech_detection.wav"
    # 1. Open the file in Read Binary mode
    with open(file_path, "rb") as file:
        # 2. Read the content into a bytes object
        wav_bytes = file.read()
    return wav_bytes

def pcm16_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = 48_000,
) -> bytes:
    output = io.BytesIO()

    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # PCM16 = 2 bytes
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)

    return output.getvalue()

def normalize(utterance: np.ndarray) -> np.ndarray:
    float_audio = utterance.astype(np.float32)
    max_val = np.max(np.abs(float_audio))
    if max_val == 0:
        return utterance
    normalized = float_audio / max_val * 0.9     # scale to 90% of max int16
    return normalized.astype(np.int16)

SAMPLE_RATE    = 16000
CHANNELS       = 1
CHUNK          = 1024
DTYPE          = np.int16

SILENCE_THRESHOLD = 0.01   # RMS amplitude (0.0–1.0 normalized)
SILENCE_FRAMES    = 20     # consecutive silent chunks before flush
                           # 20 * (1024/16000) ≈ 1.28 seconds of silence

PRE_ROLL_FRAMES = 3        # chunks to keep before speech starts
                           # avoids clipping the first syllable
MIN_DURATION_SECONDS = 0.5

@app.websocket("/ws/audio")
async def transcribe_websocket_server(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    # ── VAD state (mutated across callback calls) ─────────────────
    state = {
        "is_speaking":    False,
        "silence_count":  0,
        "speech_buffer":  [],       # accumulates chunks during speech
        "pre_roll":       [],       # small ring buffer of recent silent chunks
    }

    # ── Audio thread → queue bridge ──────────────────────────────
    def sd_callback(indata, frames, time, status):
        chunk = indata.copy()

        # Normalize int16 → float32 for RMS calculation
        rms = np.sqrt(np.mean((chunk.astype(np.float32) / 2 ** 15) ** 2))
        if rms > SILENCE_THRESHOLD:
            # ── Speech detected ───────────────────────────────────
            if not state["is_speaking"]:
                state["is_speaking"] = True
                # Prepend pre-roll so we don't clip the first syllable
                state["speech_buffer"] = list(state["pre_roll"])

            state["silence_count"] = 0
            state["speech_buffer"].append(chunk)

        else:
            # ── Silence detected ──────────────────────────────────
            if state["is_speaking"]:
                state["silence_count"] += 1
                state["speech_buffer"].append(chunk)  # include trailing silence

                if state["silence_count"] >= SILENCE_FRAMES:
                    # ── Flush complete utterance to queue ─────────
                    utterance = np.concatenate(state["speech_buffer"])
                    loop.call_soon_threadsafe(
                        queue.put_nowait, utterance
                    )

                    # Reset state
                    state["is_speaking"]   = False
                    state["silence_count"] = 0
                    state["speech_buffer"] = []

            else:
                # Not speaking — maintain pre-roll ring buffer
                state["pre_roll"].append(chunk)
                if len(state["pre_roll"]) > PRE_ROLL_FRAMES:
                    state["pre_roll"].pop(0)

    # ── Dequeue and play continuously (no gap between chunks) ────
    async def player_worker():
        while True:
            utterance = await queue.get()
            # utterance = normalize(utterance)
            # logger.info(f"--> normalize: {utterance}")
            duration = len(utterance) / SAMPLE_RATE
            if duration < MIN_DURATION_SECONDS:
                queue.task_done()
                continue

            wav_bytes = to_wav_bytes(utterance, SAMPLE_RATE)
            logger.info(f"--> wav bytes: {wav_bytes}")
            try:
                transcript = openai_client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=("audio.wav", wav_bytes, "audio/wav"),
                    response_format="text",
                    language="en"
                )
            except Exception as e:
                transcript = ""
                logger.error(e)

            await websocket.send_text(f"Done processing {transcript}")
            queue.task_done()

    # ── Keep InputStream alive while WebSocket is connected ──────
    async def receiver_worker():
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            callback=sd_callback
        ):
            await asyncio.sleep(3600)  # holds the stream open

    await asyncio.gather(receiver_worker(), player_worker())


@app.websocket("/chat/ws/audio/{mode}")
async def transcribe_websocket(websocket: WebSocket, mode: str):
    def generate_audio(text: str):
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="wav"
        )
        return response.content

    await websocket.accept()
    thread = await client.threads.create(if_exists="do_nothing")
    thread_id = thread["thread_id"]
    logger.info(f"--> WebSocket connected on Thread ID: {thread_id}")

    try:
        while True:
            # 1. receive audio bytes from mobile app
            audio_chunk  = await websocket.receive_bytes()
            wav_audio = pcm16_to_wav(
                audio_chunk,
                sample_rate=48_000,  # Must match the iOS microphone rate
            )
            logger.info(f"--> WebSocket received audio chunk type: {type(wav_audio)}")

            # debug only
            if mode == 'debug':
                logger.info(f"--> Entering local debug mode: {mode}")
                audio_chunk = generate_dummy_audio()

            # 2. transcribe
            try:
                transcript = openai_client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=("audio.wav", wav_audio, "audio/wav"),
                    response_format="text"
                )
            except Exception as e:
                logger.error(f"--> transcript model error: {e}")

            logger.info(f"--> speech to text input: {transcript}")
            #
            # if mode == 'debug':
            #     logger.info(f"--> The audio input is: {transcript}")

            transcript = ("can you help check my next meeting agenda and retrieve the attendee's profile, "
                          "prepare a meeting note. thanks")

            # 3. invoke agent to process the task
            final_response = None
            async for chunk in client.runs.stream(
                    thread_id,
                    "assistant",
                    input={"messages": [{"role": "human", "content": transcript}]},
                    stream_mode="values",
            ):
                if chunk.data and "messages" in chunk.data:
                    final_response = chunk.data["messages"][-1]

            content = final_response["content"]
            if isinstance(content, list):
                tts_input = " ".join(
                    block["text"] for block in content if block.get("type") == "text"
                )
            else:
                tts_input = content
            if mode == 'debug':
                logger.info(f"--> The agent response is: {tts_input}")

            # 4. tts
            # loop = asyncio.get_event_loop()
            # audio_bytes = await loop.run_in_executor(executor, lambda: generate_audio(tts_input))

            logger.info(f"--> Completed task processing: {thread_id}")
            # await websocket.send_bytes(audio_bytes)
            await websocket.send_text(f"{content}")
    except WebSocketDisconnect:
        print("disconnected")
    except Exception as e:
        print(f"Error handling audio stream: {e}")

@app.websocket("/chat/ws/realtime")
async def realtime_websocket(websocket: WebSocket):
    """Bridge the mobile PCM16 audio stream to the OpenAI Realtime API."""
    await websocket.accept()
    logger.info("--> Realtime WebSocket connected")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        await websocket.send_text("Realtime API error: OPENAI_API_KEY is not configured.")
        await websocket.close(code=1011)
        return

    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2")
    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with connect(url, additional_headers=headers) as openai_websocket:
            await openai_websocket.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": "Respond briefly and helpfully to the user.",
                    "output_modalities": ["text"],
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24_000},
                            "turn_detection": {
                                "type": "server_vad",
                                "create_response": True,
                                "interrupt_response": True,
                            },
                        }
                    },
                },
            }))

            async def forward_mobile_audio():
                pending_pcm = b""
                total_input_bytes = 0
                while True:
                    audio_chunk = await websocket.receive_bytes()
                    total_input_bytes += len(audio_chunk)
                    logger.info(
                        "--> Realtime input audio: chunk=%d bytes, total=%d bytes "
                        "(%.2f seconds at 48 kHz PCM16 mono)",
                        len(audio_chunk),
                        total_input_bytes,
                        total_input_bytes / (48_000 * 2),
                    )
                    pcm_48khz = pending_pcm + audio_chunk

                    # Each output sample is the average of two adjacent 48 kHz
                    # PCM16 samples. Retain incomplete samples between frames.
                    complete_bytes = len(pcm_48khz) - (len(pcm_48khz) % 4)
                    pending_pcm = pcm_48khz[complete_bytes:]
                    if not complete_bytes:
                        continue

                    samples = np.frombuffer(
                        pcm_48khz[:complete_bytes], dtype="<i2"
                    ).astype(np.int32)
                    pcm_24khz = (
                        samples.reshape(-1, 2).sum(axis=1) // 2
                    ).astype("<i2").tobytes()

                    await openai_websocket.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm_24khz).decode("ascii"),
                    }))

            async def forward_openai_events():
                response_text_parts = []
                async for raw_event in openai_websocket:
                    event = json.loads(raw_event)
                    event_type = event.get("type")

                    if event_type == "response.output_text.delta":
                        delta = event.get("delta", "")
                        response_text_parts.append(delta)
                        await websocket.send_text(delta)
                    elif event_type == "response.output_text.done":
                        response_text = event.get("text") or "".join(
                            response_text_parts
                        )
                        logger.info(
                            "--> Realtime output response: %s", response_text
                        )
                        response_text_parts.clear()
                    elif event_type == "error":
                        error = event.get("error", {})
                        message = error.get("message", "Unknown Realtime API error")
                        logger.error("--> OpenAI Realtime API error: %s", message)
                        await websocket.send_text(f"Realtime API error: {message}")

            tasks = {
                asyncio.create_task(forward_mobile_audio()),
                asyncio.create_task(forward_openai_events()),
            }
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
    except WebSocketDisconnect:
        logger.info("--> Realtime WebSocket disconnected")
    except Exception as exc:
        logger.exception("--> Realtime WebSocket failed")
        try:
            await websocket.send_text(f"Realtime API error: {exc}")
            await websocket.close(code=1011)
        except (RuntimeError, WebSocketDisconnect):
            pass

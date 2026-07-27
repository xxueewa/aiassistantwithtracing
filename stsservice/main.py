import io
import os
import re
import wave
import base64
import threading
import logging
from uuid import uuid4
from collections import deque
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
    audio = np.asarray(audio, dtype=np.int16).reshape(-1)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    buffer.seek(0)
    buffer.name = "audio.wav"
    return buffer


def content_to_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content or "")


def build_websocket_event(event_type: str, **payload) -> dict:
    return {"type": event_type, **payload}


async def send_event(
    websocket: WebSocket,
    event_type: str,
    **payload,
) -> None:
    await websocket.send_json(build_websocket_event(event_type, **payload))


def generate_speech_wav(text: str) -> bytes:
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format="wav",
    )
    return response.content

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

DEVICE_SAMPLE_RATE = 48_000
TRANSCRIBE_SAMPLE_RATE = 16_000
CHANNELS = 1
AUDIO_BLOCK_SIZE = 960  # 20 ms at 48 kHz; short blocks make VAD responsive.
DTYPE = np.int16

END_OF_UTTERANCE_SECONDS = 0.85
PRE_ROLL_SECONDS = 0.24
TRAILING_PAD_SECONDS = 0.16
MIN_SPEECH_SECONDS = 0.45
MAX_UTTERANCE_SECONDS = 12.0

NOISE_FLOOR_ALPHA = 0.03
SPEECH_RMS_MIN = 0.012
SPEECH_RMS_MARGIN = 3.0
TRIM_RMS_MIN = 0.006

SENTENCE_IDLE_FLUSH_SECONDS = 1.2
MAX_PENDING_SENTENCE_CHARS = 180
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def pcm16_rms(audio: np.ndarray) -> float:
    samples = np.asarray(audio, dtype=np.int16).reshape(-1).astype(np.float32)
    if samples.size == 0:
        return 0.0
    normalized = samples / np.iinfo(np.int16).max
    return float(np.sqrt(np.mean(normalized * normalized)))


def resample_pcm16(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.int16).reshape(-1)
    if source_rate == target_rate or samples.size == 0:
        return samples

    duration = samples.size / source_rate
    target_size = max(1, int(round(duration * target_rate)))
    source_times = np.arange(samples.size, dtype=np.float64) / source_rate
    target_times = np.arange(target_size, dtype=np.float64) / target_rate
    resampled = np.interp(
        target_times,
        source_times,
        samples.astype(np.float32),
    )
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def trim_quiet_edges(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.int16).reshape(-1)
    frame_size = max(1, int(sample_rate * 0.02))
    frame_count = samples.size // frame_size
    if frame_count == 0:
        return samples

    frames = samples[:frame_count * frame_size].reshape(frame_count, frame_size)
    rms_values = np.array([pcm16_rms(frame) for frame in frames])
    threshold = max(TRIM_RMS_MIN, float(rms_values.max(initial=0.0)) * 0.08)
    voiced = np.flatnonzero(rms_values > threshold)
    if voiced.size == 0:
        return np.array([], dtype=np.int16)

    pad = int(TRAILING_PAD_SECONDS * sample_rate)
    start = max(0, int(voiced[0]) * frame_size - pad)
    end = min(samples.size, (int(voiced[-1]) + 1) * frame_size + pad)
    return samples[start:end]


def split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    text = " ".join(buffer.split())
    if not text:
        return [], ""

    parts = SENTENCE_END_RE.split(text)
    if len(parts) == 1:
        if len(text) >= MAX_PENDING_SENTENCE_CHARS:
            return [text], ""
        return [], text

    if re.search(r"[.!?]$", text):
        return [part.strip() for part in parts if part.strip()], ""

    complete = [part.strip() for part in parts[:-1] if part.strip()]
    return complete, parts[-1].strip()

@app.websocket("/ws/audio/{thread_uuid}")
async def transcribe_websocket_server(
    websocket: WebSocket,
    thread_uuid: str,
):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    thread = await client.threads.create(
        thread_id=thread_uuid,
        if_exists="do_nothing",
    )
    thread_id = thread["thread_id"]
    logger.info("--> /ws/audio connected on Thread ID: %s", thread_id)
    await send_event(
        websocket,
        "session.ready",
        thread_id=thread_id,
    )

    quiet_frames_to_end = int(END_OF_UTTERANCE_SECONDS * DEVICE_SAMPLE_RATE)
    pre_roll_frames = max(1, int(PRE_ROLL_SECONDS * DEVICE_SAMPLE_RATE / AUDIO_BLOCK_SIZE))
    max_utterance_frames = int(MAX_UTTERANCE_SECONDS * DEVICE_SAMPLE_RATE)

    # VAD state is owned by the PortAudio callback thread. Keep it tiny and only
    # hand completed utterances to the asyncio loop with call_soon_threadsafe().
    state = {
        "is_speaking": False,
        "quiet_frames": 0,
        "total_frames": 0,
        "voiced_frames": 0,
        "noise_floor": 0.003,
        "speech_buffer": [],
        "pre_roll": deque(maxlen=pre_roll_frames),
    }

    def sd_callback(indata, frames, time, status):
        if status:
            logger.warning("--> SoundDevice input status: %s", status)

        chunk = indata.copy().reshape(-1)
        rms = pcm16_rms(chunk)
        speech_threshold = max(
            SPEECH_RMS_MIN,
            state["noise_floor"] * SPEECH_RMS_MARGIN,
        )
        is_voice = rms > speech_threshold

        if not state["is_speaking"] and not is_voice:
            # Track room tone while idle. This adaptive floor handles different
            # microphones better than one fixed silence threshold.
            state["noise_floor"] = (
                (1 - NOISE_FLOOR_ALPHA) * state["noise_floor"]
                + NOISE_FLOOR_ALPHA * max(rms, 0.0005)
            )
            state["pre_roll"].append(chunk)
            return

        if is_voice:
            if not state["is_speaking"]:
                state["is_speaking"] = True
                state["speech_buffer"] = list(state["pre_roll"])
                state["quiet_frames"] = 0
                state["total_frames"] = sum(len(item) for item in state["speech_buffer"])
                state["voiced_frames"] = 0

            state["quiet_frames"] = 0
            state["voiced_frames"] += frames
            state["speech_buffer"].append(chunk)
            state["total_frames"] += frames
        else:
            state["quiet_frames"] += frames
            state["speech_buffer"].append(chunk)
            state["total_frames"] += frames

        should_flush = (
            state["quiet_frames"] >= quiet_frames_to_end
            or state["total_frames"] >= max_utterance_frames
        )
        if not should_flush:
            return

        utterance = np.concatenate(state["speech_buffer"])
        voiced_duration = state["voiced_frames"] / DEVICE_SAMPLE_RATE
        if voiced_duration >= MIN_SPEECH_SECONDS:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                (utterance, DEVICE_SAMPLE_RATE),
            )

        state["is_speaking"] = False
        state["quiet_frames"] = 0
        state["total_frames"] = 0
        state["voiced_frames"] = 0
        state["speech_buffer"] = []
        state["pre_roll"].clear()

    async def transcription_worker():
        pending_text = ""
        last_transcript_at = loop.time()

        async def flush_sentences(force: bool = False):
            nonlocal pending_text
            sentences, pending_text = split_complete_sentences(pending_text)
            if force and pending_text:
                sentences.append(pending_text)
                pending_text = ""
            for sentence in sentences:
                response_id = str(uuid4())
                await send_event(
                    websocket,
                    "response.start",
                    response_id=response_id,
                )

                try:
                    final_response = None
                    async for chunk in client.runs.stream(
                            thread_id,
                            "assistant",
                            input={"messages": [{"role": "human", "content": sentence}]},
                            stream_mode="values",
                    ):
                        if chunk.data and "messages" in chunk.data:
                            final_response = chunk.data["messages"][-1]
                except WebSocketDisconnect:
                    raise
                except Exception as exc:
                    logger.exception("--> Agent response failed: %s", exc)
                    await send_event(
                        websocket,
                        "error",
                        code="AGENT_RESPONSE_FAILED",
                        message="Unable to generate a response",
                        response_id=response_id,
                    )
                    continue

                logger.info(f"Complete sentence: {final_response}")
                if not final_response:
                    response_text = "No response"
                elif final_response["type"] == "human":
                    response_text = "No Internet"
                else:
                    response_text = final_response["content"]

                await send_event(
                    websocket,
                    "response.delta",
                    response_id=response_id,
                    delta=response_text,
                )

                if response_text:
                    # TTS is a synchronous OpenAI SDK call; keep it off the
                    # event loop so the websocket stays responsive.
                    try:
                        audio_bytes = await loop.run_in_executor(
                            executor,
                            lambda text=response_text: generate_speech_wav(text),
                        )
                        await send_event(
                            websocket,
                            "response.audio",
                            response_id=response_id,
                            audio=base64.b64encode(audio_bytes).decode("ascii"),
                            format="wav",
                        )
                    except Exception as exc:
                        logger.exception("--> TTS generation failed: %s", exc)

                await send_event(
                    websocket,
                    "response.done",
                    response_id=response_id,
                    text=response_text,
                )

        while True:
            try:
                utterance, source_rate = await asyncio.wait_for(
                    queue.get(),
                    timeout=SENTENCE_IDLE_FLUSH_SECONDS,
                )
            except asyncio.TimeoutError:
                if pending_text and loop.time() - last_transcript_at >= SENTENCE_IDLE_FLUSH_SECONDS:
                    await flush_sentences(force=True)
                continue

            try:
                utterance = trim_quiet_edges(utterance, source_rate)
                voiced_duration = len(utterance) / source_rate
                if voiced_duration < MIN_SPEECH_SECONDS or pcm16_rms(utterance) < TRIM_RMS_MIN:
                    continue

                # Most local microphones record most cleanly at 48 kHz. We keep
                # VAD at that device rate, then resample only the accepted speech
                # segment to 16 kHz mono PCM16 before creating the WAV for STT.
                transcribe_audio = resample_pcm16(
                    utterance,
                    source_rate,
                    TRANSCRIBE_SAMPLE_RATE,
                )
                wav_bytes = to_wav_bytes(transcribe_audio, TRANSCRIBE_SAMPLE_RATE)

                # OpenAI's SDK call is synchronous. Running it in the executor
                # keeps the websocket event loop responsive while transcription
                # waits on network I/O.
                try:
                    transcript = await loop.run_in_executor(
                        executor,
                        lambda: openai_client.audio.transcriptions.create(
                            model="gpt-4o-mini-transcribe",
                            file=("audio.wav", wav_bytes, "audio/wav"),
                            response_format="text",
                            language="en",
                        ),
                    )
                except Exception as exc:
                    logger.exception("--> Transcription failed: %s", exc)
                    await send_event(
                        websocket,
                        "error",
                        code="TRANSCRIPTION_FAILED",
                        message="Unable to transcribe audio",
                    )
                    continue
                transcript = transcript.strip()
                if not transcript:
                    continue

                pending_text = f"{pending_text} {transcript}".strip()
                last_transcript_at = loop.time()
                await flush_sentences()
            except WebSocketDisconnect:
                raise
            except Exception as e:
                logger.exception("--> Transcription failed: %s", e)
            finally:
                queue.task_done()

    async def receiver_worker():
        # sounddevice opens a PortAudio stream on a native callback thread. The
        # websocket handler itself remains asyncio-based; this coroutine simply
        # keeps the stream alive until the client disconnects or the task ends.
        with sd.InputStream(
            samplerate=DEVICE_SAMPLE_RATE,
            blocksize=AUDIO_BLOCK_SIZE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=sd_callback,
        ):
            while True:
                await asyncio.sleep(1)

    tasks = {
        asyncio.create_task(receiver_worker()),
        asyncio.create_task(transcription_worker()),
    }
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
    except WebSocketDisconnect:
        logger.info("--> /ws/audio disconnected")
    finally:
        for task in tasks:
            task.cancel()


@app.post("/summarize")
def summarize(req: ChatRequest):
    PROMPT = f"You are going to summarize this sentence {req.message}. Keep the key information relating to location, schedule, meeting trip, etc. Keep the sentence less than 100 words."
    summary = openai_client.chat.completions.create(
        model = "gpt-5.6",
        messages=[
            {
                "role": "user",
                "content": PROMPT
            }
        ]
    )
    return summary.choices[0].message.content

# @app.websocket("/chat/ws/audio/{mode}")
# async def transcribe_websocket(websocket: WebSocket, mode: str):
#     await websocket.accept()
#     thread = await client.threads.create(if_exists="do_nothing")
#     thread_id = thread["thread_id"]
#     logger.info(f"--> WebSocket connected on Thread ID: {thread_id}")
#
#     try:
#         while True:
#             # 1. receive audio bytes from mobile app
#             audio_chunk  = await websocket.receive_bytes()
#             wav_audio = pcm16_to_wav(
#                 audio_chunk,
#                 sample_rate=48_000,  # Must match the iOS microphone rate
#             )
#             logger.info(f"--> WebSocket received audio chunk type: {type(wav_audio)}")
#
#             # debug only
#             if mode == 'debug':
#                 logger.info(f"--> Entering local debug mode: {mode}")
#                 audio_chunk = generate_dummy_audio()
#
#             # 2. transcribe
#             try:
#                 transcript = openai_client.audio.transcriptions.create(
#                     model="gpt-4o-mini-transcribe",
#                     file=("audio.wav", wav_audio, "audio/wav"),
#                     response_format="text"
#                 )
#             except Exception as e:
#                 logger.error(f"--> transcript model error: {e}")
#
#             logger.info(f"--> speech to text input: {transcript}")
#             #
#             # if mode == 'debug':
#             #     logger.info(f"--> The audio input is: {transcript}")
#
#             transcript = ("can you help check my next meeting agenda and retrieve the attendee's profile, "
#                           "prepare a meeting note. thanks")
#
#             # 3. invoke agent to process the task
#             final_response = None
#             async for chunk in client.runs.stream(
#                     thread_id,
#                     "assistant",
#                     input={"messages": [{"role": "human", "content": transcript}]},
#                     stream_mode="values",
#             ):
#                 if chunk.data and "messages" in chunk.data:
#                     final_response = chunk.data["messages"][-1]
#
#             content = final_response["content"]
#             if isinstance(content, list):
#                 tts_input = " ".join(
#                     block["text"] for block in content if block.get("type") == "text"
#                 )
#             else:
#                 tts_input = content
#             if mode == 'debug':
#                 logger.info(f"--> The agent response is: {tts_input}")
#
#             logger.info(f"--> Completed task processing: {thread_id}")
#             # await websocket.send_bytes(audio_bytes)
#             await websocket.send_text(f"{content}")
#     except WebSocketDisconnect:
#         print("disconnected")
#     except Exception as e:
#         print(f"Error handling audio stream: {e}")
#
# @app.websocket("/chat/ws/realtime")
# async def realtime_websocket(websocket: WebSocket):
#     """Bridge the mobile PCM16 audio stream to the OpenAI Realtime API."""
#     await websocket.accept()
#     logger.info("--> Realtime WebSocket connected")
#
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         await websocket.send_text("Realtime API error: OPENAI_API_KEY is not configured.")
#         await websocket.close(code=1011)
#         return
#
#     model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2")
#     url = f"wss://api.openai.com/v1/realtime?model={model}"
#     headers = {"Authorization": f"Bearer {api_key}"}
#
#     try:
#         async with connect(url, additional_headers=headers) as openai_websocket:
#             await openai_websocket.send(json.dumps({
#                 "type": "session.update",
#                 "session": {
#                     "type": "realtime",
#                     "instructions": "Respond briefly and helpfully to the user.",
#                     "output_modalities": ["text"],
#                     "audio": {
#                         "input": {
#                             "format": {"type": "audio/pcm", "rate": 24_000},
#                             "turn_detection": {
#                                 "type": "server_vad",
#                                 "create_response": True,
#                                 "interrupt_response": True,
#                             },
#                         }
#                     },
#                 },
#             }))
#
#             async def forward_mobile_audio():
#                 pending_pcm = b""
#                 total_input_bytes = 0
#                 while True:
#                     audio_chunk = await websocket.receive_bytes()
#                     total_input_bytes += len(audio_chunk)
#                     logger.info(
#                         "--> Realtime input audio: chunk=%d bytes, total=%d bytes "
#                         "(%.2f seconds at 48 kHz PCM16 mono)",
#                         len(audio_chunk),
#                         total_input_bytes,
#                         total_input_bytes / (48_000 * 2),
#                     )
#                     pcm_48khz = pending_pcm + audio_chunk
#
#                     # Each output sample is the average of two adjacent 48 kHz
#                     # PCM16 samples. Retain incomplete samples between frames.
#                     complete_bytes = len(pcm_48khz) - (len(pcm_48khz) % 4)
#                     pending_pcm = pcm_48khz[complete_bytes:]
#                     if not complete_bytes:
#                         continue
#
#                     samples = np.frombuffer(
#                         pcm_48khz[:complete_bytes], dtype="<i2"
#                     ).astype(np.int32)
#                     pcm_24khz = (
#                         samples.reshape(-1, 2).sum(axis=1) // 2
#                     ).astype("<i2").tobytes()
#
#                     await openai_websocket.send(json.dumps({
#                         "type": "input_audio_buffer.append",
#                         "audio": base64.b64encode(pcm_24khz).decode("ascii"),
#                     }))
#
#             async def forward_openai_events():
#                 response_text_parts = []
#                 async for raw_event in openai_websocket:
#                     event = json.loads(raw_event)
#                     event_type = event.get("type")
#
#                     if event_type == "response.output_text.delta":
#                         delta = event.get("delta", "")
#                         response_text_parts.append(delta)
#                         await websocket.send_text(delta)
#                     elif event_type == "response.output_text.done":
#                         response_text = event.get("text") or "".join(
#                             response_text_parts
#                         )
#                         logger.info(
#                             "--> Realtime output response: %s", response_text
#                         )
#                         response_text_parts.clear()
#                     elif event_type == "error":
#                         error = event.get("error", {})
#                         message = error.get("message", "Unknown Realtime API error")
#                         logger.error("--> OpenAI Realtime API error: %s", message)
#                         await websocket.send_text(f"Realtime API error: {message}")
#
#             tasks = {
#                 asyncio.create_task(forward_mobile_audio()),
#                 asyncio.create_task(forward_openai_events()),
#             }
#             done, pending = await asyncio.wait(
#                 tasks, return_when=asyncio.FIRST_COMPLETED
#             )
#             for task in pending:
#                 task.cancel()
#             await asyncio.gather(*pending, return_exceptions=True)
#             for task in done:
#                 task.result()
#     except WebSocketDisconnect:
#         logger.info("--> Realtime WebSocket disconnected")
#     except Exception as exc:
#         logger.exception("--> Realtime WebSocket failed")
#         try:
#             await websocket.send_text(f"Realtime API error: {exc}")
#             await websocket.close(code=1011)
#         except (RuntimeError, WebSocketDisconnect):
#             pass

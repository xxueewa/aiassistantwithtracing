import io
import os
import wave
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
            if mode == 'debug':
                logger.info(f"--> Entering local debug mode: {mode}")
                audio_chunk = generate_dummy_audio()

            # 2. transcribe
            transcript = openai_client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=("audio.wav", audio_chunk, "audio/wav"),
                response_format="text"
            )

            if mode == 'debug':
                logger.info(f"--> The audio input is: {transcript}")

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
            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(executor, lambda: generate_audio(tts_input))

            logger.info(f"--> Completed task processing: {thread_id}")
            await websocket.send_bytes(audio_bytes)
    except WebSocketDisconnect:
        print("disconnected")
    except Exception as e:
        print(f"Error handling audio stream: {e}")






from typing import Optional

from langgraph_sdk import get_client
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

import numpy as np
import sounddevice as sd
import openai
from pydub import AudioSegment
import io
import os
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

load_dotenv()

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None  # works on Python 3.8+

app = FastAPI()
client = get_client(url="http://localhost:2024")

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
        "calculator",
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



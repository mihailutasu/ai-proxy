from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
import os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse, JSONResponse

load_dotenv()

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class RequestModel(BaseModel):
    model: str
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = 1.0
    stream: bool = False
    messages: Optional[List[Message]] = None
    prompt: Optional[str] = None

class ResponseModel(BaseModel):
    response: str

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")
client = Groq(api_key=api_key)

def generate_grok_response(messages: Optional[List[Message]], prompt: Optional[str], model: str, temperature: float, max_tokens: int, top_p: float):
    try:
        if messages:
            messages_data = [{"role": msg.role, "content": msg.content} for msg in messages]
        else:
            messages_data = [{"role": "user", "content": prompt}]
        
        print("Sending to Grok:", messages_data)
        response = client.chat.completions.create(
            model=model,
            messages=messages_data,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False,
        )
        print("Received from Grok:", response)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in generate_grok_response: {e}")
        raise e

def generate_response_stream(messages: Optional[List[Message]], prompt: Optional[str], model: str, temperature: float, max_tokens: int, top_p: float):
    try:
        if messages:
            messages_data = [{"role": msg.role, "content": msg.content} for msg in messages]
        else:
            messages_data = [{"role": "user", "content": prompt}]
        
        print("Sending to Grok:", messages_data)
        completion = client.chat.completions.create(
            model=model,
            messages=messages_data,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True
        )
        for chunk in completion:
            content = chunk.choices[0].delta.content or ''
            print("Received chunk from Grok:", content)
            if content:
                yield f"data: {content}\n\n"
    except Exception as e:
        print(f"Error in stream: {e}")
        print(messages_data)
        yield f"data: [ERROR] {str(e)}\n\n"

@app.post("/v1/chat/completions", response_model=ResponseModel)
async def chat_completions(request: RequestModel):
    try:
        print(f"Received request: {request.json()}")
        
        if request.stream:
            return StreamingResponse(
                generate_response_stream(
                    request.messages, 
                    request.prompt,
                    request.model, 
                    request.temperature, 
                    request.max_tokens,
                    request.top_p
                ),
                media_type="text/event-stream"
            )
        else:
            response_text = generate_grok_response(
                request.messages,
                request.prompt,
                request.model, 
                request.temperature, 
                request.max_tokens,
                request.top_p
            )
            return ResponseModel(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/api/generate", response_model=ResponseModel)
async def generate(request: Request):
    try:
        raw_body = await request.body()
        print(f"Raw request body: {raw_body.decode('utf-8')}")
        
        request_model = RequestModel.parse_raw(raw_body)
        print(f"Parsed request: {request_model.json()}")

        if request_model.stream:
            return StreamingResponse(
                generate_response_stream(
                    request_model.messages, 
                    request_model.prompt,
                    request_model.model, 
                    request_model.temperature, 
                    request_model.max_tokens,
                    request_model.top_p
                ),
                media_type="text/event-stream"
            )
        else:
            response_text = generate_grok_response(
                request_model.messages,
                request_model.prompt,
                request_model.model, 
                request_model.temperature, 
                request_model.max_tokens,
                request_model.top_p
            )
            return ResponseModel(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1235)


import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Dict
from groq import Groq
import os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

load_dotenv()

app = FastAPI()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level to INFO
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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

def prepare_messages_data(messages: Optional[List[Message]], prompt: Optional[str]) -> List[dict]:
    if messages:
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    return [{"role": "user", "content": prompt}]

def get_response_stream(messages_data: List[dict], model: str, temperature: float, max_tokens: int, top_p: float, stream: bool):
    try:
        response_chunks = []
        completion = client.chat.completions.create(
            model=model,
            messages=messages_data,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream
        )

        if stream:
            for chunk in completion:
                content = chunk.choices[0].delta.content or ''
                # logger.info(f"Received chunk from Grok: {content}")
                if content:
                    response_chunks.append(content)
                    yield f"data: {content}\n\n"
        else:
            response = completion
            content = response.choices[0].message.content
            logger.info(f"Received from Grok: {content}")
            response_chunks.append(content)
            yield f"data: {content}\n\n"
            
        final_response = ''.join(response_chunks)
        logger.info(f"Final response: {final_response}")

    except Exception as e:
        logger.error(f"Error in stream: {e}")
        yield f"data: [ERROR] {str(e)}\n\n"

@app.post("/v1/chat/completions", response_model=ResponseModel)
async def chat_completions(request: RequestModel):
    try:
        logger.info(f"Received request: {request.model_dump_json()}")

        messages_data: List[Dict[str, str]] = prepare_messages_data(request.messages, request.prompt)

        # Prepare the message
        prepared_message = ' '.join(
            f"{str(key)}: {str(value)}" for item in messages_data for key, value in item.items()
        )

        # Log the prepared message
        logger.info(f"Prepared messages data: {prepared_message}")

        return StreamingResponse(
            get_response_stream(
                messages_data,
                request.model,
                request.temperature,
                request.max_tokens,
                request.top_p,
                request.stream
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/api/generate", response_model=ResponseModel)
async def generate(request: Request):
    try:
        raw_body = await request.body()
        
        logger.info(f"Raw request body: {raw_body.decode('utf-8')}")

        request_model = RequestModel.model_validate_json(raw_body)
        logger.info(f"Parsed request: {request_model.model_dump_json()}")

        messages_data = prepare_messages_data(request_model.messages, request_model.prompt)

        return StreamingResponse(
            get_response_stream(
                messages_data,
                request_model.model,
                request_model.temperature,
                request_model.max_tokens,
                request_model.top_p,
                request_model.stream
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error in generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1235)

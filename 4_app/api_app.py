from typing import Any, Union, Optional
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import tensorflow as tf
import os
import uvicorn
import threading
import asyncio
import subprocess


model_path = "/home/cdsw/models/gen-ai-model/llama-2-13b-chat.ggmlv3.q5_1.bin"

llama2_model = Llama(
    model_path=model_path,
    n_gpu_layers=64,
    n_ctx=2000
)

# Test an inference
print(llama2_model(prompt="Hello ", max_tokens=1))


app = FastAPI()


# This defines the data json format expected for the endpoint, change as needed
class TextInput(BaseModel):
    inputs: str
    parameters: Union[dict[str, Any], None]

@app.get("/")
def status_gpu_check() -> dict[str, str]:
    gpu_msg = "Available" if tf.test.is_gpu_available() else "Unavailable"
    return {
        "status": "I am ALIVE!",
        "gpu": gpu_msg
    }


@app.post("/")
def generate_text(data: TextInput) -> dict[str, str]:
    try:
        params = data.parameters or {}
        response = llama2_model(prompt=data.inputs, **params)
        model_out = response['choices'][0]['text']
        return {"generated_text": model_out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
def run_server():
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ['CDSW_APP_PORT']), log_level="warning", reload=False)

server_thread = threading.Thread(target=run_server)
server_thread.start()
from typing import Any, Union, Optional
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
import tensorflow as tf
import os
import uvicorn
import threading
import asyncio
import subprocess
from pymilvus import connections, Collection
from typing import Any, Union, Optional
import utils.vector_db_utils as vector_db
import utils.model_embedding_utils as model_embedding
from llama_cpp import Llama

## Initialize Llama2 Model on app startup
model_path = "/home/cdsw/models/gen-ai-model/llama-2-13b-chat.ggmlv3.q5_1.bin"

llama2_model = Llama(
    model_path=model_path,
    n_gpu_layers=64,
    n_ctx=2000
)

# Test an inference
print(llama2_model(prompt="Hello ", max_tokens=1))

app = FastAPI()

# Helper function for generating responses for the QA app
def get_responses(engine, temperature, token_count, question):
    if engine == "" or question == "" or engine is None or question is None:
        return "One or more fields have not been specified."
    
    if temperature == "" or temperature is None:
      temperature = 1
      
    if token_count == "" or token_count is None:
      token_count = 100

    # Load Milvus Vector DB collection
    vector_db_collection = Collection('cloudera_ml_docs')
    vector_db_collection.load()
    
    # Phase 1: Get nearest knowledge base chunk for a user question from a vector db
    vdb_question = question
    context_chunk = get_nearest_chunk_from_vectordb(vector_db_collection, vdb_question)
    vector_db_collection.release()
    
    if engine == "llama-2-13b-chat":
        # Phase 2a: Perform text generation with LLM model using found kb context chunk
        response = get_llama2_response_with_context(question, context_chunk, temperature, token_count)

    return response

# Get embeddings for a user question and query Milvus vector DB for nearest knowledge base chunk
def get_nearest_chunk_from_vectordb(vector_db_collection, question):
    # Generate embedding for user question
    question_embedding =  model_embedding.get_embeddings(question)
    
    # Define search attributes for Milvus vector DB
    vector_db_search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    
    # Execute search and get nearest vector, outputting the relativefilepath
    nearest_vectors = vector_db_collection.search(
        data=[question_embedding], # The data you are querying on
        anns_field="embedding", # Column in collection to search on
        param=vector_db_search_params,
        limit=1, # limit results to 1 but allow for more via user customization
        expr=None, 
        output_fields=['relativefilepath'], # The fields you want to retrieve from the search result.
        consistency_level="Strong"
    )

    # Return text of the nearest knowledgebase chunk
    response = ""
    for f in nearest_vectors[0]:
        response += str(load_context_chunk_from_data(f.id))
    
    return response
  
# Return the Knowledge Base doc based on Knowledge Base ID (relative file path)
def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f: # Open file in read mode
        return f.read()

  
# Pass through user input to LLM model with enhanced prompt and stop tokens
def get_llama2_response_with_context(question, context, temperature, token_count):
    question = "Answer this question based on given context. If you do not know the answer, do not make something up. This is the question: " + question
    question_and_context = question + "Here is the context: " + context.replace('\n', ' ')

    try:
        params = {
            "temperature": float(temperature),
            "max_tokens": int(token_count)
        }
        response = llama2_model(prompt=question_and_context, **params)

        model_out = response['choices'][0]['text']
        return model_out
    
    except Exception as e:
        return "Error in generating response."
    
    except Exception as e:
        return "Error in generating response."


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
        # Set defaults
        engine = "llama-2-13b-chat" # If you add more engines, you will want to begin passing this as a parameter
        temperature = 1
        token_count = 100
        
        question = data.inputs
        print(str(question))
        params = data.parameters or {}
        
        if 'temperature' in params:
            temperature = int(params['temperature'])
            
        if 'max_tokens' in params:
            token_count = int(params['max_tokens'])
            
        if 'engine' in params:
            engine = str(params['engine'])
        
        print("Using: "+ str(temperature))
        print("Using: "+ str(token_count))
        print("Using: "+ str(engine))
            
        res = get_responses(engine, temperature, token_count, question)
                            
        return {"response": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
def run_server():
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ['CDSW_APP_PORT']), log_level="warning", reload=False)

server_thread = threading.Thread(target=run_server)
server_thread.start()
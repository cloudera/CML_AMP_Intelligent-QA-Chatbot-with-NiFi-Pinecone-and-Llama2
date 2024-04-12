import os
import gradio as gr
import cmlapi
from typing import Any, Union, Optional
from pydantic import BaseModel
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import requests
import json
import time
from typing import Optional
from chromadb.utils import embedding_functions
from huggingface_hub import hf_hub_download
from model import *

if os.getenv("VECTOR_DB") == "PINECONE":
    import pinecone
    USE_PINECONE = True
else:
    USE_PINECONE = False
if os.getenv("VECTOR_DB") == "CHROMA":
    import chromadb
    USE_CHROMA = True 
else:
    USE_CHROMA = False
    
EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"


if USE_PINECONE:
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
    PINECONE_INDEX = os.getenv('COLLECTION_NAME')

    print("initialising Pinecone connection...")
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    print("Pinecone initialised")

    print(f"Getting '{PINECONE_INDEX}' as object...")
    index = pinecone.Index(PINECONE_INDEX)
    print("Success")

    # Get latest statistics from index
    current_collection_stats = index.describe_index_stats()
    print('Total number of embeddings in Pinecone index is {}.'.format(current_collection_stats.get('total_vector_count')))

    
if USE_CHROMA:
    # Connect to local Chroma data
    chroma_client = chromadb.PersistentClient(path="/home/cdsw/chroma-data")
    
    EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
    EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

    COLLECTION_NAME = os.getenv("COLLECTION_NAME")

    print("initialising Chroma DB connection...")

    print(f"Getting '{COLLECTION_NAME}' as object...")
    try:
        chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
        print("Success")
        collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    except:
        print("Creating new collection...")
        collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
        print("Success")

    # Get latest statistics from index
    current_collection_stats = collection.count()
    print('Total number of embeddings in Chroma DB index is ' + str(current_collection_stats))
    
if os.getenv("USE_CML_MODELS") == "True" or os.getenv("USE_CML_MODELS") == True:
    client = cmlapi.default_client(url=os.getenv("CDSW_API_URL").replace("/api/v1", ""), cml_api_key=os.getenv("CDSW_APIV2_KEY"))
    ## Here we assume that only one model has been deployed in the project, if this is not true this should be adjusted (this is reflected by the placeholder 0 in the array)
    model = client.list_models(project_id=os.getenv("CDSW_PROJECT_ID"))
    print(model)
    selected_model = model.models[0]

    ## Save the access key for the model to the environment variable of this project
    MODEL_ACCESS_KEY = selected_model.access_key

    MODEL_ENDPOINT = os.getenv("CDSW_API_URL").replace("https://", "https://modelservice.").replace("/api/v1", "/model?accessKey=")
    MODEL_ENDPOINT = MODEL_ENDPOINT + MODEL_ACCESS_KEY
    
else:
    import bitsandbytes as bnb
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import os
    import time
    import torch

    # Quantization
    # Here quantization is setup to use "Normal Float 4" data type for weights. 
    # This way each weight in the model will take up 4 bits of memory. 
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Create a model object with above parameters
    model_name = "NousResearch/Llama-2-7b-chat-hf"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config,
        device_map='auto',
    )
    
    # Args helper
    def opt_args_value(args, arg_name, default):
      """
      Helper function to interact with LLMs parameters for each call to the model.
      Returns value provided in args[arg_name] or the default value provided.
      """
      if arg_name in args.keys():
        return args[arg_name]
      else:
        return default

    # Define tokenizer parameters
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
# Mamke the call to 
def generate(prompt, max_new_tokens=50, temperature=0, repetition_penalty=1.0, num_beams=1, top_p=1.0, top_k=0):
    """
    Make a request to the LLM, with given parameters (or using default values).

    max_new_tokens     - at how many words will the generated response be capped?
    temperature        - a.k.a. "response creatibity". Controls randomness of the generated response (0 = least random, 1 = more random). 
    repetition_penalty - penalizes the next token if it has already been used in the response (1 = no penlaty)
    num_beams          - controls the number of token sequences generate (1 = only one sequence generated)
    top_p              - cumulative probability to determine how many tokens to keep (i.e. enough tokens will be considered, so their combined probabiliy reaches top_p)
    top_k              - numbe of highest-probability tokens to keep (i.e. only top_k "best" tokens will be considered for response)
    """
    batch = tokenizer(prompt, return_tensors='pt')

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch,
                                    max_new_tokens=max_new_tokens,
                                    repetition_penalty=repetition_penalty,
                                    temperature=temperature,
                                    num_beams=num_beams,
                                    top_p=top_p,
                                    top_k=top_k)

    output=tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Log the response along with parameters
    print("Prompt: %s" % (prompt))
    print("max_new_tokens: %s; temperature: %s; repetition_penalty: %s; num_beams: %s; top_p: %s; top_k: %s" % (max_new_tokens, temperature, repetition_penalty, num_beams, top_p, top_k))
    print("Full Response: %s" % (output))

    return output
    
def main():
    # Configure gradio QA app 
    print("Configuring gradio app")

    DESC = "This AI-powered assistant showcases the flexibility of Cloudera Machine Learning to work with 3rd party solutions for LLMs and Vector Databases, as well as internally hosted models and vector DBs. The prototype does not yet implement chat history and session context - every prompt is treated as a brand new one."
    if os.getenv("VECTOR_DB") == "CHROMA":
        additional_inputs=[gr.Radio(['Local Llama 7B'], label="Select Foundational Model", value="Local Llama 7B"), 
                        gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Select Temperature (Randomness of Response)"),
                        gr.Radio(["50", "100", "250", "500", "1000"], label="Select Number of Tokens (Length of Response)", value="100"),
                        gr.Radio(['None', 'Chroma'], label="Vector Database Choices", value="Chroma")]
    
    if os.getenv("VECTOR_DB") == "PINECONE":
        additional_inputs=[gr.Radio(['Local Llama 7B'], label="Select Foundational Model", value="Local Llama 7B"), 
                            gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Select Temperature (Randomness of Response)"),
                            gr.Radio(["50", "100", "250", "500", "1000"], label="Select Number of Tokens (Length of Response)", value="100"),
                            gr.Radio(['None', 'Pinecone'], label="Vector Database Choices", value="Pinecone")]
        
    # Create the Gradio Interface
    demo = gr.ChatInterface(
        fn=get_responses,  
        title="Enterprise Custom Knowledge Base Chatbot",
        description = DESC,
        additional_inputs=additional_inputs,
        retry_btn = None,
        undo_btn = None,
        clear_btn = None,
        autofocus = True
        )

    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,   
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_READONLY_PORT')))
    print("Gradio app ready")

# Helper function for generating responses for the QA app
def get_responses(message, history, model, temperature, token_count, vector_db):
    
    if model == "Local Llama 7B":
        
        if vector_db == "None":
            context_chunk = ""
            response = get_llama2_response_with_context(message, context_chunk, temperature, token_count)
        
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[:i+1]
                
        elif vector_db == "Pinecone":            
            # Vector search the index
            context_chunk, source, score = get_nearest_chunk_from_pinecone_vectordb(index, message)
            
            # Call CML hosted model
            response = get_llama2_response_with_context(message, context_chunk, temperature, token_count)
            
            # Add reference to specific document in the response
            response = f"{response}\n\n For additional info see: {url_from_source(source)}"
            
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[:i+1]
                
        elif vector_db == "Chroma":
            # Vector search in Chroma
            context_chunk, source = get_nearest_chunk_from_chroma_vectordb(collection, message)
            
            # Call CML hosted model
            response = get_llama2_response_with_context(message, context_chunk, temperature, token_count)
            
            # Add reference to specific document in the response
            response = f"{response}\n\n For additional info see: {url_from_source(source)}"
            
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[:i+1]

def url_from_source(source):
    url = source.replace('/home/cdsw/data/https:/', 'https://').replace('/home/cdsw/data/', 'https://').replace('.txt', '.html')
    return f"[Reference 1]({url})"

# Get embeddings for a user question and query Pinecone vector DB for nearest knowledge base chunk
def get_nearest_chunk_from_pinecone_vectordb(index, question):
    # Generate embedding for user question with embedding model
    retriever = SentenceTransformer(EMBEDDING_MODEL_REPO)
    xq = retriever.encode([question]).tolist()
    xc = index.query(xq, top_k=5,include_metadata=True)
    
    matching_files = []
    scores = []
    for match in xc['matches']:
        # extract the 'file_path' within 'metadata'
        file_path = match['metadata']['file_path']
        # extract the individual scores for each vector
        score = match['score']
        scores.append(score)
        matching_files.append(file_path)

    # Return text of the nearest knowledge base chunk 
    # Note that this ONLY uses the first matching document for semantic search. matching_files holds the top results so you can increase this if desired.
    response = load_context_chunk_from_data(matching_files[0])
    sources = matching_files[0]
    score = scores[0]
    
    print(f"Response of context chunk {response}")
    return response, sources, score
    #return "Cloudera is an Open Data Lakhouse company", "http://cloudera.com", 89 

# Return the Knowledge Base doc based on Knowledge Base ID (relative file path)
def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f: # Open file in read mode
        return f.read()

# Get embeddings for a user question and query Chroma vector DB for nearest knowledge base chunk
def get_nearest_chunk_from_chroma_vectordb(collection, question):
    ## Query Chroma vector DB 
    ## This query returns the two most similar results from a semantic search
    response = collection.query(
                    query_texts=[question],
                    n_results=1
                    # where={"metadata_field": "is_equal_to_this"}, # optional filter
                    # where_document={"$contains":"search_string"}  # optional filter
    )    
    return response['documents'][0][0], response['ids'][0][0]


# Pass through user input to LLM model with enhanced prompt and stop tokens
def get_llama2_response_with_context(question, context, temperature, token_count):
    
    llama_sys = f"<s>[INST]You are a helpful, respectful and honest assistant. If you are unsure about an answer, truthfully say \"I don't know\"."
    
    if context == "":
        # Following LLama's spec for prompt engineering
        llama_inst = f"Please answer the user question.[/INST]</s>"
        question_and_context = f"{llama_sys} {llama_inst} \n [INST] {question} [/INST]"
    else:
        # Add context to the question
        llama_inst = f"Answer the user's question based on the folloing information:\n {context}[/INST]</s>"
        question_and_context = f"{llama_sys} {llama_inst} \n[INST] {question} [/INST]"
        
    try:
        if os.getenv("USE_CML_MODELS") == "True" or os.getenv("USE_CML_MODELS") == True:
            # Build a request payload for CML hosted model
            data={ "request": {"prompt":question_and_context,"temperature":temperature,"max_new_tokens":token_count,"repetition_penalty":1.5} }

            r = requests.post(MODEL_ENDPOINT, data=json.dumps(data), headers={'Content-Type': 'application/json'})

            # Logging
            print(f"Request: {data}")
            print(f"Response: {r.json()}")
            no_inst_response = str(r.json()['response']['prediction']['response'])[len(question_and_context)-2:]

            return no_inst_response
        
        else:
            # Pick up or set defaults for inference options
            temperature = float(temperature)
            max_new_tokens = float(token_count)
            top_p = float(1.0)
            top_k = int(0)
            repetition_penalty = float(1.0)
            num_beams = int(1)

            # Generate response from the LLM
            response = generate(question_and_context, max_new_tokens, temperature, repetition_penalty, num_beams, top_p, top_k)
            return str(response)[len(question_and_context)-2:]
        
    except Exception as e:
        print(e)
        return e

if __name__ == "__main__":
    main()
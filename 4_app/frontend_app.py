import os
import gradio
import pinecone
if os.getenv('VECTOR_DB').upper() == "MILVUS":
    from milvus import default_server
    from pymilvus import connections, Collection
from typing import Any, Union, Optional
from pydantic import BaseModel
import tensorflow as tf
if os.getenv('VECTOR_DB').upper() == "MILVUS":
    import utils.vector_db_utils as vector_db
    import utils.model_embedding_utils as model_embedding
from llama_cpp import Llama
if os.getenv('VECTOR_DB').upper() == "PINECONE":
    from sentence_transformers import SentenceTransformer

## Initialize Llama2 Model on app startup
model_path = "/home/cdsw/models/gen-ai-model/llama-2-13b-chat.ggmlv3.q5_1.bin"

llama2_model = Llama(
    model_path=model_path,
    n_gpu_layers=64,
    n_ctx=2000
)

if os.getenv('VECTOR_DB').upper() == "PINECONE":
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
    PINECONE_INDEX = os.getenv('PINECONE_INDEX')

    print("initialising Pinecone connection...")
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    print("Pinecone initialised")

    print(f"Getting '{PINECONE_INDEX}' as object...")
    index = pinecone.Index(PINECONE_INDEX)
    print("Success")

    # Get latest statistics from index
    current_collection_stats = index.describe_index_stats()
    print('Total number of embeddings in Pinecone index is {}.'.format(current_collection_stats.get('total_vector_count')))


# Test an inference
print(llama2_model(prompt="Hello ", max_tokens=1))

app_css = f"""
        .gradio-header {{
            color: white;
        }}
        .gradio-description {{
            color: white;
        }}

        #custom-logo {{
            text-align: center;
        }}
        .gr-interface {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        .gradio-header {{
            background-color: rgba(0, 0, 0, 0.5);
        }}
        .gradio-input-box, .gradio-output-box {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        h1 {{
            color: white; 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: large; !important;
        }}
"""

def main():
    # Configure gradio QA app 
    print("Configuring gradio app")
    demo = gradio.Interface(fn=get_responses,
                            title="Enterprise Custom Knowledge Base Chatbot with Llama2",
                            description="This AI-powered assistant uses Cloudera DataFlow (NiFi) to scrape a website's sitemap and create a knowledge base. The information it provides as a response is context driven by what is available at the scraped websites. It uses Meta's open-source Llama2 model and the sentence transformer model all-mpnet-base-v2 to evaluate context and form an accurate response from the semantic search. It is fine tuned for questions stemming from topics in its knowledge base, and as such may have limited knowledge outside of this domain. As is always the case with prompt engineering, the better your prompt, the more accurate and specific the response.",
                            inputs=[gradio.Radio(['llama-2-13b-chat'], label="Select Model", value="llama-2-13b-chat"), gradio.Radio(['1', '2', '3'], label="Select Temperature (Randomness of Response)", value=["1", "2", "3"]), gradio.Radio(["50", "100", "250", "500", "1000"], label="Select Number of Tokens (Length of Response)", value=["50", "100", "250", "500", "1000"]), gradio.Textbox(label="Topic Weight", placeholder="This field can be used to prioritize a topic weight."), gradio.Textbox(label="Question", placeholder="Enter your question here.")],
                            outputs=[gradio.Textbox(label="Llama2 Model Response"), gradio.Textbox(label="Context Data Source(s)")],
                            allow_flagging="never",
                            css=app_css)


    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
    print("Gradio app ready")

# Helper function for generating responses for the QA app
def get_responses(engine, temperature, token_count, topic_weight, question):
    if engine is "" or question is "" or engine is None or question is None:
        return "One or more fields have not been specified."
    if temperature is "" or temperature is None:
      temperature = 1
      
    if topic_weight is "" or topic_weight is None:
      topic_weight = None
      
    if token_count is "" or token_count is None:
      token_count = 100
    
    if os.getenv('VECTOR_DB').upper() == "MILVUS":
        # Load Milvus Vector DB collection
        vector_db_collection = Collection('cloudera_ml_docs')
        vector_db_collection.load()
    
    # Phase 1: Get nearest knowledge base chunk for a user question from a vector db
    if topic_weight: 
        vdb_question = "Topic: " + topic_weight + " Question: " + question
    else:
        vdb_question = question
        
    if os.getenv('VECTOR_DB').upper() == "MILVUS":
        context_chunk, sources = get_nearest_chunk_from_milvus_vectordb(vector_db_collection, vdb_question)
        vector_db_collection.release()
        
    if os.getenv('VECTOR_DB').upper() == "PINECONE":
        context_chunk, sources = get_nearest_chunk_from_pinecone_vectordb(index, vdb_question)

    if engine == "llama-2-13b-chat":
        # Phase 2a: Perform text generation with LLM model using found kb context chunk
        response = get_llama2_response_with_context(question, context_chunk, temperature, token_count, topic_weight)

    return response, sources

# Get embeddings for a user question and query Milvus vector DB for nearest knowledge base chunk
def get_nearest_chunk_from_milvus_vectordb(vector_db_collection, question):
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
    sources = ""
    for f in nearest_vectors[0]:
        response += str(load_context_chunk_from_data(f.id))
        sources += f.id
    
    return response, sources

# Get embeddings for a user question and query Pinecone vector DB for nearest knowledge base chunk
def get_nearest_chunk_from_pinecone_vectordb(index, question):
    # Generate embedding for user question with embedding model
    retriever = SentenceTransformer('models/embedding-model')
    xq = retriever.encode([question]).tolist()
    xc = index.query(xq, top_k=5,
                 include_metadata=True)
    
    matching_files = []
    for match in xc['matches']:
        # extract the 'file_path' within 'metadata'
        file_path = match['metadata']['file_path']
        matching_files.append(file_path)

    # Return text of the nearest knowledge base chunk 
    # Note that this ONLY uses the first matching document for semantic search. matching_files holds the top results so you can increase this if desired.
    response = load_context_chunk_from_data(matching_files[0])
    sources = matching_files[0]
    return response, sources
  
# Return the Knowledge Base doc based on Knowledge Base ID (relative file path)
def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f: # Open file in read mode
        return f.read()

  
# Pass through user input to LLM model with enhanced prompt and stop tokens
def get_llama2_response_with_context(question, context, temperature, token_count, topic_weight):

    if topic_weight is not None:
        question = "Answer this question based on given context. If you do not know the answer, do not make something up. You should know that this question is about the topic " + str(topic_weight) + " This is the question: " + str(question)
    else:
        question = "Answer this question based on given context. If you do not know the answer, do not make something up. This is the question: " + str(question)
    
    question_and_context = question + "Here is the context: " + str(context)

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


if __name__ == "__main__":
    main()

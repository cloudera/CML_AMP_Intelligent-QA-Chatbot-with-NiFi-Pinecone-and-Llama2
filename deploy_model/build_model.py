##################################################################################################################################
# Purpose     : This python file is used to load a LLM as an API endpoint
# Description : The function summarize is a wrapper to be used as an entry point function when calling the model as an CML API end
#               point
# Testing     : The File can be tested on command line by simply uncommenting the main function and invoking the file as follows 
#               python3 004_deploy-and-test-models/LLM_inference.py'{"action":"summarize", "document":"this is a test of summarization"}'
#               Do make sure to comment main function  back again, before using as an API
#
##################################################################################################################################

import os
import json
import sys
import logging
from llama_cpp import Llama

# Initialize Llama2 Model on app startup
model_path = "./models/gen-ai-model/llama-2-13b-chat.ggmlv3.q5_1.bin"

llama2_model = Llama(
    model_path=model_path,
    n_gpu_layers=64,
    n_ctx=2000
)

logging.basicConfig(level=logging.ERROR)
#
model_name = "llama2-model7"


def get_llama2_response(param):
    print(param)
    if  type(param) is str :
        print('param is a json string')
        obj = json.loads(param)
        question = obj['question']
        context = obj['context']
        temperature = obj['temperature']
        token_count = obj['token_count']
        topic_weight = obj['topic_weight']
        
    elif type(param) is dict :
        print('param is a dictionary')
        question = param['question']
        context = param['context']
        temperature = param['temperature']
        token_count = param['token_count']
        topic_weight = param['topic_weight']
    
    if topic_weight != None:
        question = "Answer this question based on given context. If you do not know the answer, do not make something up. You should know that this question is about the topic " + topic_weight + " This is the question: " + question
    else:
        question = "Answer this question based on given context. If you do not know the answer, do not make something up. This is the question: " + question
    
    question_and_context = question + "Here is the context: " + context

    try:
        params = {
            "temperature": temperature,
            "max_tokens": token_count
        }
        
        response = llama2_model(prompt=question_and_context, **params)
        model_out = response['choices'][0]['text']
        return {"response": model_out}
    
    except Exception as e:
        return {"response": "Error in generating response."}

# def summarize(param):

#     print(param)
#     if  type(param) is str :
#         print('param is a json string')        
#         obj = json.loads(param)
#         data = obj['document']
#         action=obj['action']
#     elif type(param) is dict :
#         print('param is a dictionary')
#         data = param['document']
#         action=param['action']
        
#     # return json.dumps({"summary": data})
    
#     ## Initialize Llama2 Model on app startup
#     llama2_model.llama2_model
#     tokenizer = AutoTokenizer.from_pretrained('./models/')
#     model = TFAutoModelForSeq2SeqLM.from_pretrained('./models/')
#     # if 't5' in model_name:
#     #     document = "summarize: " + document
#     if action == "summarize":
#         print(f"action is :{action}")
#         if len(data) == 0 :
#             data = default_document
#         document = "summarize: " + data
#         tokenized = tokenizer([document], max_length=512, truncation=True, return_tensors='np')
#         out = model.generate(**tokenized, max_length=128)
#         response = tokenizer.decode(out[0])
#         print(response)
#         #return json.dumps({'summary': tokenizer.decode(out[0])})
#         #return (json.dumps({"my_data": data}))
#     else:
#         response = "Invalid Action"
#     return {"summary": response}


# This main function can be uncommented to test this function on command line. Use this command on terminal after 
# uncommenting main , to test if the model is getting invoked properly
# python3 004_deploy-and-test-models/LLM_inference.py'{"action":"summarize", "document":"this is a test of summarization"}'
#def main():
#     default_json = "{\"action\":\"summarize\", \"document\":\"Vish is trying something\"}"
#     args = sys.argv
#     print (args)
#     if len(args) > 1:
#         input_json = args[1]
#     else :  
#         input_json = default_json
#   
#     print(summarize(input_json))
#
#     #replicating CML Behavior
#     print(summarize(json.loads(input_json)))
#
#
#if __name__ == "__main__":
#     main()
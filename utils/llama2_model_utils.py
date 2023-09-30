from llama_cpp import Llama

## Initialize Llama2 Model on app startup
model_path = "/home/cdsw/models/gen-ai-model/llama-2-13b-chat.ggmlv3.q5_1.bin"

llama2_model = Llama(
    model_path=model_path,
    n_gpu_layers=64,
    n_ctx=2000
)
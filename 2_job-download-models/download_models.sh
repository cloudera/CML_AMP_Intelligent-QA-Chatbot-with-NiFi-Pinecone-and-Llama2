# # This script is used to pre=download files stored with git-lfs in CML Runtimes which do not have git-lfs support
# # You can use any models that can be loaded with the huggingface transformers library. See utils/model_embedding_utls.py or utils/moderl_llm_utils.py

EMBEDDING_MODEL_REPO="https://huggingface.co/sentence-transformers/all-mpnet-base-v2"
EMBDEDDING_MODEL_COMMIT="bd44305fd6a1b43c16baf96765e2ecb20bca8e1d"

download_lfs_files () {
    echo "These files must be downloaded manually since there is no git-lfs here:"
    COMMIT=$1
    git ls-files | git check-attr --stdin filter | awk -F': ' '$3 ~ /lfs/ { print $1}' | while read line; do
        echo "Downloading ${line}"
        echo $(git remote get-url $(git remote))/resolve/$COMMIT/${line}
        curl -O -L $(git remote get-url $(git remote))/resolve/$COMMIT/${line}
        echo "Downloading ${line} completed"
    done
}

GEN_AI_MODEL_URL="https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_0.gguf"
EMBEDDING_MODEL_URL="https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/all-mpnet-base-v2.tar.gz"

# Create the models directory if it doesn't exist
rm -rf ./models
mkdir -p models/gen-ai-model
mkdir -p models/embedding-model

# Download models
echo "Downloading GEN_AI model..."
curl -L -o models/gen-ai-model/llama-2-13b-chat.Q5_0.gguf ${GEN_AI_MODEL_URL} || echo "Failed to download GEN_AI model"

echo "Downloading EMBEDDING model..."
# Downloading model for generating vector embeddings
cd models
GIT_LFS_SKIP_SMUDGE=1 git clone ${EMBEDDING_MODEL_REPO} --branch main embedding-model 
git checkout ${EMBDEDDING_MODEL_COMMIT}
download_lfs_files $EMBDEDDING_MODEL_COMMIT
cd embedding-model
git lfs install
git lfs pull

echo "Model downloads complete."

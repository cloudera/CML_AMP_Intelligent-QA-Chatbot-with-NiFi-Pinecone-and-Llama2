import os
import cmlapi
import random
import string
import json

client = cmlapi.default_client(url=os.getenv("CDSW_API_URL").replace("/api/v1", ""), cml_api_key=os.getenv("CDSW_APIV2_KEY"))

project = client.get_project(project_id=os.getenv("CDSW_PROJECT_ID"))

if os.getenv("USE_CML_MODELS") and os.getenv("USE_CML_MODELS").lower() == "true" or os.getenv("USE_CML_MODELS") == True:
    llama_model = client.list_models(project.id)
    llama_model_build = client.list_model_builds(project.id, llama_model.models[0].id)

    PROJECT_ID = project.id
    MODEL_ID = llama_model_build.model_builds[0].model_id
    BUILD_ID = llama_model_build.model_builds[0].id

    model_deployment_body = cmlapi.CreateModelDeploymentRequest(project_id=PROJECT_ID, model_id=MODEL_ID, build_id=BUILD_ID, cpu=4, memory=16, nvidia_gpus=1, replicas=1)
    deploy_model_request = client.create_model_deployment(model_deployment_body, PROJECT_ID, MODEL_ID, BUILD_ID)
    print(deploy_model_request)
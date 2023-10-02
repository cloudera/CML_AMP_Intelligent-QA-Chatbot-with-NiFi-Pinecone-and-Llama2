import time

import os
import json
import string
import cmlapi
#from src.api import ApiUtility
import cdsw
from datetime import datetime

## Learn more: https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-creating-and-deploying-a-model.html

# lets us get a Handle to API 
client = cmlapi.default_client()
project_id = os.environ["CDSW_PROJECT_ID"]
project = client.get_project(project_id = project_id)

# projects = client.list_projects(search_filter=json.dumps({"name": "CML-LLM-Test"}))
# print(projects)
# project = projects.projects[0] # assuming only one project is returned by the above query


# create a model request
model_body = cmlapi.CreateModelRequest(project_id=project.id, name="llama2-model8", description="Internally Hosted Llama2 Model")
model = client.create_model(model_body, project.id)

# create a model request
runtime_details='docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-jupyterlab-python3.9-cuda:2023.08.2-b8'
model_build_body = cmlapi.CreateModelBuildRequest(project_id=project.id, model_id=model.id, file_path="./deploy_model/build_model.py", function_name="get_llama2_response", kernel="python3", runtime_identifier=runtime_details)

start_time = datetime.now()
print(start_time.strftime("%H:%M:%S"))

# Model is getting Built as a container image
model_build = client.create_model_build(model_build_body, project.id, model.id)
while model_build.status not in ["built", "build failed"]:
    print("waiting for model to build...")
    time.sleep(10)
    model_build = client.get_model_build(project.id, model.id, model_build.id)
    if model_build.status == "build failed" :
        print("model build failed, see UI for more information")
        sys.exit(1)
        
build_time = datetime.now()   
print(f"Time required for building model (sec): {(build_time - start_time).seconds}")
print("model built successfully!")


# Model is getting deployed as a container image
model_deployment_body = cmlapi.CreateModelDeploymentRequest(project_id=project.id, model_id=model.id, build_id=model_build.id, cpu=4, memory=8)
model_deployment = client.create_model_deployment(model_deployment_body, project.id, model.id, model_build.id)

while model_deployment.status not in ["stopped", "failed", "deployed"]:
    print("waiting for model to deploy...")
    time.sleep(10)
    model_deployment = client.get_model_deployment(project.id, model.id, model_build.id, model_deployment.id)

curr_time = datetime.now()

if model_deployment.status != "deployed":
    print("model deployment failed, see UI for more information")
    sys.exit(1)

if model_deployment.status == "deployed" :
    print(f"Time required for deploying model (sec): {(curr_time - start_time).seconds}")
print("model deployed successfully!")
import os
import cmlapi
import random
import string
import json

client = cmlapi.default_client(url=os.getenv("CDSW_API_URL").replace("/api/v1", ""), cml_api_key=os.getenv("CDSW_APIV2_KEY"))
available_runtimes = client.list_runtimes(search_filter=json.dumps({
    "kernel": "Python 3.10",
    "edition": "Nvidia GPU",
    "editor": "JupyterLab"
}))
print(available_runtimes)

## Set available runtimes to the latest runtime in the environment (iterator is the number that begins with 0 and advances sequentially)
## The JOB_IMAGE_ML_RUNTIME variable stores the ML Runtime which will be used to launch the job
print(available_runtimes.runtimes[1])
print(available_runtimes.runtimes[1].image_identifier)
APP_IMAGE_ML_RUNTIME = available_runtimes.runtimes[1].image_identifier

## Store the ML Runtime for any future jobs in an environment variable so we don't have to do this step again
os.environ['APP_IMAGE_ML_RUNTIME'] = APP_IMAGE_ML_RUNTIME
project = client.get_project(project_id=os.getenv("CDSW_PROJECT_ID"))


if os.getenv("USE_CML_MODELS") == "True" or os.getenv("USE_CML_MODELS") == True:
    application_request = cmlapi.CreateApplicationRequest(
         name = "CML LLM Gradio Interface",
         description = "Hosted interface for the CML LLM Gradio UI",
         project_id = project.id,
         subdomain = "cml-llm-interface",
         script = "3_app/llm_app.py",
         cpu = 2,
         memory = 8,
         runtime_identifier = os.getenv('APP_IMAGE_ML_RUNTIME')
    )
    
else:
    application_request = cmlapi.CreateApplicationRequest(
         name = "CML LLM Gradio Interface",
         description = "Hosted interface for the CML LLM Gradio UI",
         project_id = project.id,
         subdomain = "cml-llm-interface",
         script = "3_app/llm_app.py",
         cpu = 4,
         memory = 16,
         nvidia_gpu = 1,
         runtime_identifier = os.getenv('APP_IMAGE_ML_RUNTIME')
    )

app = client.create_application(
     project_id = project.id,
     body = application_request
)
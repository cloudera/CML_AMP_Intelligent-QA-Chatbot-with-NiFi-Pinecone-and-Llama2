# CDF to CML with LLama2 models
This AMP contains the files to host an Open Source Llama2-based model and an accompanying UI and API. This AMP enables organizations to deploy a custom chatbot, currated to data scraped from a website (or websites) sitemap(s) using CDF (NiFi).

![](/assets/catalog-entry.png)

ARCHITECTURE DIAGRAM HERE

## Building your custom knowledge base
To build your own custom knowledge base, you will want to follow the instructions [here](USER_START_HERE/Build_Your_Own_Knowledge_Base_Tools/README.md) in the folder `USER_START_HERE`. There are guides for a Cloudera DataFlow and Pythonic implementation of how to do this. Then, you will want to rerun the `Populate Vector DB` Job to ensure your vector DB has the latest embeddings.

## Two Flavors: UI (Front end) and API
This project allows you to access the context-driven LLM using two flavors: a UI and an API. Both are listed as Applications in CML which you can spin up/down as needed. 

### UI (Front end)

This is the default application choice for the AMP. You should be able to access the view through your applications nav. When it starts, you will be able to select the default model (`llama-2-13b-chat`), temperature (a good default is 1), number of tokens (a good default may be 100), topic weight (a domain for the corpus of knowledge to prioritize), and question for the model to process. Defaults will be selected if you choose not to answer these; however a question is required.

![](/assets/interface.png)

### REST API

#### Forming a request to the API
Requests can be formed intra-domain or cross-domain. For cross-domain requests, you'll need to ensure unauthenticated app access is allowed for the POST endpoint to be reachable. Be cognizant of the amount of tokens and temperature you feed into the payload parameters. Most requests for a couple sentences should use around 200 tokens, a paragraph could use upwards of 600-800. 

![](/assets/unauthenticated-access1.png)

Note that if this option does not appear, you may need to update your Administrative security settings to mirror the below, where "Allow applications to be configured with unauthenticated access" is checked:

![](/assets/unauthenticated-access2.png)

For request syntax: 
GET and POST to the ROOT of the CML application endpoint (e.g. `https://subdomain.domain.go01-dem.ylcu-atmi.cloudera.site/`)
Successful GET request should indicate the API is up and running:

![](/assets/GET-endpoint.png)

Forming the POST request can be done through Postman or natively in CML:

1. Postman

Form the payload/url and body to match the below, and add the header `Content-Type | application/json`

```
{
    "inputs": "What is Cloudera Data Science Workbench?",
    "parameters": {
        "temperature": 1,
        "max_tokens": 100
    }
}
```

![](/assets/postman-setup.png)

Note that in future development, `engine` may also be customized to include more than the Llama2 one which comes with the AMP deployment.

2. Pythonic (Available in the 4_app folder as an Jupyter notebook)

### The Fine Print

DISCUSS HOW IT WORKS
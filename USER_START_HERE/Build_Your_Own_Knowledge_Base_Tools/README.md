# Scripts/Utilities to Build a Knowledge Base
Scripts and utilities to build a custom knowledge base for your organizational needs. 

## NiFi / DataFlow version (Preferred)
Import the JSON configuration into your CFM or NiFi cluster. This file contains all the relevant context to store the result folder, sub-folders, and files into a /tmp directory in hdfs which can be copied into your CML project using hdfs connector / commands.

### Implementation / Setup Instructions
This can be completed using the Cloudera DataFlow data service, or using a flavor of NiFi within a Data Hub. The steps below are specifically to complete the task using DataFlow; however it is largely the same for a Data Hub implementation as well.

1. Download the Json file from `/USER_START_HERE/Build_Your_Own_Knowledge_Base_Tools/NiFi-based_sitemap_scrape/pubsecml-version-1.json` and import it into your catalog.

2. Setup in a manner similar to below screenshots:

![](/assets/build_your_own_kb_screenshots/nifi_implementation/setup-step1.png)

![](/assets/build_your_own_kb_screenshots/nifi_implementation/setup-step2.png)

![](/assets/build_your_own_kb_screenshots/nifi_implementation/setup-step3.png)

![](/assets/build_your_own_kb_screenshots/nifi_implementation/setup-step4.png)

![](/assets/build_your_own_kb_screenshots/nifi_implementation/setup-step5.png)

![](/assets/build_your_own_kb_screenshots/nifi_implementation/deploy-step1.png)

![](/assets/build_your_own_kb_screenshots/nifi_implementation/deploy-step2.png)

![](/assets/build_your_own_kb_screenshots/nifi_implementation/deploy-step3.png)

![](/assets/build_your_own_kb_screenshots/nifi_implementation/deploy-step4.png)

![](/assets/build_your_own_kb_screenshots/nifi_implementation/deploy-step5.png)

3. Lastly, once your processors are up and running, and accepting HTTP POST Requests, you can use Postman (or `SamplePOSTRequestToNiFi.ipynb` within a session) to send your first sitemap to the flow. You can use the endpoint hostname and listening port on the last page of your deployment to form your POST request.

Note, this implementation stores the downloaded files into a /tmp directory in hdfs. You will need to ensure you run the appropriate `hdfs fs -get <source> <destination>` or `hdfs fs -copyToLocal <source> <destination>` (or an S3 command equivalent) to get these into your `/data` directory in CML. Refer to https://docs.cloudera.com/machine-learning/cloud/import-data/topics/ml-accessing-data-from-hdfs.html for additional code and information.



## Python version
This version is self contained to the CML environment. Depending on the size of your sitemap file and website, you may prefer to take advantage of NiFi's parallel processing capabilities, which will steeply decrease the amount of time it takes to populate your knowledge base.

### Implementation / Setup Instructions
Note: Executing this script requires you open a session. You do not need a GPU resource to complete this using Python; however, your session should have at least 4 CPUs and 16 GB RAM available to complete this task. Use the configuration file to set variables and the requirements.txt file for dependency resolution.

1. From within your session, install the requirements at `/USER_START_HERE/Build_Your_Own_Knowledge_Base_Tools/Python-based_sitemap_scrape/requirements.txt` using ` pip install -r <path to requirements.txt>`.

2. Next, you will update the `cloudera_kb_config.conf` with the root of the website you want to scrape's sitemap. The XML namespace is typically somewhere within the header of the XML and this schema allows the python files to derive the values of your website's sitemap implementation.

3. Change directory to `Python-based_sitemap_scrape` and execute the first python script: `python 1_kb_xml_scrape.py` (Depending on your editor/aliases, you may need to use `python3 1_kb_xml_scrape.py`). After this code runs, you will have a file which contains all the html derivatives of your root and nested sitemap xmls. This will be used for the fourth step.

4. After this completes, execute `python 2_kb_html_to_text.py` to download and store the knowledge base to your `/data`. You will need to rerun the job `Populate Vector DB` once this completes.

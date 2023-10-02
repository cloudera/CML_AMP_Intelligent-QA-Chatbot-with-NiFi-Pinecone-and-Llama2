# Copyright 2023 Cloudera, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# XML PARSER AND WEB SCRAPER FOR INTERNAL LLM KNOWLEDGE BASE
# ASSUMES REQUISITE URLS AND SITE MAP SCHEMA ARE CONFIGURED IN
# cloudera_kb_config.conf FILE AND APPROPRIATE DEPENDENCIES
# ARE INSTALLED FROM PIP. 

# THE OUTPUT OF THIS PROCESS WILL STORE HTMLS IN /data WITH THE NAMING
# CONVENTION 1_htmls.txt FOR PARSING BY THE NEXT UTILITY

import requests
import configparser
from xml.etree import ElementTree as ET

config = configparser.ConfigParser()
config.read('cloudera_kb_config.conf')

def fetch_and_parse_xml(url):
    try:
        if config['DEFAULT']['xml_namespace'] == "yes":
            response = requests.get(url)
        else:
            response = requests.get(url, verify=False)

        if response.status_code == 200:
            return ET.fromstring(response.content)
        else:
            print(f"Failed to fetch URL: {url}")
            return None
    except Exception as e:
        print(f"Error fetching URL: {url}. Exception: {e}")
        return None

def extract_urls_and_scan(root, xml_namespace):
    xml_urls = []
    for elem in root.findall(f".//{{{xml_namespace}}}loc"):
        try:
            url = elem.text
            if url.endswith('.html'):
                with open("found_htmls.txt", "a") as file:
                    file.write(elem.text + "\n")
                continue  # Skip to the next iteration
            elif url.endswith('.xml'):
                print(f"XML found: {url}")
                xml_urls.append(url)
                res = fetch_and_parse_xml(url)
                extract_urls_and_scan(res, xml_namespace)
        except:
            with open("htmls/found_errors.txt", "a") as file:
                file.write(f"Error with XML at: {elem.text}\n")
    return xml_urls


def main():
    start_url = config['DEFAULT']['root_site_sitemap']
    xml_namespace = config['DEFAULT']['xml_namespace']
    root = fetch_and_parse_xml(start_url)
    extract_urls_and_scan(root, xml_namespace)


if __name__ == "__main__":
    main()

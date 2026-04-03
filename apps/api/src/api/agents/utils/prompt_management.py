
import yaml
from jinja2 import Template
from langsmith import Client

ls_client = Client()


def prompt_template_config(yaml_file,prompt_key):

    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    template_content=config["prompts"][prompt_key]

    template=Template(template_content)
    return template


def prompt_template_registry( prompt_name):

    template_content=ls_client.pull_prompt(prompt_name).messages[0].prompt.template
    template=Template(template_content)
    return template
import os

from autogen import config_list_from_json, AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

CONFIG_FILE_NAME: str = "config.json"
ABSOLUTE_DEFAULT_CONFIG_PATH: str = os.path.join(os.path.dirname(__file__), CONFIG_FILE_NAME)

def termination_msg(x):
    return isinstance(x, dict) and ("TERMINATE" == str(x.get("content", ""))[-9:].upper() or "TERMINATE." == str(x.get("content", ""))[-10:].upper())

def get_llm_config():
    config_list = config_list_from_json(env_or_file=ABSOLUTE_DEFAULT_CONFIG_PATH)
    llm_config = {
        "config_list": config_list,
    }
    return llm_config

def create_assistant_agent(*, name: str, system_message: str, description: str):
    return AssistantAgent(
        name=name,
        system_message=system_message,
        default_auto_reply="Reply `TERMINATE` if the task is done.",
        llm_config=get_llm_config(),
        description=description
    )

def create_user_proxy_agent(*, name: str, description: str):
    return UserProxyAgent(
        name=name,
        is_termination_msg=termination_msg,
        human_input_mode="NEVER",
        default_auto_reply="Reply `TERMINATE` if the task is done.",
        code_execution_config=False,
        description=description
    )

def create_retrieve_user_proxy_agent(*, name:str, retrieve_config: dict, description: str):
    return RetrieveUserProxyAgent(
    name=name,
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    max_consecutive_auto_reply=3,
    retrieve_config=retrieve_config,
    code_execution_config=False,
    description=description
)
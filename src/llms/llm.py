# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

<<<<<<< HEAD
from pathlib import Path  
from typing import Any, Dict, Union  
  
from langchain_openai import ChatOpenAI, AzureChatOpenAI  
  
from src.config import load_yaml_config  
from src.config.agents import LLMType  
  
# Cache for LLM instances - use union to support both ChatOpenAI and AzureChatOpenAI  
_llm_cache: dict[LLMType, Union[ChatOpenAI, AzureChatOpenAI]] = {}  
  
  
def _create_llm_use_conf(llm_type: LLMType, conf: Dict[str, Any]) -> Union[ChatOpenAI, AzureChatOpenAI]:  
    llm_type_map = {  
        "reasoning": conf.get("REASONING_MODEL"),  
        "basic": conf.get("BASIC_MODEL"),  
        "vision": conf.get("VISION_MODEL"),  
    }  
    llm_conf = llm_type_map.get(llm_type)  
    if not llm_conf:  
        raise ValueError(f"Unknown LLM type: {llm_type}")  
    if not isinstance(llm_conf, dict):  
        raise ValueError(f"Invalid LLM Conf: {llm_type}")  
  
    # handle Azure specific configurations
    if llm_conf.get("use_azure"):  
        azure_config = {  
            "azure_endpoint": llm_conf.get("api_base"),  
            "azure_deployment": llm_conf.get("deployment_name"),  
            "api_version": llm_conf.get("api_version"),  
            "api_key": llm_conf.get("api_key"),  
        }  
        # remove azure specific keys from llm_conf  
        config_keys = ["use_azure", "api_base", "deployment_name", "api_version", "api_key"]  
        cleaned_conf = {k: v for k, v in llm_conf.items() if k not in config_keys}  
        # combine cleaned_conf with azure_config
        final_conf = {**cleaned_conf, **azure_config}  
        return AzureChatOpenAI(**final_conf)  # 使用AzureChatOpenAI  
    else:  
        # use default ChatOpenAI  
        return ChatOpenAI(**llm_conf)  # 使用ChatOpenAI  
  
  
def get_llm_by_type(  
    llm_type: LLMType,  
) -> Union[ChatOpenAI, AzureChatOpenAI]:  
    """  
    Get LLM instance by type. Returns cached instance if available.  
    """  
    if llm_type in _llm_cache:  
        return _llm_cache[llm_type]  
  
    conf = load_yaml_config(  
        str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())  
    )  
    llm = _create_llm_use_conf(llm_type, conf)  
    _llm_cache[llm_type] = llm  
    return llm  
  
  
# Initialize LLMs for different purposes - now these will be cached  
basic_llm = get_llm_by_type("basic")  
  
# In the future, we will use reasoning_llm and vl_llm for different purposes  
# reasoning_llm = get_llm_by_type("reasoning")  
# vl_llm = get_llm_by_type("vision")  
  
  
if __name__ == "__main__":  
    print(basic_llm.invoke("Hello"))
=======
from pathlib import Path
from typing import Any, Dict
import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from src.config import load_yaml_config
from src.config.agents import LLMType

# Cache for LLM instances
_llm_cache: Dict[LLMType, BaseChatModel] = {}


def _get_env_llm_conf(llm_type: str) -> Dict[str, Any]:
    """
    Get LLM configuration from environment variables.
    Environment variables should follow the format: {LLM_TYPE}__{KEY}
    e.g., BASIC_MODEL__api_key, BASIC_MODEL__base_url
    """
    prefix = f"{llm_type.upper()}_MODEL__"
    conf = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            conf_key = key[len(prefix) :].lower()
            conf[conf_key] = value
    return conf


def _create_llm_use_conf(llm_type: LLMType, conf: Dict[str, Any]) -> BaseChatModel:
    """
    Create LLM instance using configuration for given type.
    Supports both OpenAI and Azure OpenAI models.
    """
    llm_type_map = {
        "reasoning": conf.get("REASONING_MODEL", {}),
        "basic": conf.get("BASIC_MODEL", {}),
        "vision": conf.get("VISION_MODEL", {}),
    }
    llm_conf = llm_type_map.get(llm_type)
    if not llm_conf:
        raise ValueError(f"Unknown LLM type: {llm_type}")
    if not isinstance(llm_conf, dict):
        raise ValueError(f"Invalid LLM Conf: {llm_type}")
    return ChatOpenAI(**llm_conf)


def get_llm_by_type(
    llm_type: LLMType,
) -> BaseChatModel:
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    conf = load_yaml_config(
        str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())
    )
    llm = _create_llm_use_conf(llm_type, conf)
    _llm_cache[llm_type] = llm
    return llm


# In the future, we will use reasoning_llm and vl_llm for different purposes
# reasoning_llm = get_llm_by_type("reasoning")
# vl_llm = get_llm_by_type("vision")


if __name__ == "__main__":
    # Initialize LLMs for different purposes - now these will be cached
    basic_llm = get_llm_by_type("basic")
    print(basic_llm.invoke("Hello"))

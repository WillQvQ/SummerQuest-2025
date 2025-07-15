import json
from qwen_agent.llm import get_chat_model

def get_current_temperature(location: str, unit: str = "celsius"):
    """Get current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, and the unit in a dict
    """
    return {
        "temperature": 26.1,
        "location": location,
        "unit": unit,
    }


def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    """Get temperature at a location and date.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        date: The date to get the temperature for, in the format "Year-Month-Day".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, the date and the unit in a dict
    """
    return {
        "temperature": 25.9,
        "location": location,
        "date": date,
        "unit": unit,
    }


def get_function_by_name(name):
    if name == "get_current_temperature":
        return get_current_temperature
    if name == "get_temperature_date":
        return get_temperature_date

TOOLS = [
    {
        "type": "function", 
        "function": {
            "name": "search", 
            "description": "搜索引擎，在需要实时信息的时候需要调用此工具", 
            "parameters": {
                "type": "object", 
                "properties": {
                    "keyword": {"type": "string", "description": "使用搜索引擎所需的关键词"}, 
                    "top_k": {"type": "number", "default": 3, "description": "返回的搜索结果数量"}
                }, 
        "required": ["keyword"]}
        }
    }
]

MESSAGES = [
    {"role": "user",  "content": "What's the temperature in San Francisco now? How about tomorrow? Current Date: 2024-09-30."},
]

llm = get_chat_model({
    "model": "/remote-home1/share/models/Qwen2.5-0.5B-Instruct",
    "model_server": "http://localhost:8000/v1",
    "api_key": "EMPTY",
    "generate_cfg": {
      "extra_body": {
        "chat_template_kwargs": {"enable_thinking": False}  # default to True
      }
    }
})
import pdb; pdb.set_trace()  # Debugging line to inspect the llm object
messages = MESSAGES[:]
functions = [tool["function"] for tool in TOOLS]
for responses in llm.chat(
    messages=messages,
    functions=functions,
):
    print(f"Response: {responses}")
messages.extend(responses)

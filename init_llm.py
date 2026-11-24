"""
init_chat_model 初始化聊天模型
"""
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from env_utils import DEEPSEEK_API_KEY, GLM_BASE_URL, GLM_API_KEY, DEEPSEEK_BASE_URL

deepseek_llm_flash: BaseChatModel = init_chat_model(
    model="deepseek-v4.1-flash",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    extra_body={
        "thinking": {"type": "disabled"}  # 关闭思考模式
    }
)

glm_llm: BaseChatModel = init_chat_model(
    "glm-5.3-flash",
    model_provider="openai",
    temperature=1.0,
    base_url=GLM_BASE_URL,
    api_key=GLM_API_KEY,
    profile={
        "max_input_tokens": 128000,
        "max_output_tokens": 8192,
        "tool_calling": True,
        "structured_output": True,
    }
)
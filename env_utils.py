import os

from dotenv import load_dotenv

# override=True 确保.env文件优先
load_dotenv(override=True)

# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GLM_API_KEY = os.getenv("GLM_API_KEY")
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")


DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
GLM_BASE_URL = os.getenv("GLM_BASE_URL")


DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

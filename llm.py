from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,LLM_MAX_RETRIES,LLM_TIMEOUT

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout = LLM_TIMEOUT, max_retries = LLM_MAX_RETRIES)

def chat(messages: list[dict], temperature: float = 0.2) -> str:
    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return res.choices[0].message.content
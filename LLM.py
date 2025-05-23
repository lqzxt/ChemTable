from openai import OpenAI
from openai import APIError, APIConnectionError
from zhipuai import ZhipuAI


def call_LLM(mes, model_name="gpt-4.1-2025-04-14", temperature=0, try_limit=3):
    url = "YOUR_OPENAI_API_ENDPOINT"
    key = "YOUR_OPENAI_API_KEY"
    if model_name != "gpt-4.1-2025-04-14" and model_name != "gpt-4.1-mini-2025-04-14" and model_name != "gpt-4.1-nano-2025-04-14":
        url = "YOUR_ALTERNATIVE_API_ENDPOINT"
        key = "YOUR_ALTERNATIVE_API_KEY"

    if model_name == "qwen2.5-vl-72b-instruct":
        url = "YOUR_ALTERNATIVE_API_ENDPOINT"
        key = "YOUR_QWEN_API_KEY"

    if model_name == "intern_vl":
        url = "YOUR_ALTERNATIVE_API_ENDPOINT"
        key = "YOUR_QWEN_API_KEY"

    if model_name == "glm-4v-plus":
        client = ZhipuAI(api_key="YOUR_ZHIPU_API_KEY")
    else:
        client = OpenAI(
            base_url=url,
            api_key=key,
            timeout=90
        )
    while try_limit > 0:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=mes,
                temperature=temperature,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except (APIError, APIConnectionError) as e:
            print(f"API Error: {e}")
            try_limit -= 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            try_limit -= 1
    return "Error!!!"


def call_qwen_llm(mes, model_name="qwen2.5-vl-7b-instruct", temperature=0, try_limit=3):
    url = "YOUR_ALTERNATIVE_API_ENDPOINT"
    key = "YOUR_QWEN_API_KEY"
    client = OpenAI(
        api_key=key,
        base_url=url,
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=mes,
        temperature=temperature,
        max_tokens=2048
    )
    return completion.choices[0].message.content

"""OpenRouter API client."""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def chat(
    messages: list[dict],
    model: str = "anthropic/claude-haiku-4-5",
    temperature: float | None = None,
    max_tokens: int = 4096,
    retries: int = 3,
) -> tuple[str, dict]:
    """Call the OpenRouter chat API. Returns (response_text, usage_dict)."""
    client = get_client()
    for attempt in range(retries):
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "model": model,
            }
            return resp.choices[0].message.content, usage
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("All retries exhausted")

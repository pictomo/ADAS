import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm_provider import client

if __name__ == "__main__":
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            reasoning_effort="minimal",
            # temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Reply in JSON format.",
                },
                {
                    "role": "user",
                    "content": 'Give me a random word. Return as JSON like {"word": "..."}',
                },
            ],
            max_completion_tokens=1024,
            response_format={"type": "json_object"},
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

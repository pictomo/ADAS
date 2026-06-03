import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm_provider import client

if __name__ == "__main__":
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'API is working' in one sentence."}],
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

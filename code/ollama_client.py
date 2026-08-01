import ollama
import json
import re


MODEL = "gemma4"


def ask_llm(prompt: str):

    response = ollama.chat(
    model=MODEL,
    options={
        "temperature": 0.2
    },
    messages=[
        {
            "role": "system",
            "content": (
                "You are a WhatsApp Notification Router.\n"
                "Return ONLY valid JSON.\n"
                "No markdown.\n"
                "No explanations.\n"
                "No thinking.\n"
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)

    text = response["message"]["content"]

    # Extract JSON even if model adds extra text
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise Exception("No JSON found:\n" + text)

    return json.loads(match.group())
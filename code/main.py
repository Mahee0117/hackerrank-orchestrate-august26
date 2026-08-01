from loader import load_data
from ollama_client import ask_llm

data = load_data()

messages = data["messages"]

# First message only
msg = messages.iloc[0]

print("\n===== FIRST MESSAGE =====\n")

print(msg["message_text"])

prompt = f"""
You are an AI system that routes WhatsApp notifications.

You must classify ONE incoming WhatsApp message.

Allowed action values ONLY:
- notify
- digest
- mute

Allowed message_type values ONLY:
- personal
- urgent
- event
- payment
- business_update
- promotion
- greeting
- forward
- spam
- scam
- unknown

Return ONLY JSON.

Incoming Message:

{msg["message_text"]}

Return:

{{
    "action":"",
    "message_type":"",
    "reason":"",
    "confidence":0.0
}}
"""

result = ask_llm(prompt)

print("\n===== AI RESULT =====\n")

print(result)
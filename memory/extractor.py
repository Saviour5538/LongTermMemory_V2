import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EXTRACTION_SYSTEM_PROMPT = """
You are a memory extraction assistant.

Your job is to read a conversation and extract important FACTS about the USER worth remembering long-term.

Rules:
- Extract ONLY facts about the user (name, preferences, goals, habits, background, changes in habits, etc.)
- IMPORTANT: Also extract when a user says they STOPPED or CHANGED something (e.g. "I quit chess" is a fact)
- Each fact must be a short, clear, standalone sentence in third person ("User's name is Adarsh")
- Do NOT extract facts about the assistant
- Do NOT include general opinions or knowledge, only personal user facts
- If there are no meaningful facts, return an empty list []
- Do NOT extract past/historical facts (e.g. "User used to...", "User had..." or things the user says they no longer do)
- Only extract CURRENT, present-tense facts about the user

You MUST respond with ONLY a valid JSON array of strings. Nothing else.
No explanation. No markdown. No extra text.

Example output:
["User's name is Adarsh", "User switched from chess to guitar", "User has been learning guitar for 2 months"]
"""


def extract_facts(conversation: list) -> list:
    """
    Takes a conversation (list of message dicts) and returns extracted facts.
    """
    conversation_text = ""
    for message in conversation:
        role    = message["role"].capitalize()
        content = message["content"]
        conversation_text += f"{role}: {content}\n"

    print("[EXTRACTOR] Extracting facts from conversation...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # Stronger model = better extraction
        temperature=0,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Extract facts from this conversation:\n\n{conversation_text}"}
        ]
    )

    raw_output = response.choices[0].message.content.strip()
    print(f"[EXTRACTOR] Raw output: {raw_output}")

    try:
        facts = json.loads(raw_output)
        if not isinstance(facts, list):
            print("[EXTRACTOR] Warning: LLM did not return a list.")
            return []
        facts = [f for f in facts if isinstance(f, str) and f.strip()]
        print(f"[EXTRACTOR] Extracted {len(facts)} facts: {facts}")
        return facts
    except json.JSONDecodeError:
        print(f"[EXTRACTOR] JSON parse error. Got: {raw_output}")
        return []
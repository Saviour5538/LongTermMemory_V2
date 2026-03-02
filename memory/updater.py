import os
import json
import traceback
from groq import Groq
from dotenv import load_dotenv
from memory.vector_store import (
    add_memory,
    update_memory,
    delete_memory,
    search_similar_memories
)

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

UPDATE_SYSTEM_PROMPT = """
You are a memory management assistant keeping a user's memory store accurate.

You will be given:
1. A NEW FACT about the user
2. EXISTING SIMILAR MEMORIES (each with an exact ID)

Your job: choose exactly ONE action:

╔══════════╦══════════════════════════════════════════════════════════════════╗
║ Action   ║ When to use                                                      ║
╠══════════╬══════════════════════════════════════════════════════════════════╣
║ ADD      ║ New fact is completely new — not in any existing memory          ║
║ UPDATE   ║ New fact refines/changes an existing memory (prefer over DELETE) ║
║ DELETE   ║ New fact directly contradicts an existing memory (use UPDATE if  ║
║          ║ possible, DELETE only if the old memory is now completely false) ║
║ NOOP     ║ The new fact is already fully captured in an existing memory     ║
╚══════════╩══════════════════════════════════════════════════════════════════╝

CRITICAL RULES:
1. For NOOP: only use if the existing memory says EXACTLY the same thing.
2. If memory says "User likes chess" and new fact says "User now plays guitar instead of chess":
   → UPDATE the chess memory to "User plays guitar" (do NOT add a duplicate)
3. If memory says "User likes chess" and new fact says "User quit chess entirely":
   → DELETE that memory
4. NEVER add a fact that already exists with same meaning → use NOOP
5. You MUST use the EXACT memory_id from the list provided — copy it character for character.

Respond ONLY with valid JSON. No explanation. No markdown.

For ADD:    {"action": "ADD"}
For UPDATE: {"action": "UPDATE", "memory_id": "<exact id from list>", "new_text": "<merged updated fact>"}
For DELETE: {"action": "DELETE", "memory_id": "<exact id from list>"}
For NOOP:   {"action": "NOOP", "reason": "<why it already exists>"}
"""


def process_single_fact(fact: str) -> dict:
    print(f"\n[UPDATER] Processing: '{fact}'")

    try:
        # ── Step 1: Retrieve similar memories ─────────────────────────────
        similar_memories = search_similar_memories(fact, top_s=5)
        valid_ids = {m["id"] for m in similar_memories}

        print(f"[UPDATER] Found {len(similar_memories)} similar memories.")

        # ── Step 2: Build prompt — show EXACT IDs to LLM ──────────────────
        if similar_memories:
            memories_text = ""
            for mem in similar_memories:
                # Show full ID so LLM can copy it exactly
                memories_text += (
                    f"  ID: {mem['id']}\n"
                    f"  Text: {mem['text']}\n"
                    f"  Similarity: {mem['score']:.4f}\n\n"
                )
        else:
            memories_text = "  (No similar memories found — this is new information)"

        user_prompt = (
            f"NEW FACT:\n{fact}\n\n"
            f"EXISTING SIMILAR MEMORIES:\n{memories_text}"
            f"Decide the action. If UPDATE or DELETE, use an EXACT ID from the list above."
        )

        # ── Step 3: LLM decision ───────────────────────────────────────────
        print("[UPDATER] Asking LLM to decide action...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # Stronger model = better decisions
            temperature=0,
            messages=[
                {"role": "system", "content": UPDATE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ]
        )

        raw_output = response.choices[0].message.content.strip()
        print(f"[UPDATER] LLM raw decision: {raw_output}")

        # ── Step 4: Parse decision ─────────────────────────────────────────
        # Strip markdown fences if LLM added them despite instructions
        if raw_output.startswith("```"):
            raw_output = raw_output.split("```")[1]
            if raw_output.startswith("json"):
                raw_output = raw_output[4:]
            raw_output = raw_output.strip()

        try:
            decision = json.loads(raw_output)
        except json.JSONDecodeError:
            print(f"[UPDATER] JSON parse failed. Defaulting to ADD.")
            add_memory(fact)
            return {"action": "ADD", "fallback": True}

        action = decision.get("action", "").upper()

        # ── Step 5: Execute — with ID validation ───────────────────────────
        if action == "ADD":
            add_memory(fact)
            print(f"[UPDATER] ✅ ADD — stored new memory.")

        elif action == "UPDATE":
            memory_id = decision.get("memory_id", "").strip()
            new_text  = decision.get("new_text", fact).strip()

            # Validate ID — LLMs sometimes hallucinate or truncate IDs
            if memory_id in valid_ids:
                update_memory(memory_id, new_text)
                print(f"[UPDATER] ✅ UPDATE — memory {memory_id[:8]}... → '{new_text}'")
            elif similar_memories:
                # Fallback: use top-scoring memory
                best = similar_memories[0]
                print(f"[UPDATER] ⚠️  ID mismatch — using top-match fallback: {best['id'][:8]}...")
                update_memory(best["id"], new_text)
            else:
                print(f"[UPDATER] ⚠️  UPDATE but no valid ID and no similar memories — adding as new.")
                add_memory(fact)

        elif action == "DELETE":
            memory_id = decision.get("memory_id", "").strip()

            if memory_id in valid_ids:
                delete_memory(memory_id)
                print(f"[UPDATER] ✅ DELETE — memory {memory_id[:8]}... removed.")
            elif similar_memories:
                # Fallback: delete top-scoring memory
                best = similar_memories[0]
                print(f"[UPDATER] ⚠️  ID mismatch — deleting top-match: {best['id'][:8]}...")
                delete_memory(best["id"])
            else:
                print(f"[UPDATER] ⚠️  DELETE but no valid target. Skipping.")

        elif action == "NOOP":
            reason = decision.get("reason", "already exists")
            print(f"[UPDATER] ⏭️  NOOP — {reason}")

        else:
            print(f"[UPDATER] ⚠️  Unknown action '{action}'. Adding as new.")
            add_memory(fact)

        return decision

    except Exception as e:
        print(f"[UPDATER] ERROR: {e}")
        traceback.print_exc()
        return {"action": "ERROR"}


def process_all_facts(facts: list) -> list:
    if not facts:
        print("[UPDATER] No facts to process.")
        return []

    print(f"\n[UPDATER] Processing {len(facts)} facts...")
    results = []

    for i, fact in enumerate(facts):
        print(f"\n── Fact {i+1}/{len(facts)} {'─'*40}")
        result = process_single_fact(fact)
        results.append({"fact": fact, "decision": result})

    print("\n[UPDATER] Update phase complete.")
    return results
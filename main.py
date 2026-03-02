# main.py — CLI test runner for the memory pipeline

from memory.extractor import extract_facts
from memory.updater import process_all_facts
from memory.vector_store import get_all_memories


def run_memory_pipeline(conversation: list, label: str = ""):
    print("\n" + "═" * 60)
    if label:
        print(f"  {label}")
    print("  LONG TERM MEMORY PIPELINE")
    print("═" * 60)

    # Step 1: Extract
    print("\n📥 STEP 1: Extraction Phase")
    facts = extract_facts(conversation)

    if not facts:
        print("  No facts extracted. Pipeline done.")
        return

    print(f"\n✅ Candidate Facts Ω = {{w₁...w{len(facts)}}}:")
    for i, f in enumerate(facts):
        print(f"   w{i+1}: {f}")

    # Step 2: Update
    print("\n🔄 STEP 2: Update Phase")
    results = process_all_facts(facts)

    # Step 3: Show summary
    print("\n📊 ACTIONS TAKEN:")
    for r in results:
        action = r["decision"].get("action", "?")
        print(f"   [{action}] {r['fact']}")

    # Step 4: Show memory state
    print("\n🧠 MEMORY STORE (current state):")
    all_mems = get_all_memories()
    if not all_mems:
        print("   (empty)")
    else:
        for i, m in enumerate(all_mems):
            print(f"   [{i+1}] {m['text']}")
            print(f"        id: {m['id']}")

    print("\n" + "═" * 60 + "\n")


if __name__ == "__main__":

    # ── TEST 1: New info → expect ADD ──────────────────────────────────────
    run_memory_pipeline(
        label="TEST 1: New information (expect ADD)",
        conversation=[
            {"role": "user",      "content": "Hi! I'm Adarsh. I study computer science and love playing chess. I also hike on weekends."},
            {"role": "assistant", "content": "Nice to meet you Adarsh! Chess and hiking are great hobbies."}
        ]
    )

    # ── TEST 2: Changed info → expect UPDATE/DELETE ────────────────────────
    run_memory_pipeline(
        label="TEST 2: Changed hobby (expect UPDATE or DELETE+ADD)",
        conversation=[
            {"role": "user",      "content": "Actually I switched from chess to playing guitar 2 months ago. I quit chess completely."},
            {"role": "assistant", "content": "That's awesome! Guitar is a wonderful instrument."}
        ]
    )

    # ── TEST 3: Same info → expect NOOP ───────────────────────────────────
    run_memory_pipeline(
        label="TEST 3: Repeated info (expect NOOP)",
        conversation=[
            {"role": "user",      "content": "Just so you know, I'm Adarsh and I'm a CS student."},
            {"role": "assistant", "content": "Yes, I remember you Adarsh!"}
        ]
    )

    # ── TEST 4: New unrelated info → expect ADD ────────────────────────────
    run_memory_pipeline(
        label="TEST 4: New unrelated fact (expect ADD)",
        conversation=[
            {"role": "user",      "content": "I recently started learning Spanish too."},
            {"role": "assistant", "content": "Impressive! Learning a new language is very rewarding."}
        ]
    )
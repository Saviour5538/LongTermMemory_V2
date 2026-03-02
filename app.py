# app.py

import os
import json
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from memory.extractor import extract_facts
from memory.updater import process_all_facts
from memory.vector_store import (
    get_all_memories,
    search_similar_memories,
    clear_all_memories
)

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Long Term Memory", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; }
    .fact-box {
        background: #1a1a2e; border-left: 4px solid #7c3aed;
        padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;
        color: #e2e8f0; font-size: 14px;
    }
    .memory-box {
        background: #1a1a2e; border-left: 4px solid #22c55e;
        padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;
        color: #e2e8f0; font-size: 13px;
    }
    .action-ADD    { background:#14532d44; border:2px solid #22c55e; padding:10px; border-radius:8px; text-align:center; color:#22c55e; font-size:18px; font-weight:bold; }
    .action-UPDATE { background:#78350f44; border:2px solid #f59e0b; padding:10px; border-radius:8px; text-align:center; color:#f59e0b; font-size:18px; font-weight:bold; }
    .action-DELETE { background:#7f1d1d44; border:2px solid #ef4444; padding:10px; border-radius:8px; text-align:center; color:#ef4444; font-size:18px; font-weight:bold; }
    .action-NOOP   { background:#1f1f2e44; border:2px solid #6b7280; padding:10px; border-radius:8px; text-align:center; color:#6b7280; font-size:18px; font-weight:bold; }
    .chat-user      { background:#1e1b4b; padding:12px 16px; border-radius:12px 12px 12px 0; margin:8px 0; color:#c4b5fd; font-size:14px; }
    .chat-assistant { background:#1e293b; padding:12px 16px; border-radius:12px 12px 0 12px; margin:8px 0 8px auto; color:#94a3b8; font-size:14px; }
    .memory-context { background:#0f2a1a; border:1px solid #22c55e33; padding:8px 12px; border-radius:6px; color:#4ade80; font-size:12px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = []


# ── Assistant with memory injection ───────────────────────────────────────
def generate_assistant_response(user_message: str, chat_history: list) -> tuple[str, list]:
    """
    Retrieves relevant memories and injects them into the system prompt
    so the assistant actually USES the long-term memory.
    Returns (response_text, relevant_memories_used)
    """
    # Retrieve memories relevant to user's current message
    relevant_memories = search_similar_memories(user_message, top_s=5)

    memory_context = ""
    if relevant_memories:
        memory_lines = "\n".join([f"- {m['text']}" for m in relevant_memories])
        memory_context = f"\n\nWhat you already know about this user (from memory):\n{memory_lines}\n\nUse this to personalize your response naturally."

    messages = [
        {
            "role": "system",
            "content": f"You are a helpful, friendly assistant with long-term memory about the user. Have a natural conversation.{memory_context}"
        }
    ]

    # Last 6 turns of chat history for short-term context
    for turn in chat_history[-6:]:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": user_message})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        messages=messages
    )
    return response.choices[0].message.content.strip(), relevant_memories


# ── Memory pipeline with UI ────────────────────────────────────────────────
def run_pipeline_with_ui(user_msg: str, assistant_msg: str) -> dict:
    """
    Runs the full memory pipeline and renders step-by-step UI.
    Uses last few chat turns for better extraction context.
    """
    # Pass recent history for richer extraction context
    conversation = []
    for turn in st.session_state.chat_history[-3:]:
        conversation.append({"role": "user",      "content": turn["user"]})
        conversation.append({"role": "assistant", "content": turn["assistant"]})
    conversation.append({"role": "user",      "content": user_msg})
    conversation.append({"role": "assistant", "content": assistant_msg})

    logs = {"facts": [], "steps": []}

    with st.status("🧠 Running Memory Pipeline...", expanded=True) as status:

        # ── Step 1: Extract facts ──────────────────────────────────────────
        st.write("📥 **Step 1: Extraction Phase**")
        facts = extract_facts(conversation)
        logs["facts"] = facts

        if not facts:
            st.warning("No facts extracted from this message.")
            status.update(label="Pipeline complete — no facts found", state="complete")
            return logs

        st.write(f"✅ Extracted **{len(facts)} candidate facts** (Ω set):")
        for i, f in enumerate(facts):
            st.markdown(f"&nbsp;&nbsp;&nbsp;`ω{i+1}:` {f}")

        st.write("---")

        # ── Step 2: Process each fact ──────────────────────────────────────
        st.write("🔄 **Step 2: Update Phase** — For each ωᵢ: retrieve → decide → act")

        for i, fact in enumerate(facts):
            with st.expander(f"ω{i+1}: `{fact}`", expanded=True):

                # Retrieve similar memories
                from memory.vector_store import search_similar_memories as _search
                similar = _search(fact, top_s=5)
                st.write(f"🔍 **Retrieved {len(similar)} similar memories:**")

                if similar:
                    for m in similar:
                        st.markdown(
                            f"&nbsp;&nbsp;• `{m['id'][:8]}...` | score: `{m['score']:.4f}` | _{m['text']}_"
                        )
                else:
                    st.write("&nbsp;&nbsp;_(empty store — this is new info)_")

                # Run the updater for this single fact
                from memory.updater import process_single_fact
                decision = process_single_fact(fact)
                action   = decision.get("action", "?").upper()

                # Show action badge
                st.markdown(
                    f"<div class='action-{action}' style='margin-top:8px'>{action}</div>",
                    unsafe_allow_html=True
                )

                if action == "UPDATE":
                    st.write(f"&nbsp;&nbsp;New text: _{decision.get('new_text', '')}_")
                elif action == "NOOP":
                    st.write(f"&nbsp;&nbsp;Reason: _{decision.get('reason', 'already captured')}_")

                logs["steps"].append({
                    "fact":     fact,
                    "similar":  similar,
                    "action":   action,
                    "decision": decision
                })

        status.update(label="✅ Memory Pipeline Complete!", state="complete")

    return logs


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Memory Store")
    st.caption("Qdrant Vector DB — Live")

    col_r, col_c = st.columns(2)
    with col_r:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col_c:
        if st.button("🗑️ Clear All", use_container_width=True):
            clear_all_memories()
            st.session_state.chat_history  = []
            st.session_state.pipeline_logs = []
            st.success("Cleared!")
            st.rerun()

    st.divider()
    memories = get_all_memories()

    if memories:
        st.markdown(f"**{len(memories)} memories stored:**")
        for i, mem in enumerate(memories):
            st.markdown(f"""
            <div class='memory-box'>
                <span style='color:#888;font-size:11px'>#{i+1} · {mem['id'][:8]}...</span><br>
                {mem['text']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No memories yet.\nStart chatting!")


# ── Main layout ────────────────────────────────────────────────────────────
st.markdown("# 🧠 Long Term Memory System")
st.caption("Mem0 Architecture · Groq LLM · HuggingFace Embeddings · Qdrant Vector DB")
st.divider()

chat_col, pipe_col = st.columns([1, 1], gap="large")

with chat_col:
    st.markdown("### 💬 Conversation")
    for turn in st.session_state.chat_history:
        st.markdown(f"""
        <div class='chat-user'><strong>👤 You</strong><br>{turn['user']}</div>
        <div class='chat-assistant'><strong>🤖 Assistant</strong><br>{turn['assistant']}</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    user_input = st.text_input(
        "Your message:",
        placeholder="Tell me something about yourself...",
        key="user_input_box",
        label_visibility="collapsed"
    )
    send_btn = st.button("Send 💬", type="primary", use_container_width=True)

with pipe_col:
    st.markdown("### ⚙️ Memory Pipeline")
    st.caption("Runs automatically after each message")
    if not st.session_state.pipeline_logs:
        st.info("Pipeline output will appear here after you send a message.")


# ── Handle send ────────────────────────────────────────────────────────────
if send_btn and user_input.strip():

    with chat_col:
        with st.spinner("Thinking..."):
            assistant_response, memories_used = generate_assistant_response(
                user_input, st.session_state.chat_history
            )

        # Show which memories were injected
        if memories_used:
            st.markdown(
                f"<div class='memory-context'>🧠 Used {len(memories_used)} memories for context</div>",
                unsafe_allow_html=True
            )

        st.markdown(f"""
        <div class='chat-user'><strong>👤 You</strong><br>{user_input}</div>
        <div class='chat-assistant'><strong>🤖 Assistant</strong><br>{assistant_response}</div>
        """, unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "user":      user_input,
            "assistant": assistant_response
        })

    with pipe_col:
        logs = run_pipeline_with_ui(user_input, assistant_response)
        st.session_state.pipeline_logs.append(logs)

        if logs.get("steps"):
            st.markdown("#### 📊 Actions Summary")
            action_counts = {}
            for step in logs["steps"]:
                a = step["action"]
                action_counts[a] = action_counts.get(a, 0) + 1

            cols = st.columns(len(action_counts))
            for idx, (action, count) in enumerate(action_counts.items()):
                with cols[idx]:
                    st.markdown(
                        f"<div class='action-{action}'>{action}<br><small>{count}x</small></div>",
                        unsafe_allow_html=True
                    )

    st.rerun()
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langraph_rag_backend import (
    USE_LLM_TITLES,
    chatbot,
    delete_thread,
    find_fork_config,
    friendly_error,
    generate_title,
    get_thread_documents,
    get_thread_title,
    ingest_pdf,
    retrieve_all_threads,
    save_stopped_turn,
    touch_thread,
)

st.set_page_config(page_title="Multi Utility Chatbot", page_icon="💬")

TOOL_LABELS = {
    "web_search": "Searching the web",
    "get_stock_price": "Fetching the stock price",
    "calculator": "Calculating",
    "rag_tool": "Reading your document",
}
STOPPED_NOTE = "\n\n*⏹ Generation stopped.*"


# =========================== Utilities ===========================
def new_thread_id() -> str:
    return str(uuid.uuid4())


def reset_chat():
    st.session_state["thread_id"] = new_thread_id()
    st.session_state["message_history"] = []
    st.session_state["editing"] = None
    st.session_state["pending"] = None


def messages_to_history(messages) -> list:
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
            # Empty AI messages are tool-call requests with no text, so skip them.
            history.append({"role": "assistant", "content": msg.content})
        # ToolMessage (raw tool output) is skipped so old chats look like live chats.
    return history


def switch_thread(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    st.session_state["thread_id"] = str(thread_id)
    st.session_state["message_history"] = messages_to_history(state.values.get("messages", []))
    st.session_state["editing"] = None


def finish_title(thread_key: str, text: str):
    # Title is set AFTER the answer so it never delays the response.
    if get_thread_title(thread_key) == "New Conversation":
        generate_title(text, thread_key, use_llm=USE_LLM_TITLES)
    touch_thread(thread_key)


# ======================= Session Initialization ===================
st.session_state.setdefault("message_history", [])
st.session_state.setdefault("thread_id", new_thread_id())
st.session_state.setdefault("editing", None)  # index of the user message being edited
st.session_state.setdefault("pending", None)  # turn that is being generated right now

thread_key = str(st.session_state["thread_id"])

# If the previous run was cut short (Stop button, or any click while generating),
# save what had been generated so far.
pending = st.session_state["pending"]
if pending:
    st.session_state["pending"] = None
    save_stopped_turn(pending["thread_id"], pending["text"], pending["partial"])
    if pending["thread_id"] == thread_key:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": pending["partial"].strip() + STOPPED_NOTE}
        )
    finish_title(pending["thread_id"], pending["text"])

# An edited message waiting to be sent: rewind the chat to just before it.
turn = None  # {"text": str, "config": dict | None}
resend = st.session_state.pop("resend", None)
if resend and resend["text"]:
    try:
        fork_config = find_fork_config(thread_key, resend["index"])
        kept, seen = [], -1
        for item in st.session_state["message_history"]:
            if item["role"] == "user":
                seen += 1
                if seen == resend["index"]:
                    break
            kept.append(item)
        st.session_state["message_history"] = kept
        turn = {"text": resend["text"], "config": fork_config}
    except ValueError as e:
        st.error(str(e))

# ============================ Sidebar ============================
st.sidebar.title("Multi Utility Chatbot")
st.sidebar.markdown(f"**Current chat:** {get_thread_title(thread_key)}")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

thread_docs = get_thread_documents(thread_key)
if thread_docs:
    st.sidebar.success("Documents in this chat:")
    for doc in thread_docs:
        st.sidebar.markdown(
            f"- `{doc['filename']}` ({doc['chunks']} chunks, {doc['pages']} pages)"
        )
    doc_mode = st.sidebar.toggle(
        "Prefer my documents",
        value=True,
        key="doc_mode",
        help="On: every question is checked against your PDFs first. "
        "Off: the model decides by itself when to use them.",
    )
else:
    doc_mode = True
    st.sidebar.info("General chat mode. Attach a PDF in the chat bar to ask about a document.")

st.sidebar.subheader("Past conversations")
threads = retrieve_all_threads()  # newest first; only chats that have messages
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for tid in threads:
        col1, col2 = st.sidebar.columns([4, 1])
        is_current = str(tid) == thread_key
        with col1:
            if st.button(
                get_thread_title(tid),
                key=f"side-thread-{tid}",
                use_container_width=True,
                type="primary" if is_current else "secondary",
            ):
                switch_thread(tid)
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del-{tid}"):
                delete_thread(tid)
                if is_current:
                    reset_chat()
                st.rerun()

# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

user_index = -1
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        if message["role"] != "user":
            st.markdown(message["content"])
            continue

        user_index += 1
        if st.session_state["editing"] == user_index:
            edited = st.text_area(
                "Edit your message",
                value=message["content"],
                key=f"edit-box-{user_index}",
                label_visibility="collapsed",
            )
            send_col, cancel_col, _ = st.columns([1, 1, 4])
            if send_col.button("Send", key=f"edit-send-{user_index}", type="primary"):
                st.session_state["resend"] = {"index": user_index, "text": edited.strip()}
                st.session_state["editing"] = None
                st.rerun()
            if cancel_col.button("Cancel", key=f"edit-cancel-{user_index}"):
                st.session_state["editing"] = None
                st.rerun()
        else:
            st.markdown(message["content"])
            if st.button("✏️", key=f"edit-btn-{user_index}", help="Edit and resend"):
                st.session_state["editing"] = user_index
                st.rerun()

user_input = st.chat_input(
    "Ask anything, or attach a PDF",
    accept_file=True,
    file_type=["pdf"],
)

if user_input:
    # With accept_file=True, user_input has .text and .files
    text = user_input.text if hasattr(user_input, "text") else user_input
    files = user_input.files if hasattr(user_input, "files") else []

    # 1. Index any attached PDF first
    ingest_failed = False
    for f in files or []:
        if any(d["filename"] == f.name for d in thread_docs):
            st.info(f"`{f.name}` is already indexed in this chat.")
            continue
        with st.status(f"Indexing {f.name}…", expanded=True) as status_box:
            try:
                ingest_pdf(f.getvalue(), thread_id=thread_key, filename=f.name)
                status_box.update(label=f"✅ {f.name} indexed", state="complete", expanded=False)
            except ValueError as e:  # empty, too large, unreadable, or scanned PDF
                ingest_failed = True
                status_box.update(label=f"❌ {f.name} failed", state="error", expanded=True)
                st.error(str(e))
            except Exception as e:
                ingest_failed = True
                status_box.update(label=f"❌ {f.name} failed", state="error", expanded=True)
                st.error(friendly_error(e))

    # 2. Only a file, no text: refresh so the sidebar shows the new document
    #    (if indexing failed, keep the error message on screen instead)
    if not text:
        if files and not ingest_failed:
            st.rerun()
        st.stop()

    turn = {"text": text, "config": None}

# 3. Generate the answer (for a new message or an edited one)
if turn:
    text = turn["text"]
    st.session_state["message_history"].append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)

    base_configurable = (turn["config"] or {}).get("configurable", {"thread_id": thread_key})
    CONFIG = {
        "configurable": {**base_configurable, "doc_mode": doc_mode},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    # What has been generated so far lives in session_state, so if the run is
    # interrupted (Stop button) the next run can still save the partial answer.
    st.session_state["pending"] = {"thread_id": thread_key, "text": text, "partial": ""}

    with st.chat_message("assistant"):
        stop_slot = st.empty()
        stop_slot.button("⏹ Stop generating", key="stop-generating")
        status_holder = {"box": None}

        def show_tool_status(tool_name: str):
            label = f"🔧 {TOOL_LABELS.get(tool_name, tool_name)}…"
            if status_holder["box"] is None:
                status_holder["box"] = st.status(label, expanded=True)
            else:
                status_holder["box"].update(label=label, state="running", expanded=True)

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=text)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Show the tool status as soon as the model asks for a tool.
                for call in getattr(message_chunk, "tool_call_chunks", None) or []:
                    if call.get("name"):
                        show_tool_status(call["name"])

                if isinstance(message_chunk, ToolMessage):
                    show_tool_status(getattr(message_chunk, "name", None) or "tool")

                if isinstance(message_chunk, AIMessage) and isinstance(message_chunk.content, str):
                    if message_chunk.content:
                        st.session_state["pending"]["partial"] += message_chunk.content
                        yield message_chunk.content

        try:
            ai_message = st.write_stream(ai_only_stream())
        except Exception as e:
            st.session_state["pending"] = None
            st.session_state["message_history"].pop()  # keep the UI in sync with saved state
            st.error(friendly_error(e))
            st.stop()

        stop_slot.empty()
        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Done", state="complete", expanded=False)

    st.session_state["pending"] = None
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message or "(No response from the model.)"}
    )
    finish_title(thread_key, text)
    st.rerun()  # refresh the sidebar (title, order, documents)
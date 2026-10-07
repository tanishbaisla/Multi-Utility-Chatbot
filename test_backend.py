"""
Offline tests: no Ollama needed (models are replaced with fakes).
Run with:  pytest -q
"""
import os
import tempfile

# Point the app at a temporary database and index folder BEFORE importing it.
_TMP = tempfile.mkdtemp()
os.environ["DB_PATH"] = os.path.join(_TMP, "test.db")
os.environ["INDEX_DIR"] = os.path.join(_TMP, "indexes")

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage

import langraph_rag_backend as be


def make_pdf(text: str) -> bytes:
    """Build a tiny one-page PDF containing `text` (empty text = no readable text)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET" if text else ""
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    ).encode()
    return out


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(be, "embeddings", DeterministicFakeEmbedding(size=64))


# ---------------- calculator ----------------
@pytest.mark.parametrize(
    "expr, expected",
    [("(12 * 5) + 3", 63), ("2 ** 10", 1024), ("2 ^ 3", 8), ("sqrt(144)", 12.0), ("-5 + 2", -3)],
)
def test_calculator_ok(expr, expected):
    assert be.calculator.invoke({"expression": expr})["result"] == expected


@pytest.mark.parametrize(
    "expr",
    ["__import__('os').system('echo hi')", "open('x')", "1/0", "9 ** 9999", "", "abs(a)", "1 +"],
)
def test_calculator_rejects_bad_input(expr):
    assert "error" in be.calculator.invoke({"expression": expr})


# ---------------- stock tool ----------------
def test_stock_without_key(monkeypatch):
    monkeypatch.setattr(be, "ALPHA_VANTAGE_KEY", None)
    assert "error" in be.get_stock_price.invoke({"symbol": "AAPL"})


# ---------------- tool-call recovery ----------------
def test_recover_tool_call_from_json_text():
    raw = AIMessage(content='{"name": "calculator", "parameters": {"expression": "2+2"}}')
    fixed = be._recover_tool_call(raw, {"calculator"})
    assert fixed.tool_calls and fixed.tool_calls[0]["name"] == "calculator"


def test_recover_ignores_normal_text_and_unknown_tools():
    normal = AIMessage(content="Hello there")
    assert be._recover_tool_call(normal, {"calculator"}) is normal
    unknown = AIMessage(content='{"name": "hack", "parameters": {}}')
    assert be._recover_tool_call(unknown, {"calculator"}) is unknown


# ---------------- documents and RAG ----------------
def test_ingest_and_retrieve_persists_to_disk():
    tid = "thread-rag"
    summary = be.ingest_pdf(make_pdf("Tanish builds RAG chatbots with LangGraph"), tid, "a.pdf")
    assert summary["chunks"] >= 1
    be.ingest_pdf(make_pdf("Second file about FAISS indexes"), tid, "b.pdf")
    assert [d["filename"] for d in be.get_thread_documents(tid)] == ["a.pdf", "b.pdf"]

    be._STORES.clear()  # simulate an app restart: the index must load from disk
    result = be.rag_tool.invoke(
        {"query": "What does Tanish build?"}, config={"configurable": {"thread_id": tid}}
    )
    sources = {r["source"] for r in result["results"]}
    assert sources == {"a.pdf", "b.pdf"}
    assert all(r["page"] == 1 for r in result["results"])


def test_rag_tool_uses_config_thread_and_never_leaks_other_chats():
    be.ingest_pdf(make_pdf("private text"), "thread-owner", "owner.pdf")
    result = be.rag_tool.invoke({"query": "x"}, config={"configurable": {"thread_id": "someone-else"}})
    assert "error" in result


def test_duplicate_upload_is_not_indexed_twice():
    tid = "thread-dup"
    first = be.ingest_pdf(make_pdf("hello world"), tid, "same.pdf")
    second = be.ingest_pdf(make_pdf("hello world"), tid, "same.pdf")
    assert first == second
    assert len(be.get_thread_documents(tid)) == 1


def test_scanned_or_bad_pdfs_give_clear_errors():
    with pytest.raises(ValueError, match="No readable text"):
        be.ingest_pdf(make_pdf(""), "thread-empty", "scan.pdf")
    with pytest.raises(ValueError):
        be.ingest_pdf(b"", "thread-empty", "empty.pdf")
    with pytest.raises(ValueError):
        be.ingest_pdf(b"this is not a pdf", "thread-empty", "fake.pdf")


def test_size_limit(monkeypatch):
    monkeypatch.setattr(be, "MAX_PDF_MB", 0)
    with pytest.raises(ValueError, match="larger"):
        be.ingest_pdf(make_pdf("hi"), "thread-big", "big.pdf")


# ---------------- prompts and routing ----------------
def test_system_prompt_depends_on_documents():
    assert "No document has been uploaded" in be._system_prompt("thread-none").content
    be.ingest_pdf(make_pdf("some text"), "thread-prompt", "notes.pdf")
    prompt = be._system_prompt("thread-prompt").content
    assert "notes.pdf" in prompt and "rag_tool" in prompt


# ---------------- full graph with a stub model ----------------
class StubModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "calculator", "args": {"expression": "(12*5)+3"}, "id": "c1", "type": "tool_call"}
                ],
            )
        return AIMessage(content="The answer is 63.")


def test_graph_runs_tool_loop_titles_ordering_and_delete(monkeypatch):
    monkeypatch.setattr(be, "llm_without_rag", StubModel())
    tid = "thread-graph"
    config = {"configurable": {"thread_id": tid}}

    out = be.chatbot.invoke({"messages": [HumanMessage(content="what is (12*5)+3?")]}, config=config)
    assert out["messages"][-1].content == "The answer is 63."
    assert any(getattr(m, "name", None) == "calculator" for m in out["messages"])

    assert be.get_thread_title(tid) == "New Conversation"
    be.generate_title("what is (12*5)+3?", tid)
    assert be.get_thread_title(tid) == "what is (12*5)+3?"
    assert len(be._quick_title("x" * 100)) <= 41

    assert tid in be.retrieve_all_threads()

    # newest-first ordering
    monkeypatch.setattr(be, "llm_without_rag", StubModel())
    be.chatbot.invoke(
        {"messages": [HumanMessage(content="hi")]}, config={"configurable": {"thread_id": "thread-newer"}}
    )
    be.touch_thread("thread-newer")
    threads = be.retrieve_all_threads()
    assert threads.index("thread-newer") < threads.index(tid)

    be.delete_thread(tid)
    assert tid not in be.retrieve_all_threads()
    assert be.get_thread_title(tid) == "New Conversation"


def test_delete_thread_removes_documents_and_index():
    tid = "thread-del"
    be.ingest_pdf(make_pdf("to be deleted"), tid, "gone.pdf")
    assert os.path.isdir(be._index_path(tid))
    be.delete_thread(tid)
    assert be.get_thread_documents(tid) == []
    assert not os.path.isdir(be._index_path(tid))


# ---------------- friendly errors ----------------
def test_friendly_errors():
    assert "Ollama" in be.friendly_error(ConnectionError("Connection refused"))
    assert "ollama pull" in be.friendly_error(Exception("model 'x' not found"))

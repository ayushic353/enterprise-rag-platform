"""
Streamlit UI for the Enterprise RAG Platform.
Uses the exact same rag/ pipeline as the FastAPI layer in api/main.py,
so behavior is identical whether you're clicking buttons here or
calling the REST API directly.
"""
import streamlit as st

from rag import config, ingest, retrieval
from rag.vector_store import get_store

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚", layout="wide")

st.title("📚 Enterprise Knowledge Assistant")
st.caption("RAG-powered chat over your company documents — Groq/Ollama · ChromaDB · multimodal")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if st.button("Ingest Documents", type="primary", disabled=not uploaded_files):
        with st.spinner("Processing and embedding documents..."):
            total_chunks = 0
            for f in uploaded_files:
                try:
                    chunks, metadatas = ingest.process_file(f.name, f.read())
                    total_chunks += retrieval.ingest_chunks(chunks, metadatas)
                except ValueError as exc:
                    st.error(f"{f.name}: {exc}")
            st.success(f"Ingested {total_chunks} chunks from {len(uploaded_files)} file(s).")

    st.divider()
    store = get_store()
    st.metric("Chunks indexed", store.count())
    sources = store.list_sources()
    if sources:
        st.write("**Indexed sources:**")
        for s in sources:
            st.write(f"- {s}")

    st.divider()
    if st.button("🗑️ Clear knowledge base"):
        store.reset()
        st.session_state.messages = []
        st.rerun()

st.divider()
query_image = st.file_uploader(
    "Optionally attach an image to your question", type=["png", "jpg", "jpeg"], key="qimg"
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"**{s['source']}** (relevance: {s['relevance']})")
                    st.caption(s["excerpt"])

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            image_bytes = query_image.read() if query_image else None
            result = retrieval.answer_question(question, image_bytes=image_bytes)
            st.markdown(result["answer"])
            st.caption(f"via {result['provider']}")
            if result["sources"]:
                with st.expander("Sources"):
                    for s in result["sources"]:
                        st.write(f"**{s['source']}** (relevance: {s['relevance']})")
                        st.caption(s["excerpt"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )

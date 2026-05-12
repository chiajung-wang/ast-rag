from __future__ import annotations
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import graph
from retrieval.pipeline import read_file
from ui.helpers import parse_citations, build_permalink
from indexer.corpus_config import DB_PATH

load_dotenv()

for _var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    if not os.getenv(_var):
        st.error(f"Missing env var: {_var}. Add it to .env and restart.")
        st.stop()

if not os.path.exists(DB_PATH):
    st.error("Index not found. Run `make index` first.")
    st.stop()

st.set_page_config(page_title="ast-rag", page_icon="🔍")
st.title("ast-rag — langchain-core Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("Ask about langchain-core..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke({
                "messages": st.session_state.messages,
                "retrieved_chunks": [],
            })
        answer = result["messages"][-1].content
        st.markdown(answer)

        citations = parse_citations(answer)
        for path, start, end in citations:
            with st.expander(f"[{path}:{start}-{end}]"):
                try:
                    source = read_file(path, start, end)
                    st.code(source, language="python")
                except Exception:
                    st.warning("Source not available.")
                st.markdown(f"[View on GitHub]({build_permalink(path, start, end)})")

        st.session_state.messages.append(AIMessage(content=answer))

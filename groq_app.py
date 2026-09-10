import os

import streamlit as st

from groq_chat import (
    DEFAULT_MODEL,
    build_knowledge_items,
    ask_groq,
    load_cv,
    load_dotenv_file,
    select_context_items,
)


def load_streamlit_secret():
    if os.getenv("GROQ_API_KEY"):
        return

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key


st.set_page_config(page_title="Ask DuRu", layout="centered")

load_dotenv_file()
load_streamlit_secret()

items = build_knowledge_items(load_cv())

model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
context_limit = int(os.getenv("DURU_CONTEXT_LIMIT", "6"))
show_context = os.getenv("DURU_SHOW_CONTEXT", "").lower() in {"1", "true", "yes"}

if not os.getenv("GROQ_API_KEY"):
    st.error("DuRu is not configured yet. Add GROQ_API_KEY to Streamlit secrets.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello, I am DuRu, Rustam's personal assistant. "
                "Ask me about him, his work, research, skills, projects, or publications."
            ),
        }
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

question = st.chat_input("Ask about Rustam")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    context_items = select_context_items(question, items, limit=context_limit)
    context = "\n\n".join(item.content for item in context_items)

    if show_context:
        with st.expander("Selected CV context"):
            st.write(context)

    with st.chat_message("assistant"):
        with st.spinner("DuRu is thinking..."):
            try:
                answer = ask_groq(question, context, model=model)
            except RuntimeError as error:
                answer = f"Error: {error}"

            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

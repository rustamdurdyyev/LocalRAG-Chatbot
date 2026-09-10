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


st.set_page_config(page_title="DuRu CV Assistant", layout="centered")

load_dotenv_file()
load_streamlit_secret()

items = build_knowledge_items(load_cv())

st.title("DuRu CV Assistant")
st.caption("Rustam Durdyyev's AI CV and portfolio assistant")

with st.sidebar:
    st.subheader("Settings")
    model = st.text_input("Groq model", value=os.getenv("GROQ_MODEL", DEFAULT_MODEL))
    context_limit = st.slider("Context limit", min_value=3, max_value=12, value=6)
    show_context = st.toggle("Show selected context", value=False)

    if os.getenv("GROQ_API_KEY"):
        st.success("Groq API key loaded")
    else:
        entered_api_key = st.text_input("Groq API key", type="password")
        if entered_api_key:
            os.environ["GROQ_API_KEY"] = entered_api_key.strip()
            st.success("Groq API key loaded for this session")
        else:
            st.warning("Add GROQ_API_KEY in .env, Streamlit secrets, or this field")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello, I am DuRu, Rustam's personal CV assistant. "
                "Ask me about his skills, education, publications, experience, projects, or activities."
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

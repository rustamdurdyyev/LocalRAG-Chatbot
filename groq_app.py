import os
import inspect
import uuid

import streamlit as st

from chat_logging import log_chat_interaction
from groq_chat import (
    DEFAULT_MODEL,
    build_knowledge_items,
    ask_groq,
    load_cv,
    load_dotenv_file,
    select_context_items,
)


CONTACT_KEYWORDS = ["contact", "email", "linkedin", "reach", "message"]
STREAMLIT_SECRET_KEYS = [
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "DURU_CONTEXT_LIMIT",
    "DURU_SHOW_CONTEXT",
    "SUPABASE_URL",
    "SUPABASE_SECRET_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_KEY",
    "SUPABASE_PUBLISHABLE_KEY",
    "SUPABASE_ANON_KEY",
]


def is_contact_request(question):
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in CONTACT_KEYWORDS)


def select_context_items_compatible(question, items, limit, include_contact):
    parameters = inspect.signature(select_context_items).parameters
    if "include_contact" in parameters:
        return select_context_items(
            question,
            items,
            limit=limit,
            include_contact=include_contact,
        )

    return select_context_items(question, items, limit=limit)


def ask_groq_compatible(question, context, model, allow_contact_invitation):
    parameters = inspect.signature(ask_groq).parameters
    if "allow_contact_invitation" in parameters:
        return ask_groq(
            question,
            context,
            model=model,
            allow_contact_invitation=allow_contact_invitation,
        )

    return ask_groq(question, context, model=model)


def load_streamlit_secrets():
    for key in STREAMLIT_SECRET_KEYS:
        if os.getenv(key):
            continue

        try:
            value = st.secrets.get(key)
        except Exception:
            value = None

        if value:
            os.environ[key] = str(value)


st.set_page_config(page_title="Ask DuRu", layout="centered")

load_dotenv_file()
load_streamlit_secrets()

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

if "contact_invitation_shown" not in st.session_state:
    st.session_state.contact_invitation_shown = False

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

question = st.chat_input("Ask about Rustam")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    contact_request = is_contact_request(question)
    allow_contact_suggestion = not st.session_state.contact_invitation_shown
    include_contact = allow_contact_suggestion or contact_request
    context_items = select_context_items_compatible(
        question,
        items,
        limit=context_limit,
        include_contact=include_contact,
    )
    context = "\n\n".join(item.content for item in context_items)

    if show_context:
        with st.expander("Selected CV context"):
            st.write(context)

    with st.chat_message("assistant"):
        with st.spinner("DuRu is thinking..."):
            try:
                answer = ask_groq_compatible(
                    question,
                    context,
                    model=model,
                    allow_contact_invitation=allow_contact_suggestion,
                )
            except RuntimeError as error:
                answer = f"Error: {error}"

            st.write(answer)

    answer_lower = answer.lower()
    contact_markers = [
        "leave your name",
        "leave a message",
        "preferred contact",
        "rustam can contact you",
        "contact rustam",
        "contacted via linkedin",
        "via linkedin",
    ]
    if any(marker in answer_lower for marker in contact_markers):
        st.session_state.contact_invitation_shown = True

    log_chat_interaction(
        session_id=st.session_state.session_id,
        question=question,
        answer=answer,
        model=model,
        source="streamlit-groq",
    )

    st.session_state.messages.append({"role": "assistant", "content": answer})

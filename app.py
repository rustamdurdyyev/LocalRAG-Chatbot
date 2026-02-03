import os
import json
import shutil
import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="DuRu Assistant", layout="wide")

# ============================================================
# 1. Load CV
# ============================================================
with open("cv.json", "r", encoding="utf-8") as f:
    cv = json.load(f)

# ============================================================
# 2. Create Documents (same as main.py)
# ============================================================
documents = []
ids = []

documents.append(Document(
    page_content=cv["professional_summary"],
    metadata={"type": "summary"},
    id="summary"
))
ids.append("summary")

skills_text = "\n".join([
    "Programming Languages: " + ", ".join(cv["technical_skills"]["programming_languages"]),
    "Machine Learning: " + ", ".join(cv["technical_skills"]["machine_learning"]),
    "Python Libraries: " + ", ".join(cv["technical_skills"]["python_libraries"]),
    "Statistical Analysis: " + ", ".join(cv["technical_skills"]["statistical_analysis"]),
    "Cloud & HPC: " + ", ".join(cv["technical_skills"]["cloud_and_hpc"])
])
documents.append(Document(page_content=skills_text, metadata={"type": "skills"}, id="skills"))
ids.append("skills")

for i, exp in enumerate(cv["experience"]):
    exp_text = (
        f"Role: {exp['role']}\n"
        f"Organization: {exp['organization']}\n"
        f"Location: {exp.get('location','')}\n"
        f"Duration: {exp.get('start_date','')} - {exp.get('end_date','')}\n"
        f"Responsibilities: {' '.join(exp['responsibilities'])}"
    )
    documents.append(Document(page_content=exp_text, metadata={"type": "experience"}, id=f"experience_{i}"))
    ids.append(f"experience_{i}")

for i, edu in enumerate(cv["education"]):
    edu_text = (
        f"Degree: {edu['degree']}\n"
        f"Field: {edu.get('field','')}\n"
        f"Institution: {edu['institution']}\n"
        f"Location: {edu['location']}\n"
        f"Duration: {edu['start_date']} - {edu['end_date']}"
    )
    documents.append(Document(page_content=edu_text, metadata={"type": "education"}, id=f"education_{i}"))
    ids.append(f"education_{i}")

for i, pub in enumerate(cv.get("publications", [])):
    pub_text = (
        f"Title: {pub['title']}\n"
        f"Year: {pub['year']}\n"
        f"Authors: {', '.join(pub['authors'])}\n"
        f"Journal: {pub['journal']}\n"
        f"DOI/Link: {pub.get('doi', pub.get('link','N/A'))}"
    )
    documents.append(Document(page_content=pub_text, metadata={"type": "publication"}, id=f"publication_{i}"))
    ids.append(f"publication_{i}")

for i, award in enumerate(cv.get("awards_and_grants", [])):
    documents.append(Document(page_content=award, metadata={"type": "award"}, id=f"award_{i}"))
    ids.append(f"award_{i}")

for i, lang in enumerate(cv.get("languages", [])):
    lang_text = f"{lang['language']}: {lang['proficiency']}"
    documents.append(Document(page_content=lang_text, metadata={"type": "language"}, id=f"language_{i}"))
    ids.append(f"language_{i}")

# ============================================================
# 3. Vector Store (persistent)
# ============================================================
db_path = "./chroma_langchain_db"

@st.cache_resource
def load_vector_store():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    if not os.path.exists(db_path):
        os.makedirs(db_path, exist_ok=True)
        vector_store = Chroma(
            collection_name="about_me",
            persist_directory=db_path,
            embedding_function=embeddings
        )
        vector_store.add_documents(documents=documents, ids=ids)
        return vector_store

    return Chroma(
        collection_name="about_me",
        persist_directory=db_path,
        embedding_function=embeddings
    )

vector_store = load_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 6})

# ============================================================
# 4. Helper
# ============================================================
def get_all_items(doc_type: str):
    return [doc.page_content for doc in documents if doc.metadata.get("type") == doc_type]

# ============================================================
# 5. LLM + Template (strict extraction)
# ============================================================
template = """
You are a friendly assistant who knows Rustam Durdyyev personally.
Answer questions in a human, positive, and engaging way.
Rustam is a researcher and data scientist specializing in computational science and engineering.

If the user asks about, refers to, or indirectly mentions any of the following fields:
• education
• publications
• awards
• languages
• experience
• skills

—you MUST follow these rules:

1. Provide **all items** from every mentioned field.
2. Copy each item **exactly as it appears in the context**, without rewriting or paraphrasing.
3. Present each item as a separate bullet point.
4. Never skip or merge items.
5. Never summarize inside the bullet list.
6. After listing all items, add a short, warm, positive summary about Rustam’s strengths or achievements.
7. If multiple fields are mentioned, list all items from each field.
8. If you choose to mention a field yourself, you must also list **all items** from that field.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3.2")
chain = prompt | model

# ============================================================
# 6. Chat UI
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("Ask a question about Rustam")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    q_lower = question.lower()

    # FIELD DETECTION
    if "publication" in q_lower or "paper" in q_lower or "article" in q_lower:
        context_docs = get_all_items("publication")

    elif "award" in q_lower or "honor" in q_lower or "scholarship" in q_lower:
        context_docs = get_all_items("award")

    elif "language" in q_lower:
        context_docs = get_all_items("language")

    elif "education" in q_lower or "study" in q_lower or "degree" in q_lower:
        context_docs = get_all_items("education")

    elif "experience" in q_lower or "work" in q_lower or "career" in q_lower:
        context_docs = get_all_items("experience")

    elif "rustam" in q_lower or "about him" in q_lower or "who is he" in q_lower:
        # GENERAL BIO → include EVERYTHING
        context_docs = (
            get_all_items("education")
            + get_all_items("experience")
            + get_all_items("award")
            + get_all_items("publication")
            + get_all_items("language")
            + get_all_items("skills")
        )

    else:
        # fallback semantic search
        docs = retriever.vectorstore.similarity_search(query=question, k=6)
        context_docs = [d.page_content for d in docs]

    full_context = "\n\n".join(context_docs)

    answer = chain.invoke({"context": full_context, "question": question})

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

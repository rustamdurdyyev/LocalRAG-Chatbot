# main.py

import os
import json
import shutil
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# 1. Load CV
# ============================================================
with open("cv.json", "r", encoding="utf-8") as f:
    cv = json.load(f)

# ============================================================
# 2. Create Documents
# ============================================================
documents = []
ids = []

# --- Summary ---
documents.append(Document(
    page_content=cv["professional_summary"],
    metadata={"type": "summary", "name": cv["personal_info"]["full_name"]},
    id="summary"
))
ids.append("summary")

# --- Skills ---
skills_text = "\n".join([
    "Programming Languages: " + ", ".join(cv["technical_skills"]["programming_languages"]),
    "Machine Learning: " + ", ".join(cv["technical_skills"]["machine_learning"]),
    "Python Libraries: " + ", ".join(cv["technical_skills"]["python_libraries"]),
    "Cloud & HPC: " + ", ".join(cv["technical_skills"]["cloud_and_hpc"])
])
documents.append(Document(
    page_content=skills_text,
    metadata={"type": "skills"},
    id="skills"
))
ids.append("skills")

# --- Experience ---
for i, exp in enumerate(cv["experience"]):
    exp_text = f"Role: {exp['role']}\nOrganization: {exp['organization']}\nResponsibilities: {' '.join(exp['responsibilities'])}"
    documents.append(Document(
        page_content=exp_text,
        metadata={"type": "experience", "role": exp["role"], "organization": exp["organization"]},
        id=f"experience_{i}"
    ))
    ids.append(f"experience_{i}")

# --- Education ---
for i, edu in enumerate(cv["education"]):
    edu_text = f"Degree: {edu['degree']}\nInstitution: {edu['institution']}\nLocation: {edu.get('location','')}\nDuration: {edu.get('start_date','')} - {edu.get('end_date','')}"
    documents.append(Document(
        page_content=edu_text,
        metadata={"type": "education", "degree": edu["degree"], "institution": edu["institution"]},
        id=f"education_{i}"
    ))
    ids.append(f"education_{i}")

# --- Publications ---
for i, pub in enumerate(cv.get("publications", [])):
    pub_text = f"Title: {pub['title']}\nYear: {pub['year']}\nAuthors: {', '.join(pub.get('authors', []))}\nJournal: {pub.get('journal', 'N/A')}\nDOI/Link: {pub.get('doi', pub.get('link','N/A'))}"
    documents.append(Document(
        page_content=pub_text,
        metadata={"type": "publication", "title": pub["title"], "year": pub["year"]},
        id=f"publication_{i}"
    ))
    ids.append(f"publication_{i}")

# --- Awards ---
for i, award in enumerate(cv.get("awards_and_grants", [])):
    documents.append(Document(
        page_content=award,
        metadata={"type": "award", "award": award},
        id=f"award_{i}"
    ))
    ids.append(f"award_{i}")

# --- Languages ---
for i, lang in enumerate(cv.get("languages", [])):
    lang_text = f"{lang['language']}: {lang['proficiency']}"
    documents.append(Document(
        page_content=lang_text,
        metadata={"type": "language", "language": lang['language'], "proficiency": lang['proficiency']},
        id=f"language_{i}"
    ))
    ids.append(f"language_{i}")

# ============================================================
# 3. Vector Store (Chroma)
# ============================================================
db_path = "./chroma_langchain_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Delete old DB to force fresh indexing
if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print("🗑 Old DB deleted for fresh indexing")

vector_store = Chroma(
    collection_name="about_me",
    persist_directory=db_path,
    embedding_function=embeddings
)
vector_store.add_documents(documents=documents, ids=ids)
print("✅ Documents added to vector store")

# Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 6})

# ============================================================
# 4. Helper: Get all items of a specific type
# ============================================================
def get_all_items(doc_type: str):
    """Return all page_contents for documents of a given type."""
    all_docs = []
    for doc in documents:
        if doc.metadata.get("type") == doc_type:
            all_docs.append(doc.page_content)
    return all_docs

# ============================================================
# 5. LLM + Prompt
# ============================================================
model = OllamaLLM(model="llama3.2")
template = """
You are a friendly assistant who knows Rustam Durdyyev personally.
Answer questions in a human, positive, and engaging way.
Rustam is a researcher and data scientist specializing in computational science and engineering.
If the user asks for specific fields (publications, awards, languages, education), provide all items.
Include CV details where relevant, and always answer confidently.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# ============================================================
# 6. Interactive loop
# ============================================================
positive_intro = (
    "I know Rustam personally. "
    "He is an outstanding data scientist with a strong background in computational science and engineering. "
    "He has published scientific articles, "
    "applies machine learning in innovative ways, and is passionate about solving complex problems."
)

intro_message = (
    "Hello! I am DuRu, Rustam's personal assistant. "
    "I’m glad to answer questions about his CV, professional work, or even personal life.\n"
    "Rustam Durdyyev is a researcher and data scientist with expertise in computational science and engineering, focusing on data-driven simulations, machine learning applications, and solving complex scientific problems.\n"
    "You can ask me about his publications, awards, education, languages,or any other relevant info."
)
print(intro_message)

while True:
    question = input("\nAsk a question about Rustam (q to quit): ").strip()
    if question.lower() == "q":
        print("Goodbye! DuRu signing off. See you later!")
        break

    # Check if user is asking about a specific field
    q_lower = question.lower()
    if "publication" in q_lower or "paper" in q_lower or "article" in q_lower:
        context_docs = get_all_items("publication")
    elif "award" in q_lower or "honor" in q_lower or "scholarship" in q_lower:
        context_docs = get_all_items("award")
    elif "language" in q_lower:
        context_docs = get_all_items("language")
    elif "education" in q_lower:
        context_docs = get_all_items("education")
    else:
        # fallback semantic search for general questions
        docs = retriever.vectorstore.similarity_search(query=question, k=6)
        context_docs = [d.page_content for d in docs]

    # Combine with positive intro
    full_context = positive_intro + "\n\n" + "\n\n".join(context_docs)

    # Generate answer
    result = chain.invoke({"context": full_context, "question": question})
    print("\nAnswer:\n", result)

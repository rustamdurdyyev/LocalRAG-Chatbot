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
CV_PATH = "cv.json"
with open(CV_PATH, "r", encoding="utf-8") as f:
    cv = json.load(f)

# ============================================================
# 2. Create Documents
# ============================================================
documents = []
ids = []

# --- Summary ---
documents.append(
    Document(
        page_content=cv["professional_summary"],
        metadata={
            "type": "summary",
            "name": cv["personal_info"]["full_name"],
            "title": cv["personal_info"]["title"],
            "location": cv["personal_info"]["location"]["city"]
        },
        id="summary"
    )
)
ids.append("summary")

# --- Skills ---
skills_text = "\n".join([
    "Programming Languages: " + ", ".join(cv["technical_skills"]["programming_languages"]),
    "Machine Learning: " + ", ".join(cv["technical_skills"]["machine_learning"]),
    "Python Libraries: " + ", ".join(cv["technical_skills"]["python_libraries"]),
    "Statistical Analysis: " + ", ".join(cv["technical_skills"]["statistical_analysis"]),
    "Cloud & HPC: " + ", ".join(cv["technical_skills"]["cloud_and_hpc"])
])
documents.append(
    Document(
        page_content=skills_text,
        metadata={"type": "skills", "name": cv["personal_info"]["full_name"]},
        id="skills"
    )
)
ids.append("skills")

# --- Experience ---
for i, exp in enumerate(cv["experience"]):
    exp_text = f"""
Role: {exp['role']}
Organization: {exp['organization']}
Location: {exp.get('location', 'N/A')}
Duration: {exp.get('start_date', '')} - {exp.get('end_date', '')}

Responsibilities:
{" ".join(exp['responsibilities'])}
"""
    documents.append(
        Document(
            page_content=exp_text.strip(),
            metadata={
                "type": "experience",
                "role": exp["role"],
                "organization": exp["organization"]
            },
            id=f"experience_{i}"
        )
    )
    ids.append(f"experience_{i}")

# --- Education ---
for i, edu in enumerate(cv["education"]):
    edu_text = f"Degree: {edu['degree']}\nInstitution: {edu['institution']}\nLocation: {edu.get('location','')}\nDuration: {edu.get('start_date','')} - {edu.get('end_date','')}"
    documents.append(
        Document(
            page_content=edu_text.strip(),
            metadata={"type": "education", "degree": edu["degree"], "institution": edu["institution"]},
            id=f"education_{i}"
        )
    )
    ids.append(f"education_{i}")

# --- Publications ---
for i, pub in enumerate(cv.get("publications", [])):
    pub_text = f"Title: {pub['title']}\nYear: {pub['year']}\nAuthors: {', '.join(pub.get('authors', []))}\nJournal: {pub.get('journal','N/A')}\nDOI/Link: {pub.get('doi', pub.get('link','N/A'))}"
    documents.append(
        Document(
            page_content=pub_text.strip(),
            metadata={"type": "publication", "year": int(pub["year"]), "title": pub["title"]},
            id=f"publication_{i}"
        )
    )
    ids.append(f"publication_{i}")

# --- Awards ---
for i, award in enumerate(cv.get("awards_and_grants", [])):
    documents.append(
        Document(
            page_content=award,
            metadata={"type": "award", "award": award},
            id=f"award_{i}"
        )
    )
    ids.append(f"award_{i}")

# --- Languages ---
for i, lang in enumerate(cv.get("languages", [])):
    lang_text = f"{lang['language']}: {lang['proficiency']}"
    documents.append(
        Document(
            page_content=lang_text.strip(),
            metadata={"type": "language", "language": lang['language'], "proficiency": lang['proficiency']},
            id=f"language_{i}"
        )
    )
    ids.append(f"language_{i}")

# ============================================================
# 3. Embeddings + Chroma Vector Store
# ============================================================
DB_PATH = "./chroma_langchain_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print("🗑 Old DB deleted for fresh indexing")

vector_store = Chroma(
    collection_name="about_me",
    persist_directory=DB_PATH,
    embedding_function=embeddings
)
vector_store.add_documents(documents=documents, ids=ids)
print("✅ Documents added to vector store")

# ============================================================
# 4. Retriever
# ============================================================
retriever = vector_store.as_retriever(search_kwargs={"k": 6})

# --- Helper for structured fields ---
def get_all_by_type(doc_type: str):
    data = vector_store.get(where={"type": doc_type})
    return [d["page_content"] for d in data["documents"]]

def get_context(query: str):
    q = query.lower()
    if "publication" in q or "paper" in q or "article" in q:
        return get_all_by_type("publication")
    if "award" in q or "honor" in q or "scholarship" in q:
        return get_all_by_type("award")
    if "language" in q:
        return get_all_by_type("language")
    if "education" in q:
        return get_all_by_type("education")
    # fallback semantic search
    docs = retriever.vectorstore.similarity_search(query=query, k=6)
    return [d.page_content for d in docs]

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
    "I know Rustam personally. He is an outstanding data scientist with a strong background "
    "in computational science and engineering. He has published scientific articles, "
    "applies machine learning in innovative ways, and is passionate about solving complex problems."
)

while True:
    question = input("\nAsk a question about Rustam (q to quit): ").strip()
    if question.lower() == "q":
        break

    # Get structured + semantic context
    context_docs = get_context(question)
    full_context = positive_intro + "\n\n" + "\n\n".join(context_docs)

    # Invoke LLM
    result = chain.invoke({"context": full_context, "question": question})
    print("\nAnswer:\n", result)

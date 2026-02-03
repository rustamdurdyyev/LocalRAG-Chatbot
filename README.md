Local AI Chatbot with RAG (Ollama + Streamlit)

This project is a local AI chatbot built with Retrieval-Augmented Generation (RAG) using Ollama.
It allows you to chat with an AI model that answers questions based on your own data (for example, your CV stored in cv.json).

You can run the chatbot:
- In the terminal (main.py)
- In a web interface using Streamlit (app.py)

Everything runs locally on your machine — no external APIs required.

--------------------------------------------------

Features

- 100% local AI using Ollama
- RAG (Retrieval-Augmented Generation) from your own JSON data
- Terminal-based chatbot
- Web UI with Streamlit
- Simple setup and easy to extend

--------------------------------------------------

Project Structure

main.py          Terminal chatbot
vector.py        Vectorizes cv.json (creates embeddings)
app.py           Streamlit web application
cv.json          Knowledge base (your data)
requirements.txt
README.md

--------------------------------------------------

Prerequisites

- Python 3.9 or higher
- Ollama installed locally

--------------------------------------------------

Install Ollama

Download and install Ollama from:
https://ollama.com

Pull a model (example):

ollama pull llama3

Start Ollama:

ollama serve

Test it:

ollama run llama3

--------------------------------------------------

Installation

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

(Optional) Create a virtual environment:

python -m venv venv
source venv/bin/activate    (Windows: venv\Scripts\activate)

Install dependencies:

pip install -r requirements.txt

--------------------------------------------------

Build the Vector Database (RAG Setup)

Before running the chatbot, you need to vectorize your data:

python vector.py

This reads cv.json and prepares embeddings for retrieval.

--------------------------------------------------

Run in Terminal Mode

python main.py

--------------------------------------------------

Run Web App (Streamlit UI)

streamlit run app.py

Then open the link shown in your terminal (usually http://localhost:8501)

--------------------------------------------------

How RAG Works in This Project

1. Your data (cv.json) is converted into vector embeddings
2. When a user asks a question, relevant information is retrieved
3. The retrieved context is sent to Ollama
4. Ollama generates a grounded, context-aware answer

--------------------------------------------------

Customize Your Data

You can replace cv.json with your own data (for example: documents, notes, FAQs).
After changing the file, rebuild the vectors:

python vector.py

--------------------------------------------------

This project was inspired by the Tech With Tim tutorial:  
https://www.youtube.com/watch?v=E4l91XKQSgw (How to build a local AI agent with Python, Ollama, LangChain & RAG) :contentReference[oaicite:0]{index=0}

You can also find an example vector loader implementation here:  
https://github.com/techwithtim/LocalAIAgentWithRAG/blob/main/vector.py :contentReference[oaicite:1]{index=1}

--------------------------------------------------

License

This project is open-source. Feel free to use, modify, and share.

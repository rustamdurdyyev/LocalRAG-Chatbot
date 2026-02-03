## 🤖 Local AI Chatbot with RAG (Ollama + Streamlit)

This project is a local AI chatbot built with Retrieval-Augmented Generation (RAG) using Ollama.
It allows you to chat with an AI model that answers questions based on your own data (for example, your CV stored in cv.json).

You can run the chatbot:
- In the terminal (main.py)
- In a web interface using Streamlit (app.py)

Everything runs locally on your machine — no external APIs required.

--------------------------------------------------

## ✨ Features

- 100% local AI using Ollama
- RAG (Retrieval-Augmented Generation) from your own JSON data
- Terminal-based chatbot
- Web UI with Streamlit
- Simple setup and easy to extend

--------------------------------------------------

## 📂 Project Structure

main.py          Terminal chatbot
vector.py        Vectorizes cv.json (creates embeddings)
app.py           Streamlit web application
cv.json          Knowledge base (your data)
requirements.txt
README.md

--------------------------------------------------

## 🧰 Setup Requirements

- Python 3.9 or higher
- Ollama installed locally

--------------------------------------------------

## 🦙 Installing Ollama

Download and install Ollama from:
https://ollama.com

Pull a model (example):

ollama pull llama3

Start Ollama:

ollama serve

Test it:

ollama run llama3

--------------------------------------------------

## 📦 Installing the Project

Clone the repository:

git clone https://github.com/rustamdurdyyev/LocalRAG-Chatbot/tree/main

(Optional) Create a virtual environment:

python -m venv venv
source venv/bin/activate    (Windows: venv\Scripts\activate)

Install dependencies:

pip install -r requirements.txt

--------------------------------------------------

## 💻 Terminal Mode

python main.py

--------------------------------------------------

## 🌐 Streamlit Web App

streamlit run app.py

Then open the link shown in your terminal (usually http://localhost:8501)

--------------------------------------------------

## 🧠 How RAG Works

1. Your data (cv.json) is converted into vector embeddings
2. When a user asks a question, relevant information is retrieved
3. The retrieved context is sent to Ollama
4. Ollama generates a grounded, context-aware answer

--------------------------------------------------

## 🔄 Customize Your Data

You can replace cv.json with your own data (for example: documents, notes, FAQs).
After changing the file, rebuild the vectors:

python vector.py

--------------------------------------------------

## 📜 License

This project is open-source. Feel free to use, modify, and share.

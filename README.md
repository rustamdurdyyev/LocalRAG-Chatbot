# DuRu CV Assistant

DuRu is a personal AI assistant for Rustam Durdyyev's portfolio. It answers questions about Rustam's CV, education, skills, experience, publications, awards, projects, activities, and languages.

The project supports two modes:

- Local mode with Ollama and Chroma RAG
- Groq API mode for a lightweight portfolio demo

## Features

- CV-based question answering from `cv.json`
- Portfolio project and activity summaries
- Terminal chatbot
- Streamlit web chatbot
- Local Ollama support
- Groq API support with `.env` configuration
- Portfolio-friendly structure

## Project Structure

```text
app.py              Streamlit app using local Ollama
main.py             Terminal chatbot using local Ollama
vector.py           Builds the local Chroma vector database
groq_chat.py        Terminal chatbot using Groq API
groq_app.py         Streamlit chatbot using Groq API
cv.json             CV knowledge base
requirements.txt    Python dependencies
.env.example        Example environment variables
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run Groq Terminal Chat

Create a free Groq API key:

```text
https://console.groq.com/keys
```

Set the key in PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
python groq_chat.py
```

Or create a local `.env` file:

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

Then run:

```powershell
python groq_chat.py
```

## Run Groq Web App

```powershell
streamlit run groq_app.py
```

## Run Local Ollama Version

Install Ollama:

```text
https://ollama.com
```

Pull the required models:

```powershell
ollama pull llama3.2
ollama pull mxbai-embed-large
```

Run the terminal version:

```powershell
python main.py
```

Run the web version:

```powershell
streamlit run app.py
```

## Portfolio Deployment Note

Do not put `GROQ_API_KEY` inside frontend code or upload it to GitHub. GitHub Pages is static hosting, so it cannot safely hide API keys.

Recommended portfolio setup:

```text
GitHub Pages portfolio -> link to deployed Streamlit app
```

Deploy the Streamlit app on a backend-friendly service such as Streamlit Community Cloud, Render, Railway, or Hugging Face Spaces, and store `GROQ_API_KEY` as a secret there.

## Keep Streamlit Awake

This repo includes a GitHub Actions workflow at `.github/workflows/keep-streamlit-awake.yml` that pings `https://rustamdurdyyev.streamlit.app/` every 12 hours.

If the deployed URL changes later, add the new app URL to GitHub:

1. Open the GitHub repository.
2. Go to `Settings` -> `Secrets and variables` -> `Actions`.
3. Add a repository variable named `STREAMLIT_APP_URL`.
4. Set its value to the new deployed Streamlit URL.

You can also start it manually from the `Actions` tab with the `Keep Streamlit Awake` workflow.

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_CV_PATH = "cv.json"
DEFAULT_CONTEXT_LIMIT = 6


@dataclass
class KnowledgeItem:
    item_type: str
    content: str


def load_dotenv_file(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


def load_cv(path=DEFAULT_CV_PATH):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_knowledge_items(cv):
    items = []

    personal_info = cv.get("personal_info", {})
    name = personal_info.get("full_name", "Rustam Durdyyev")

    summary = cv.get("professional_summary")
    if summary:
        items.append(KnowledgeItem("summary", f"Name: {name}\nSummary: {summary}"))

    technical_skills = cv.get("technical_skills", {})
    skill_lines = []
    for label, key in [
        ("Programming Languages", "programming_languages"),
        ("Machine Learning", "machine_learning"),
        ("Python Libraries", "python_libraries"),
        ("Statistical Analysis", "statistical_analysis"),
        ("Cloud and HPC", "cloud_and_hpc"),
        ("Simulation and Molecular Modelling", "simulation_and_molecular_modelling"),
    ]:
        values = technical_skills.get(key, [])
        if values:
            skill_lines.append(f"{label}: {', '.join(values)}")
    if skill_lines:
        items.append(KnowledgeItem("skills", "\n".join(skill_lines)))

    for experience in cv.get("experience", []):
        responsibilities = " ".join(experience.get("responsibilities", []))
        text = "\n".join(
            [
                f"Role: {experience.get('role', '')}",
                f"Organization: {experience.get('organization', '')}",
                f"Location: {experience.get('location', '')}",
                f"Duration: {experience.get('start_date', '')} - {experience.get('end_date', '')}",
                f"Responsibilities: {responsibilities}",
            ]
        )
        items.append(KnowledgeItem("experience", text.strip()))

    for education in cv.get("education", []):
        text = "\n".join(
            [
                f"Degree: {education.get('degree', '')}",
                f"Field: {education.get('field', '')}",
                f"Institution: {education.get('institution', '')}",
                f"Location: {education.get('location', '')}",
                f"Duration: {education.get('start_date', '')} - {education.get('end_date', '')}",
            ]
        )
        items.append(KnowledgeItem("education", text.strip()))

    for publication in cv.get("publications", []):
        text = "\n".join(
            [
                f"Title: {publication.get('title', '')}",
                f"Year: {publication.get('year', '')}",
                f"Authors: {', '.join(publication.get('authors', []))}",
                f"Journal: {publication.get('journal', 'N/A')}",
                f"DOI/Link: {publication.get('doi', publication.get('link', 'N/A'))}",
            ]
        )
        items.append(KnowledgeItem("publication", text.strip()))

    for award in cv.get("awards_and_grants", []):
        items.append(KnowledgeItem("award", award))

    for language in cv.get("languages", []):
        text = f"{language.get('language', '')}: {language.get('proficiency', '')}"
        items.append(KnowledgeItem("language", text.strip()))

    for activity in cv.get("portfolio_activities", []):
        lines = [
            f"Title: {activity.get('title', '')}",
            f"Category: {activity.get('category', '')}",
            f"Summary: {activity.get('summary', '')}",
        ]
        technologies = activity.get("technologies", [])
        if technologies:
            lines.append(f"Technologies: {', '.join(technologies)}")
        if activity.get("link"):
            lines.append(f"Link: {activity['link']}")

        items.append(KnowledgeItem("portfolio_activity", "\n".join(lines).strip()))

    return items


def normalize_words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def select_context_items(question, items, limit=DEFAULT_CONTEXT_LIMIT):
    question_lower = question.lower()
    field_keywords = {
        "publication": ["publication", "paper", "article", "journal", "doi"],
        "award": ["award", "honor", "scholarship", "grant"],
        "language": ["language", "speak"],
        "education": ["education", "study", "degree", "university", "school", "phd", "doctorate"],
        "experience": ["experience", "work", "career", "job", "role"],
        "skills": ["skill", "python", "machine learning", "programming", "cloud"],
        "portfolio_activity": [
            "project",
            "portfolio",
            "activity",
            "activities",
            "blog",
            "bookshelf",
            "sudoku",
            "crop",
            "soccer",
            "football",
            "manchester",
            "running",
            "hiking",
            "parkrun",
            "outside work",
            "personal",
        ],
    }

    requested_types = [
        item_type
        for item_type, keywords in field_keywords.items()
        if any(keyword in question_lower for keyword in keywords)
    ]

    if requested_types:
        return [item for item in items if item.item_type in requested_types]

    question_words = normalize_words(question)
    scored_items = []
    for index, item in enumerate(items):
        score = len(question_words.intersection(normalize_words(item.content)))
        if score:
            scored_items.append((score, -index, item))

    if not scored_items:
        return items[:limit]

    scored_items.sort(reverse=True)
    return [item for _, _, item in scored_items[:limit]]


def ask_groq(question, context, model=None):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Add it to a .env file or run this in PowerShell: "
            "$env:GROQ_API_KEY='your_groq_key_here'"
        )

    model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are DuRu, Rustam Durdyyev's personal assistant. "
                    "Answer using only the provided CV and portfolio context. "
                    "If the context does not contain the answer, say that it is not listed in the CV data. "
                    "Keep answers clear, friendly, and portfolio-ready."
                ),
            },
            {
                "role": "user",
                "content": f"CV context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
        "temperature": 0.3,
        "max_completion_tokens": 500,
    }

    request = urllib.request.Request(
        GROQ_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "duru-cv-assistant/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        if error.code == 401:
            detail = "Invalid or missing Groq API key. Create a new key at https://console.groq.com/keys."
        elif error.code == 403 and "1010" in detail:
            detail = (
                "Groq/Cloudflare blocked this request before it reached the API. "
                "This is usually caused by VPN/proxy/network restrictions or a blocked client/IP. "
                "Try turning off VPN, using another network, or testing from Groq Console Playground."
            )
        elif error.code == 429:
            detail = "Groq free-plan rate limit reached. Wait a little and try again."
        elif error.code == 400 and "model" in detail.lower():
            detail = f"The model '{model}' is not available for this account. Try setting GROQ_MODEL to another model."
        raise RuntimeError(f"Groq API error {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach Groq API: {error.reason}") from error

    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as error:
        raise RuntimeError(f"Unexpected Groq API response: {result}") from error


def main():
    parser = argparse.ArgumentParser(description="Chat with Rustam's CV using the Groq API.")
    parser.add_argument("--cv", default=DEFAULT_CV_PATH, help="Path to the CV JSON file.")
    parser.add_argument("--model", default=None, help=f"Groq model name. Default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--context-limit",
        type=int,
        default=DEFAULT_CONTEXT_LIMIT,
        help="Number of relevant CV items to send when no exact section is requested.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the selected CV context before sending the question.",
    )
    args = parser.parse_args()

    load_dotenv_file()
    items = build_knowledge_items(load_cv(args.cv))

    print("Hello, I am DuRu, Rustam's personal CV assistant.")
    print("Ask me about his skills, education, publications, experience, projects, or activities.")
    print("Type q anytime to quit.")

    while True:
        question = input("\nHow can I help you? ").strip()
        if question.lower() == "q":
            print("Goodbye. DuRu is signing off.")
            break

        context_items = select_context_items(question, items, limit=args.context_limit)
        context = "\n\n".join(item.content for item in context_items)

        if args.show_context:
            print(f"\nSelected context:\n{context}")

        try:
            answer = ask_groq(question, context, model=args.model)
        except RuntimeError as error:
            print(f"\nError: {error}")
            sys.exit(1)

        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()

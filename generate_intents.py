import json
import os
import pathlib
import re
import ollama
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "info@ingrainsystem.com")
CONTACT_PHONE = os.getenv("CONTACT_PHONE", "+91-9010481048")

BASE_DIR = pathlib.Path(__file__).parent.parent
RAW_PAGES_PATH = BASE_DIR / "backend" / "data" / "raw" / "iryax_pages.json"
INTENTS_PATH = BASE_DIR / "backend" / "data" / "processed" / "structured_intents.json"

INTENTS_CONFIG = [
    {
        "name": "pricing",
        "keywords": ["price", "prices", "pricing", "cost", "costs", "fee", "fees", "subscription", "rate", "rates"],
        "phrases": ["price list", "list price", "pricing plans", "how much", "subscription plans", "software pricing"],
        "exact_queries": ["prices", "pricing", "cost", "price list", "list price", "list plans"],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract ONLY the plan names and their prices into a simple markdown bulleted list. Do NOT include any descriptions, explanations, or features. Do NOT use any markdown headings (like ###). Start directly with 'Here is the price list from Iryax Global:'."
    },
    {
        "name": "basic_plan",
        "keywords": ["basic plan", "free plan", "free tier", "basic tier"],
        "phrases": ["what is the basic plan", "details of basic plan", "free plan features", "what do i get for free"],
        "exact_queries": ["basic plan", "free plan"],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract the exact details of the 'Basic Plan'. List its price and all the features included. Format as a markdown list."
    },
    {
        "name": "premium_plan",
        "keywords": ["premium plan", "premium tier"],
        "phrases": ["what is the premium plan", "details of premium plan", "premium plan features"],
        "exact_queries": ["premium plan"],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract the exact details of the 'Premium' plan. List its price and all the features included. Format as a markdown list."
    },
    {
        "name": "enterprise_plan",
        "keywords": ["enterprise plan", "enterprise tier"],
        "phrases": ["what is the enterprise plan", "details of enterprise plan", "enterprise plan features"],
        "exact_queries": ["enterprise plan"],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract the exact details of the 'Enterprise' plan. List its price and all the features included. Format as a markdown list."
    },
    {
        "name": "custom_plan",
        "keywords": ["custom plan", "bespoke plan", "custom pricing"],
        "phrases": ["what is the custom plan", "details of custom plan", "custom plan features"],
        "exact_queries": ["custom plan"],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract the exact details of the 'Custom' plan. List all the features included. Format as a markdown list."
    },
    {
        "name": "products",
        "keywords": ["product", "products", "module", "modules"],
        "phrases": ["list products", "what products", "core products", "what modules", "what do you offer", "your products", "list of products", "product list", "list price products", "all products", "all modules"],
        "exact_queries": ["products", "modules"],
        "url_source": "https://iryax.com/products",
        "prompt": "Extract ONLY the actual core software products and modules (e.g. Recruitment, Payroll, Task Management) from the text. Ignore marketing slogans or category headers like 'HIRE SMARTER', 'SYSTEMS', or 'PRODUCTIVITY'. Format as a clean, simple markdown numbered list of just the product names, each with a brief 1-sentence description. Do not add conversational filler. Start directly with 'Here is the list of Iryax Global's core products:'."
    },
    {
        "name": "recruitment_product",
        "keywords": ["recruitment", "hiring", "hire", "ats", "applicant tracking", "candidate", "job posting", "interviews"],
        "phrases": ["recruitment module", "how does recruitment work", "hiring software", "recruitment features", "what is recruitment"],
        "exact_queries": ["recruitment", "hiring software"],
        "url_source": "https://iryax.com/recruitment",
        "prompt": "Summarize the key features of the Recruitment product. Mention features like applicant tracking, job postings, and interviews. Keep it strictly to 1 or 2 short sentences. Do not use bullet points or lists."
    },
    {
        "name": "payroll_product",
        "keywords": ["payroll", "attendance", "leave", "timesheet", "shifts", "roster", "salary"],
        "phrases": ["payroll module", "attendance management", "how does payroll work", "payroll features", "leave management"],
        "exact_queries": ["payroll", "attendance"],
        "url_source": "https://iryax.com/attendance",
        "prompt": "Summarize the key features of the Payroll and Attendance product. Mention time tracking, leave management, and payroll processing. Keep it strictly to 1 or 2 short sentences. Do NOT use any headings (like ###) or bullet points. Output only plain text."
    },
    {
        "name": "task_product",
        "keywords": ["task", "tasks", "project", "projects", "sprints", "agile", "collaboration", "planning"],
        "phrases": ["task management", "project management", "how does task management work", "task features", "sprint planning"],
        "exact_queries": ["task management", "projects"],
        "url_source": "https://iryax.com/task-management",
        "prompt": "Summarize the key features of the Task Management product. Mention features like task planning, automation, and team collaboration. Keep it strictly to 1 or 2 short sentences. Do NOT use any headings (like ###) or bullet points. Output only plain text."
    },
    {
        "name": "camp_product",
        "keywords": ["camp", "camps", "medical camp", "health camp", "patients", "volunteers"],
        "phrases": ["medical camp management", "how do medical camps work", "health camp software", "camp features"],
        "exact_queries": ["medical camps", "camps"],
        "url_source": "https://iryax.com/camp",
        "prompt": "Summarize the key features of the Medical Camp Management product. Mention features like volunteer coordination, patient registration, and health reports. Keep it strictly to 1 or 2 short sentences. Do NOT use any headings (like ###) or bullet points. Output only plain text."
    },
    {
        "name": "workspace_product",
        "keywords": ["workspace", "coworking", "office", "cabin", "desk", "booking", "room", "iot"],
        "phrases": ["coworking space management", "workspace features", "how to book a cabin", "room booking", "smart office"],
        "exact_queries": ["workspace", "coworking"],
        "url_source": "https://iryax.com/workspace",
        "prompt": "Summarize the key features of the Workspace Management product. Mention features like desk booking, cabin reservations, and IoT climate/AV control. Keep it strictly to 1 or 2 short sentences. Do NOT use any headings (like ###) or bullet points. Output only plain text."
    },
    {
        "name": "demo",
        "keywords": ["demo", "book", "schedule", "meeting", "call"],
        "phrases": ["book a demo", "how to book a demo", "schedule a demo", "request a demo", "i want a demo"],
        "exact_queries": ["demo", "book demo", "schedule demo"],
        "url_source": "https://iryax.com/contact",
        "prompt": "Extract the exact steps and details required to book a demo. Mention the fields required (like Full Name, Work Email, etc). Format as a clean markdown list."
    },
    {
        "name": "how_it_works",
        "keywords": ["how", "works", "process", "workflow"],
        "phrases": ["how does it work", "how it works", "explain the process", "how to use iryax"],
        "exact_queries": ["how it works", "how does it work"],
        "url_source": "https://iryax.com/how-it-works",
        "prompt": "Summarize how the platform works based on the text. Format it cleanly with markdown. Start directly with 'Here is how Iryax Global works:'."
    },
    {
        "name": "security",
        "keywords": ["security", "privacy", "data", "protection", "gdpr", "secure"],
        "phrases": ["is my data secure", "privacy policy", "data protection", "security features"],
        "exact_queries": ["security", "privacy", "data privacy"],
        "url_source": "https://iryax.com/privacy-policy",
        "prompt": "Summarize the Privacy Policy and Security measures. Mention what information is collected, how it is protected, and that payments are secure. Format it cleanly with markdown."
    },
    {
        "name": "contact",
        "keywords": ["contact", "email", "phone", "address", "office", "location"],
        "phrases": ["how to contact", "where is your office", "phone number", "email address", "contact info", "contact details", "reach out"],
        "exact_queries": ["contact", "email", "phone", "address"],
        "url_source": "https://iryax.com/contact",
        "prompt": "Extract the contact information (Email, Phone/WhatsApp, and Office Address). Format it cleanly as a markdown bulleted list. Do not add conversational filler. Start directly with 'Here is the contact information for Iryax Global:'."
    },
    {
        "name": "careers",
        "keywords": ["job", "jobs", "career", "careers", "opening", "openings", "vacancy", "vacancies", "hiring", "recruit", "position", "positions", "internship", "internships"],
        "phrases": ["job openings", "career opportunities", "work with us", "hiring"],
        "exact_queries": ["jobs", "careers"],
        "url_source": "https://iryax.com/careers",
        "prompt": "Summarize the active job openings across different departments. Format it cleanly with markdown headers. Do not add conversational filler. Start directly with 'Here is information about active careers and job openings at Iryax Global:'."
    },
    {
        "name": "about",
        "keywords": [],
        "phrases": ["about iryax", "who is iryax", "what is iryax", "tell me about iryax", "company overview", "what does iryax do", "about your company", "about company", "iryax overview", "tell me about company", "explain iryax", "what is your company"],
        "exact_queries": ["about", "overview", "company"],
        "url_source": "https://iryax.com/about",
        "prompt": "Summarize what Iryax Global is and what problems it solves based on the text. Format it cleanly with markdown. Do not add conversational filler. Start directly with 'Iryax Global is an enterprise technology company'."
    },
    {
        "name": "greeting",
        "keywords": ["hi", "hello", "hey", "greetings", "morning", "evening"],
        "phrases": ["hi there", "hello there", "good morning", "good evening", "how are you", "how are you doing"],
        "exact_queries": ["hi", "hello", "hey", "hii", "helloo"],
        "static_answer": "Hello! I am Iryax Assistant. I can help you find information about our products, pricing, careers, and contact details. How can I help you today?"
    },
    {
        "name": "identity",
        "keywords": ["who are you", "what are you", "are you a bot"],
        "phrases": ["tell me who you are", "what is your name"],
        "exact_queries": ["who are you", "who are you?", "what are you", "what are you?", "are you a bot", "are you a bot?"],
        "static_answer": "I am Iryax Assistant, the official AI for Iryax Global. I can help you with products, pricing, careers, and contact info."
    },
]

def generate_structured_intents():
    print(f"[IntentGen] Starting LLM intent generation using {MODEL_NAME}...")
    if not RAW_PAGES_PATH.exists():
        print("[IntentGen] RAW_PAGES_PATH not found. Aborting.")
        return

    with open(RAW_PAGES_PATH, "r", encoding="utf-8") as f:
        pages = json.load(f)

    generated_intents = []

    for config in INTENTS_CONFIG:
        if "static_answer" in config:
            generated_intents.append({
                "name": config["name"],
                "keywords": config["keywords"],
                "phrases": config["phrases"],
                "exact_queries": config["exact_queries"],
                "answer": config["static_answer"],
                "sources": []
            })
            print(f"[IntentGen] Added static intent: {config['name']}")
            continue
            
        url = config["url_source"]
        if url not in pages:
            print(f"[IntentGen] Skipping {config['name']} - URL {url} not scraped.")
            continue
            
        markdown = pages[url].get("markdown", "")
        # Clean markdown to reduce context size
        markdown = re.sub(r'!\[.*?\]\(.*?\)', '', markdown)
        markdown = markdown[:3000] # Cap length just in case
        
        print(f"[IntentGen] Generating intent: {config['name']}")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant for Iryax Global. Provide output exactly as requested."},
            {"role": "user", "content": f"{config['prompt']}\n\nContext text:\n{markdown}"}
        ]
        
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                options={"temperature": 0.1, "num_predict": 500}
            )
            answer = response["message"]["content"].strip()
            
            # Ensure contact details are appended
            if "For additional information" not in answer:
                answer += f"\n\nFor additional information, please contact us at {CONTACT_EMAIL} or call {CONTACT_PHONE}."
                
            generated_intents.append({
                "name": config["name"],
                "keywords": config["keywords"],
                "phrases": config["phrases"],
                "exact_queries": config["exact_queries"],
                "answer": answer,
                "sources": [url]
            })
        except Exception as e:
            print(f"[IntentGen] Failed to generate {config['name']}: {e}")

    if generated_intents:
        # Save to file
        INTENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INTENTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"intents": generated_intents}, f, indent=2, ensure_ascii=False)
        print(f"[IntentGen] Successfully saved {len(generated_intents)} generated intents to structured_intents.json.")

if __name__ == "__main__":
    generate_structured_intents()

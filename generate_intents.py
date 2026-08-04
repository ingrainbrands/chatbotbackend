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
        "keywords": ["price", "prices", "pricing", "cost", "costs", "fee", "fees", "subscription", "rate", "rates", "monthly", "yearly", "annual", "payment", "billing", "package", "packages", "plans", "tier", "tiers"],
        "phrases": ["price list", "list price", "pricing plans", "how much", "subscription plans", "software pricing", "how much does it cost", "what are your prices", "tell me about pricing", "what do you charge", "is it expensive", "affordable plans", "best price", "cheapest plan", "whats the cost", "monthly subscription", "annual subscription", "payment plans", "billing options"],
        "exact_queries": ["prices", "pricing", "cost", "price list", "list price", "list plans", "how much", "whats the price"],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract ONLY the plan names and their prices into a simple markdown bulleted list. Do NOT include any descriptions, explanations, or features. Do NOT use any markdown headings (like ###). Start directly with 'Here is the price list from Iryax Global:'."
    },
    {
        "name": "basic_plan",
        "keywords": ["basic plan", "free plan", "free tier", "basic tier", "starter", "starter plan", "beginner", "entry level", "cheapest", "minimal", "essential"],
        "phrases": ["what is the basic plan", "details of basic plan", "free plan features", "what do i get for free", "tell me about basic plan", "is there a free version", "whats included in basic", "basic plan price", "starter plan features", "entry level plan"],
        "exact_queries": ["basic plan", "free plan", "starter plan", "what is included in basic plan"],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract the exact details of the 'Basic Plan'. List its price and all the features included. Format as a markdown list."
    },
    {
        "name": "premium_plan",
        "keywords": ["premium plan", "premium tier", "advanced", "pro", "professional", "plus", "ultimate", "unlimited", "top tier", "best plan", "full access"],
        "phrases": ["what is the premium plan", "details of premium plan", "premium plan features", "tell me about premium", "whats in premium", "premium price", "professional plan", "advanced features", "pro version"],
        "exact_queries": ["premium plan", "professional plan", "what comes with premium"],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract the exact details of the 'Premium' plan. List its price and all the features included. Format as a markdown list."
    },
    {
        "name": "enterprise_plan",
        "keywords": ["enterprise plan", "enterprise tier", "corporate", "business", "organization", "large scale", "company", "team", "department", "multi user", "volume"],
        "phrases": ["what is the enterprise plan", "details of enterprise plan", "enterprise plan features", "tell me about enterprise", "corporate plan", "business plan", "organization pricing", "team plans", "company plan"],
        "exact_queries": ["enterprise plan", "business plan", "corporate pricing"],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract the exact details of the 'Enterprise' plan. List its price and all the features included. Format as a markdown list."
    },
    {
        "name": "custom_plan",
        "keywords": ["custom plan", "bespoke plan", "custom pricing", "tailored", "personalized", "customized", "flexible", "negotiable", "special", "unique"],
        "phrases": ["what is the custom plan", "details of custom plan", "custom plan features", "can i get a custom plan", "bespoke solution", "tailored pricing", "flexible plans", "customized solution", "special pricing", "negotiable rates"],
        "exact_queries": ["custom plan", "bespoke plan", "can i customize"],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract the exact details of the 'Custom' plan. List all the features included. Format as a markdown list."
    },
    {
        "name": "products",
        "keywords": ["product", "products", "module", "modules", "features", "offerings", "solutions", "services", "software", "tools", "applications", "platform", "system", "integrations"],
        "phrases": ["list products", "what products", "core products", "what modules", "what do you offer", "your products", "list of products", "product list", "list price products", "all products", "all modules", "tell me about your products", "what software do you have", "what solutions do you provide", "what tools are available", "product catalog", "product offerings"],
        "exact_queries": ["products", "modules", "what do you offer", "list of all products"],
        "url_source": "https://iryax.com/products",
        "prompt": "Extract ONLY the actual core software products and modules (e.g. Recruitment, Payroll, Task Management) from the text. Ignore marketing slogans or category headers like 'HIRE SMARTER', 'SYSTEMS', or 'PRODUCTIVITY'. Format as a clean, simple markdown numbered list of just the product names, each with a brief 1-sentence description. Do not add conversational filler. Start directly with 'Here is the list of Iryax Global's core products:'."
    },
    {
        "name": "recruitment_product",
        "keywords": ["recruitment", "hiring", "hire", "ats", "applicant tracking", "candidate", "job posting", "interviews", "talent", "staffing", "applicants", "resume", "screening", "onboarding", "job board"],
        "phrases": ["recruitment module", "how does recruitment work", "hiring software", "recruitment features", "what is recruitment", "tell me about recruitment", "applicant tracking system", "candidate management", "job posting features", "interview scheduling", "talent acquisition", "hiring process", "recruitment tools"],
        "exact_queries": ["recruitment", "hiring software", "ats", "applicant tracking"],
        "url_source": "https://iryax.com/recruitment",
        "prompt": "Summarize the key features of the Recruitment product. Mention features like applicant tracking, job postings, and interviews. Keep it strictly to 1 or 2 short sentences. Do not use bullet points or lists."
    },
    {
        "name": "payroll_product",
        "keywords": ["payroll", "attendance", "leave", "timesheet", "shifts", "roster", "salary", "wages", "compensation", "benefits", "deductions", "tax", "overtime", "time tracking", "absence", "holidays", "working hours"],
        "phrases": ["payroll module", "attendance management", "how does payroll work", "payroll features", "leave management", "tell me about payroll", "time tracking system", "shift scheduling", "salary processing", "attendance tracking", "leave requests", "timesheet management", "overtime tracking", "payroll processing", "employee attendance"],
        "exact_queries": ["payroll", "attendance", "time tracking", "leave management"],
        "url_source": "https://iryax.com/attendance",
        "prompt": "Summarize the key features of the Payroll and Attendance product. Mention time tracking, leave management, and payroll processing. Keep it strictly to 1 or 2 short sentences. Do NOT use any headings (like ###) or bullet points. Output only plain text."
    },
    {
        "name": "task_product",
        "keywords": ["task", "tasks", "project", "projects", "sprints", "agile", "collaboration", "planning", "management", "workflow", "productivity", "deadline", "milestone", "progress", "teamwork", "assignment", "tracking"],
        "phrases": ["task management", "project management", "how does task management work", "task features", "sprint planning", "tell me about task management", "agile project management", "team collaboration tools", "workflow automation", "project planning", "task assignment", "progress tracking", "deadline management", "collaboration features", "team productivity"],
        "exact_queries": ["task management", "projects", "project management"],
        "url_source": "https://iryax.com/task-management",
        "prompt": "Summarize the key features of the Task Management product. Mention features like task planning, automation, and team collaboration. Keep it strictly to 1 or 2 short sentences. Do NOT use any headings (like ###) or bullet points. Output only plain text."
    },
    {
        "name": "camp_product",
        "keywords": ["camp", "camps", "medical camp", "health camp", "patients", "volunteers", "healthcare", "clinic", "medical", "field", "outreach", "mobile clinic", "health services", "community", "screening"],
        "phrases": ["medical camp management", "how do medical camps work", "health camp software", "camp features", "tell me about medical camps", "healthcare camp management", "patient registration", "volunteer coordination", "health screenings", "community health", "mobile medical camps", "field clinics"],
        "exact_queries": ["medical camps", "camps", "health camp"],
        "url_source": "https://iryax.com/camp",
        "prompt": "Summarize the key features of the Medical Camp Management product. Mention features like volunteer coordination, patient registration, and health reports. Keep it strictly to 1 or 2 short sentences. Do NOT use any headings (like ###) or bullet points. Output only plain text."
    },
    {
        "name": "workspace_product",
        "keywords": ["workspace", "coworking", "office", "cabin", "desk", "booking", "room", "iot", "workspace management", "coworking space", "meeting room", "conference room", "hot desk", "shared office", "space booking", "facility management", "smart office", "climate control", "av control"],
        "phrases": ["coworking space management", "workspace features", "how to book a cabin", "room booking", "smart office", "tell me about workspace", "desk booking", "meeting room reservation", "cabin rental", "office space management", "iot workspace", "smart workspace", "conference room booking"],
        "exact_queries": ["workspace", "coworking", "office booking", "room booking"],
        "url_source": "https://iryax.com/workspace",
        "prompt": "Summarize the key features of the Workspace Management product. Mention features like desk booking, cabin reservations, and IoT climate/AV control. Keep it strictly to 1 or 2 short sentences. Do NOT use any headings (like ###) or bullet points. Output only plain text."
    },
    {
        "name": "demo",
        "keywords": ["demo", "book", "schedule", "meeting", "call", "consultation", "walkthrough", "presentation", "show", "tour", "trial", "test", "demonstration", "live demo", "product tour"],
        "phrases": ["book a demo", "how to book a demo", "schedule a demo", "request a demo", "i want a demo", "can i get a demo", "see a demo", "product demonstration", "live walkthrough", "demo request", "schedule a meeting", "consultation call", "product tour", "show me how it works"],
        "exact_queries": ["demo", "book demo", "schedule demo", "request demo", "live demo"],
        "url_source": "https://iryax.com/contact",
        "prompt": "Extract the exact steps and details required to book a demo. Mention the fields required (like Full Name, Work Email, etc). Format as a clean markdown list."
    },
    {
        "name": "how_it_works",
        "keywords": ["how", "works", "process", "workflow", "steps", "procedure", "flow", "mechanism", "guide", "tutorial", "implementation", "setup", "getting started"],
        "phrases": ["how does it work", "how it works", "explain the process", "how to use iryax", "tell me how it works", "whats the workflow", "step by step", "implementation process", "getting started guide", "setup process", "how to start", "how to implement", "workflow overview", "process flow"],
        "exact_queries": ["how it works", "how does it work", "workflow", "how to use"],
        "url_source": "https://iryax.com/how-it-works",
        "prompt": "Summarize how the platform works based on the text. Format it cleanly with markdown. Start directly with 'Here is how Iryax Global works:'."
    },
    {
        "name": "security",
        "keywords": ["security", "privacy", "data", "protection", "gdpr", "secure", "encryption", "confidentiality", "safe", "trust", "compliance", "policy", "terms", "protection", "safety"],
        "phrases": ["is my data secure", "privacy policy", "data protection", "security features", "tell me about security", "how is data protected", "gdpr compliance", "data privacy", "secure platform", "trustworthy", "security measures", "encryption methods", "data safety", "privacy policy details"],
        "exact_queries": ["security", "privacy", "data privacy", "gdpr", "is it secure"],
        "url_source": "https://iryax.com/privacy-policy",
        "prompt": "Summarize the Privacy Policy and Security measures. Mention what information is collected, how it is protected, and that payments are secure. Format it cleanly with markdown."
    },
    {
        "name": "contact",
        "keywords": ["contact", "email", "phone", "address", "office", "location", "reach", "support", "help", "customer service", "headquarters", "offices", "map", "directions"],
        "phrases": ["how to contact", "where is your office", "phone number", "email address", "contact info", "contact details", "reach out", "get in touch", "support email", "helpdesk", "customer care", "office location", "headquarters address", "support phone", "whatsapp number"],
        "exact_queries": ["contact", "email", "phone", "address", "office location", "support"],
        "url_source": "https://iryax.com/contact",
        "prompt": "Extract the contact information (Email, Phone/WhatsApp, and Office Address). Format it cleanly as a markdown bulleted list. Do not add conversational filler. Start directly with 'Here is the contact information for Iryax Global:'."
    },
    {
        "name": "careers",
        "keywords": ["job", "jobs", "career", "careers", "opening", "openings", "vacancy", "vacancies", "hiring", "recruit", "position", "positions", "internship", "internships", "employment", "work", "opportunity", "opportunities", "role", "roles", "staff", "employee", "talent", "recruitment", "apply", "application"],
        "phrases": ["job openings", "career opportunities", "work with us", "hiring", "tell me about careers", "open positions", "current vacancies", "how to apply", "join our team", "career page", "employment opportunities", "vacant positions", "job vacancies", "internship opportunities", "jobs at iryax", "apply for job", "current openings"],
        "exact_queries": ["jobs", "careers", "vacancies", "hiring", "current openings"],
        "url_source": "https://iryax.com/careers",
        "prompt": "Summarize the active job openings across different departments. Format it cleanly with markdown headers. Do not add conversational filler. Start directly with 'Here is information about active careers and job openings at Iryax Global:'."
    },
    {
        "name": "about",
        "keywords": ["about", "company", "overview", "history", "mission", "vision", "values", "story", "founder", "team", "culture", "background", "info", "introduction"],
        "phrases": ["about iryax", "who is iryax", "what is iryax", "tell me about iryax", "company overview", "what does iryax do", "about your company", "about company", "iryax overview", "tell me about company", "explain iryax", "what is your company", "background of iryax", "company history", "mission and vision", "what does your company do"],
        "exact_queries": ["about", "overview", "company", "what is iryax", "who are you"],
        "url_source": "https://iryax.com/about",
        "prompt": "Summarize what Iryax Global is and what problems it solves based on the text. Format it cleanly with markdown. Do not add conversational filler. Start directly with 'Iryax Global is an enterprise technology company'."
    },
    {
        "name": "greeting",
        "keywords": ["hi", "hello", "hey", "greetings", "morning", "evening", "sup", "yo", "howdy", "hiya", "hey there", "good day", "afternoon"],
        "phrases": ["hi there", "hello there", "good morning", "good evening", "how are you", "how are you doing", "hey, hows it going", "nice to meet you", "greetings", "good afternoon", "hello everyone", "hi everyone"],
        "exact_queries": ["hi", "hello", "hey", "hii", "helloo", "good morning", "good evening"],
        "static_answer": "Hello! I am Iryax Assistant. I can help you find information about our products, pricing, careers, and contact details. How can I help you today?"
    },
    {
        "name": "identity",
        "keywords": ["who are you", "what are you", "are you a bot", "ai", "assistant", "your name", "chatbot", "automated", "virtual assistant", "ai assistant"],
        "phrases": ["tell me who you are", "what is your name", "are you a real person", "are you automated", "whats your name", "tell me about yourself", "who am i talking to", "is this a bot", "are you an ai", "virtual assistant", "who made you"],
        "exact_queries": ["who are you", "who are you?", "what are you", "what are you?", "are you a bot", "are you a bot?", "whats your name", "tell me about yourself"],
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

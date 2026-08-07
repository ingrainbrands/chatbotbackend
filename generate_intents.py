import json
import os
import pathlib
import re
import ollama
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "info@ingrainsystem.com")
CONTACT_PHONE = os.getenv("CONTACT_PHONE", "+91-9010481048")

ROOT_DIR = pathlib.Path(__file__).parent
RAW_PAGES_PATH = ROOT_DIR / "data" / "raw" / "iryax_pages.json"
INTENTS_PATH = ROOT_DIR / "data" / "processed" / "structured_intents.json"

INTENTS_CONFIG = [
    {
        "name": "pricing",
        "keywords": [
            "price", "prices", "pricing", "cost", "costs", "fee", "fees",
            "subscription", "rate", "rates", "monthly", "yearly", "annual",
            "payment", "billing", "package", "packages", "plans", "plan",
            "tier", "tiers", "charge", "charges", "amount", "budget",
            "affordable", "expensive", "cheap", "cheapest", "value",
            "how much", "quote", "quotation", "invoice", "rupees", "inr",
            "per user", "per month", "per year", "cost per user",
            "license", "licensing", "licence", "renewal", "renew"
        ],
        "phrases": [
            "price list", "list price", "pricing plans", "how much",
            "subscription plans", "software pricing", "how much does it cost",
            "what are your prices", "tell me about pricing", "what do you charge",
            "is it expensive", "affordable plans", "best price", "cheapest plan",
            "whats the cost", "monthly subscription", "annual subscription",
            "payment plans", "billing options", "how much will it cost",
            "what is the pricing", "show me pricing", "pricing details",
            "what are the charges", "what is the fee", "what all plans are there",
            "how much is the subscription", "plan pricing", "cost of plans",
            "is there a free plan", "i need pricing info", "give me price details",
            "cost for my company", "pricing for small business",
            "per user pricing", "monthly cost", "yearly cost",
            "what is the charge", "cost breakdown", "pricing structure",
            "what does it cost", "price per month", "price per year"
        ],
        "exact_queries": [
            "prices", "pricing", "cost", "price list", "list price",
            "list plans", "how much", "whats the price", "what is the price",
            "plans and pricing", "price", "fees", "charges", "rate"
        ],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract all pricing plans into a single compact bulleted list. Each plan must be 1 short single-line bullet in this exact format: • **[Plan Name]** ([Price]) — [3-4 top key features separated by commas]. Do not output sub-bullets, headings, or extra lines. Start directly with 'Here are the pricing plans from Iryax Global:'."
    },
    {
        "name": "basic_plan",
        "keywords": [
            "basic plan", "free plan", "free tier", "basic tier",
            "starter", "starter plan", "beginner", "entry level",
            "cheapest", "minimal", "essential", "basic", "low cost",
            "introductory", "trial plan", "light plan", "simple plan",
            "small business plan", "small team plan", "cheapest option",
            "minimum plan", "starting plan", "first plan", "lowest plan"
        ],
        "phrases": [
            "what is the basic plan", "details of basic plan",
            "free plan features", "what do i get for free",
            "tell me about basic plan", "is there a free version",
            "whats included in basic", "basic plan price",
            "starter plan features", "entry level plan",
            "what comes in basic plan", "basic plan details",
            "basic plan cost", "how much is the basic plan",
            "is there a cheapest plan", "which is the lowest plan",
            "can i start for free", "minimum cost plan",
            "what is included in basic", "basic tier details",
            "most affordable plan", "small company plan",
            "plan for startups", "plan for small teams"
        ],
        "exact_queries": [
            "basic plan", "free plan", "starter plan",
            "what is included in basic plan", "basic", "starter"
        ],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract the exact details of the 'Basic Plan'. List its price and all the features included. Format as a markdown list."
    },
    {
        "name": "premium_plan",
        "keywords": [
            "premium plan", "premium tier", "advanced", "pro",
            "professional", "plus", "ultimate", "unlimited", "top tier",
            "best plan", "full access", "premium", "higher plan",
            "upgraded plan", "next plan", "better plan", "mid plan",
            "standard plan", "growth plan", "scale plan", "business plan",
            "most popular", "recommended plan", "popular option"
        ],
        "phrases": [
            "what is the premium plan", "details of premium plan",
            "premium plan features", "tell me about premium",
            "whats in premium", "premium price", "professional plan",
            "advanced features", "pro version", "what is premium",
            "premium plan cost", "how much is premium",
            "premium plan details", "what do i get in premium",
            "upgrade to premium", "premium vs basic",
            "which plan is better", "middle plan", "mid tier plan",
            "recommended plan for me", "what does premium include",
            "premium benefits", "premium advantages", "best value plan"
        ],
        "exact_queries": [
            "premium plan", "professional plan", "what comes with premium",
            "premium", "pro plan", "advanced plan"
        ],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract the exact details of the 'Premium' plan. List its price and all the features included. Format as a markdown list."
    },
    {
        "name": "enterprise_plan",
        "keywords": [
            "enterprise plan", "enterprise tier", "corporate", "business",
            "organization", "large scale", "company", "team", "department",
            "multi user", "volume", "enterprise", "bulk", "large team",
            "big company", "corporation", "conglomerate", "enterprise pricing",
            "enterprise features", "enterprise license", "unlimited users",
            "scalable plan", "large organization", "500 users", "1000 users",
            "many employees", "all departments", "company wide", "group plan",
            "group license", "b2b", "saas enterprise", "dedicated account"
        ],
        "phrases": [
            "what is the enterprise plan", "details of enterprise plan",
            "enterprise plan features", "tell me about enterprise",
            "corporate plan", "business plan", "organization pricing",
            "team plans", "company plan", "enterprise pricing",
            "plan for large companies", "plan for big teams",
            "enterprise plan cost", "how much is enterprise",
            "enterprise license cost", "unlimited users plan",
            "plan for 100 employees", "plan for large organizations",
            "what is enterprise tier", "bulk plan",
            "what does enterprise include", "enterprise benefits",
            "enterprise for my company", "enterprise for corporate",
            "volume discounts", "do you have group plans",
            "plan for government", "plan for schools", "plan for hospitals"
        ],
        "exact_queries": [
            "enterprise plan", "business plan", "corporate pricing",
            "enterprise", "large team plan", "plan for big company"
        ],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract the exact details of the 'Enterprise' plan. List its price and all the features included. Format as a markdown list."
    },
    {
        "name": "custom_plan",
        "keywords": [
            "custom plan", "bespoke plan", "custom pricing", "tailored",
            "personalized", "customized", "flexible", "negotiable", "special",
            "unique", "custom", "own plan", "build your plan",
            "design plan", "special pricing", "tailor made", "made to order",
            "my own requirements", "specific needs", "special requirements",
            "adjusted plan", "modified plan", "own package",
            "add ons", "add on features", "selective features",
            "pick and choose", "mix and match", "bundle", "custom bundle"
        ],
        "phrases": [
            "what is the custom plan", "details of custom plan",
            "custom plan features", "can i get a custom plan",
            "bespoke solution", "tailored pricing", "flexible plans",
            "customized solution", "special pricing", "negotiable rates",
            "can i pick my own features", "i need a custom solution",
            "can you make a plan for me", "i have specific requirements",
            "personalized plan", "do you offer customization",
            "can we negotiate pricing", "is there a custom option",
            "build my own plan", "design my own package",
            "do you have add on features", "selective modules",
            "can i choose what i need", "tailored to my business",
            "custom enterprise plan", "special deal",
            "can i get a special price", "i want something different"
        ],
        "exact_queries": [
            "custom plan", "bespoke plan", "can i customize",
            "custom", "customized plan", "personalized plan"
        ],
        "url_source": "https://iryax.com/price",
        "prompt": "Extract the exact details of the 'Custom' plan. List all the features included. Format as a markdown list."
    },
    {
        "name": "products",
        "keywords": [
            "product", "products", "module", "modules", "features", "offerings",
            "solutions", "services", "software", "tools", "applications",
            "platform", "system", "integrations", "what you offer",
            "capabilities", "functionality", "what does iryax do",
            "suite", "product suite", "software suite", "hr software",
            "hr tools", "business software", "erp", "hrms", "hris",
            "management software", "enterprise software", "all features",
            "complete solution", "full suite", "what can you do",
            "list everything", "all modules", "all products"
        ],
        "phrases": [
            "list products", "what products", "core products", "what modules",
            "what do you offer", "your products", "list of products",
            "product list", "list price products", "all products", "all modules",
            "tell me about your products", "what software do you have",
            "what solutions do you provide", "what tools are available",
            "product catalog", "product offerings", "what services do you provide",
            "what can iryax do", "what does iryax offer", "your services",
            "list all services", "what are your capabilities",
            "complete product list", "tell me everything you offer",
            "full product list", "show me all products",
            "which products are available", "available modules",
            "do you have hr software", "do you have payroll software",
            "what business tools do you have", "what does this platform do"
        ],
        "exact_queries": [
            "products", "modules", "what do you offer", "list of all products",
            "services", "solutions", "all modules", "all products",
            "what are your products", "product list"
        ],
        "url_source": "https://iryax.com/products",
        "prompt": "Extract ONLY the actual core software products and modules (e.g. Recruitment, Payroll, Task Management) from the text. Ignore marketing slogans or category headers like 'HIRE SMARTER', 'SYSTEMS', or 'PRODUCTIVITY'. Format as a clean, simple markdown numbered list of just the product names, each with a brief 1-sentence description. Do not add conversational filler. Start directly with 'Here is the list of Iryax Global\\'s core products:'."
    },
    {
        "name": "recruitment_product",
        "keywords": [
            "recruitment", "hiring", "hire", "ats", "applicant tracking",
            "candidate", "job posting", "interviews", "talent", "staffing",
            "applicants", "resume", "screening", "onboarding", "job board",
            "recruiter", "hiring manager", "talent acquisition", "hr hiring",
            "job application", "employee selection", "shortlist",
            "interview scheduling", "job offer", "offer letter",
            "background verification", "pre employment", "cv screening",
            "bulk hiring", "campus hiring", "lateral hiring",
            "open positions", "job requisition", "headcount",
            "workforce planning", "manpower", "new hire"
        ],
        "phrases": [
            "recruitment module", "how does recruitment work",
            "hiring software", "recruitment features", "what is recruitment",
            "tell me about recruitment", "applicant tracking system",
            "candidate management", "job posting features",
            "interview scheduling", "talent acquisition", "hiring process",
            "recruitment tools", "how to post a job", "how to track applicants",
            "manage job applications", "track candidates",
            "recruitment automation", "hr recruitment",
            "how does hiring work in iryax", "onboarding new employees",
            "candidate pipeline", "shortlisting candidates",
            "screening resumes", "bulk recruitment",
            "how to hire employees using iryax", "recruiting software"
        ],
        "exact_queries": [
            "recruitment", "hiring software", "ats", "applicant tracking",
            "recruitment module", "hiring", "candidates", "job postings"
        ],
        "url_source": "https://iryax.com/recruitment",
        "prompt": "Summarize the key features of the Recruitment product. Mention features like applicant tracking, job postings, and interviews. Keep it strictly to 1 or 2 short sentences. Do not use bullet points or lists."
    },
    {
        "name": "payroll_product",
        "keywords": [
            "payroll", "attendance", "leave", "timesheet", "shifts", "roster",
            "salary", "wages", "compensation", "benefits", "deductions",
            "tax", "overtime", "time tracking", "absence", "holidays",
            "working hours", "payslip", "pay slip", "salary slip",
            "monthly salary", "salary processing", "salary calculation",
            "tds", "pf", "esi", "epf", "gratuity", "pay structure",
            "ctc", "gross salary", "net salary", "pay roll",
            "employee attendance", "daily attendance", "leave balance",
            "leave request", "sick leave", "casual leave", "annual leave",
            "half day", "late coming", "early departure", "punch in",
            "punch out", "biometric", "geofencing", "remote attendance",
            "wfh", "work from home attendance", "shift management",
            "night shift", "rotational shift", "weekly off", "holiday list",
            "national holiday", "payroll compliance", "statutory compliance"
        ],
        "phrases": [
            "payroll module", "attendance management", "how does payroll work",
            "payroll features", "leave management", "tell me about payroll",
            "time tracking system", "shift scheduling", "salary processing",
            "attendance tracking", "leave requests", "timesheet management",
            "overtime tracking", "payroll processing", "employee attendance",
            "how to process salary", "salary management software",
            "how does attendance work", "track employee hours",
            "manage employee leaves", "leave approval process",
            "payroll automation", "auto payroll", "payroll system",
            "hr payroll", "manage attendance remotely",
            "payslip generation", "salary slip software",
            "payroll for small business", "manage shifts",
            "track working hours", "calculate salary automatically",
            "statutory deductions", "compliance payroll",
            "how to manage employee attendance", "leave tracker"
        ],
        "exact_queries": [
            "payroll", "attendance", "time tracking", "leave management",
            "salary", "payroll module", "attendance management",
            "payroll system", "salary processing"
        ],
        "url_source": "https://iryax.com/attendance",
        "prompt": "Summarize the key features of the Payroll and Attendance product. Mention time tracking, leave management, and payroll processing. Keep it strictly to 1 or 2 short sentences. Do NOT use any headings (like ###) or bullet points. Output only plain text."
    },
    {
        "name": "task_product",
        "keywords": [
            "task", "tasks", "project", "projects", "sprints", "agile",
            "collaboration", "planning", "management", "workflow",
            "productivity", "deadline", "milestone", "progress", "teamwork",
            "assignment", "tracking", "kanban", "board", "backlog",
            "to do list", "todo", "checklist", "project tracking",
            "project planning", "team tasks", "assign task",
            "task status", "task update", "task completion",
            "project status", "work items", "jira like", "trello like",
            "team productivity", "daily tasks", "weekly tasks",
            "task reminder", "task notification", "overdue tasks",
            "project timeline", "gantt chart", "resource allocation",
            "team workload", "capacity planning", "project reporting"
        ],
        "phrases": [
            "task management", "project management",
            "how does task management work", "task features",
            "sprint planning", "tell me about task management",
            "agile project management", "team collaboration tools",
            "workflow automation", "project planning", "task assignment",
            "progress tracking", "deadline management",
            "collaboration features", "team productivity",
            "how to manage tasks", "assign tasks to team",
            "track project progress", "manage projects online",
            "team task management", "task management software",
            "how does project management work in iryax",
            "manage team tasks", "project collaboration",
            "sprint management", "agile task management",
            "remote team task management", "digital task board",
            "manage deadlines", "track team performance"
        ],
        "exact_queries": [
            "task management", "projects", "project management",
            "tasks", "task management module", "sprint", "agile"
        ],
        "url_source": "https://iryax.com/task-management",
        "prompt": "Summarize the key features of the Task Management product. Mention features like task planning, automation, and team collaboration. Keep it strictly to 1 or 2 short sentences. Do NOT use any headings (like ###) or bullet points. Output only plain text."
    },
    {
        "name": "camp_product",
        "keywords": [
            "camp", "camps", "medical camp", "health camp", "patients",
            "volunteers", "healthcare", "clinic", "medical", "field",
            "outreach", "mobile clinic", "health services", "community",
            "screening", "health checkup", "free checkup", "blood test",
            "diagnostic camp", "eye camp", "dental camp", "diabetes screening",
            "bp check", "blood pressure", "health drive", "ngo",
            "charity camp", "community health program", "rural health",
            "village camp", "corporate health camp", "employee health",
            "wellness camp", "annual health checkup", "camp coordinator",
            "camp organizer", "doctor attendance", "nurse attendance",
            "patient records", "medical records", "health report",
            "camp report", "camp analytics", "camp volunteers",
            "volunteer management", "patient registration", "camp registration"
        ],
        "phrases": [
            "medical camp management", "how do medical camps work",
            "health camp software", "camp features",
            "tell me about medical camps", "healthcare camp management",
            "patient registration", "volunteer coordination",
            "health screenings", "community health", "mobile medical camps",
            "field clinics", "how to manage a medical camp",
            "organize a health camp", "track patients in camp",
            "manage volunteers in camp", "camp management software",
            "health camp app", "community health outreach",
            "free health checkup management", "annual health camp",
            "ngo health camp software", "corporate wellness camp",
            "rural medical camp", "diagnostic camp management",
            "how to register patients in camp", "camp reporting",
            "health camp analytics", "doctor coordination in camp"
        ],
        "exact_queries": [
            "medical camps", "camps", "health camp",
            "medical camp management", "camp module", "health camps"
        ],
        "url_source": "https://iryax.com/camp",
        "prompt": "Summarize the key features of the Medical Camp Management product. Mention features like volunteer coordination, patient registration, and health reports. Keep it strictly to 1 or 2 short sentences. Do NOT use any headings (like ###) or bullet points. Output only plain text."
    },
    {
        "name": "workspace_product",
        "keywords": [
            "workspace", "coworking", "office", "cabin", "desk", "booking",
            "room", "iot", "workspace management", "coworking space",
            "meeting room", "conference room", "hot desk", "shared office",
            "space booking", "facility management", "smart office",
            "climate control", "av control", "rent a desk", "rent a cabin",
            "office rental", "book office space", "temporary office",
            "flexi desk", "flexible workspace", "work station",
            "private cabin", "semi private", "open desk", "day pass",
            "monthly desk", "co working membership", "office membership",
            "visitor management", "access control", "smart ac",
            "smart projector", "smart tv", "iot office", "connected office",
            "automated office", "book meeting room", "conference booking",
            "board room", "pod", "phone booth", "focus room", "quiet space",
            "event space", "training room", "seminar hall", "auditorium"
        ],
        "phrases": [
            "coworking space management", "workspace features",
            "how to book a cabin", "room booking", "smart office",
            "tell me about workspace", "desk booking",
            "meeting room reservation", "cabin rental",
            "office space management", "iot workspace", "smart workspace",
            "conference room booking", "how does workspace booking work",
            "how to reserve a desk", "book a conference room",
            "manage office space", "coworking software",
            "how to book office space", "workspace management software",
            "smart office management", "iot office management",
            "desk reservation system", "office booking system",
            "how to manage shared office", "coworking platform",
            "book a meeting room online", "manage workstations",
            "hot desking software", "hybrid workplace management",
            "remote office booking", "work from office booking"
        ],
        "exact_queries": [
            "workspace", "coworking", "office booking", "room booking",
            "workspace module", "desk booking", "cabin booking",
            "meeting room", "coworking space"
        ],
        "url_source": "https://iryax.com/workspace",
        "prompt": "Summarize the key features of the Workspace Management product. Mention features like desk booking, cabin reservations, and IoT climate/AV control. Keep it strictly to 1 or 2 short sentences. Do NOT use any headings (like ###) or bullet points. Output only plain text."
    },
    {
        "name": "demo",
        "keywords": [
            "demo", "book", "schedule", "meeting", "call", "consultation",
            "walkthrough", "presentation", "show", "tour", "trial",
            "test", "demonstration", "live demo", "product tour",
            "free demo", "online demo", "video call", "zoom call",
            "teams call", "google meet", "virtual meeting", "webinar",
            "explore product", "try the product", "see it in action",
            "hands on", "free trial", "pilot", "proof of concept",
            "poc", "request meeting", "sales call", "sales demo",
            "product preview", "trial period", "evaluation"
        ],
        "phrases": [
            "book a demo", "how to book a demo", "schedule a demo",
            "request a demo", "i want a demo", "can i get a demo",
            "see a demo", "product demonstration", "live walkthrough",
            "demo request", "schedule a meeting", "consultation call",
            "product tour", "show me how it works", "can i try it",
            "how do i schedule a demo", "book a free demo",
            "request a free trial", "i want to see the product",
            "show me the product", "can you give a demo",
            "how to request a demo", "set up a demo", "arrange a demo",
            "book a product tour", "can we have a call",
            "schedule a product walkthrough", "i want to evaluate the product",
            "is there a free trial", "can i test it first",
            "want to explore before buying", "pilot program"
        ],
        "exact_queries": [
            "demo", "book demo", "schedule demo", "request demo",
            "live demo", "free demo", "product demo", "trial", "book a demo"
        ],
        "url_source": "https://iryax.com/contact",
        "prompt": "Extract the exact steps and details required to book a demo. Mention the fields required (like Full Name, Work Email, etc). Format as a clean markdown list."
    },
    {
        "name": "how_it_works",
        "keywords": [
            "how", "works", "process", "workflow", "steps", "procedure",
            "flow", "mechanism", "guide", "tutorial", "implementation",
            "setup", "getting started", "onboarding", "deployment",
            "install", "integration", "configure", "configuration",
            "start using", "begin", "use it", "run it",
            "technical setup", "cloud based", "on premise", "saas",
            "web based", "mobile app", "api integration",
            "data migration", "import data", "export data",
            "user roles", "admin setup", "user management"
        ],
        "phrases": [
            "how does it work", "how it works", "explain the process",
            "how to use iryax", "tell me how it works", "whats the workflow",
            "step by step", "implementation process", "getting started guide",
            "setup process", "how to start", "how to implement",
            "workflow overview", "process flow", "how do i get started",
            "how do i begin", "how does iryax work", "how to set it up",
            "how to onboard", "is it cloud based", "how to install",
            "is it web based", "can i use it on mobile",
            "how to configure", "first steps with iryax",
            "quick start guide", "how to get started with iryax",
            "onboarding process", "how long does setup take",
            "can i import my existing data"
        ],
        "exact_queries": [
            "how it works", "how does it work", "workflow", "how to use",
            "getting started", "setup", "implementation", "process"
        ],
        "url_source": "https://iryax.com/how-it-works",
        "prompt": "Summarize how the platform works based on the text. Format it cleanly with markdown. Start directly with 'Here is how Iryax Global works:'."
    },
    {
        "name": "security",
        "keywords": [
            "security", "privacy", "data", "protection", "gdpr", "secure",
            "encryption", "confidentiality", "safe", "trust", "compliance",
            "policy", "terms", "safety", "data breach", "hack",
            "cyberattack", "vulnerability", "ssl", "https",
            "two factor", "2fa", "mfa", "password", "access control",
            "role based access", "rbac", "audit log", "audit trail",
            "data storage", "data residency", "cloud security",
            "server security", "backup", "data backup", "disaster recovery",
            "uptime", "sla", "data ownership", "delete data",
            "right to delete", "data portability", "terms of service",
            "terms and conditions", "cookie policy", "data retention",
            "iso certified", "soc2", "soc 2", "information security"
        ],
        "phrases": [
            "is my data secure", "privacy policy", "data protection",
            "security features", "tell me about security",
            "how is data protected", "gdpr compliance", "data privacy",
            "secure platform", "trustworthy", "security measures",
            "encryption methods", "data safety", "privacy policy details",
            "where is my data stored", "is my information safe",
            "how do you protect data", "can my data be hacked",
            "is there data encryption", "two factor authentication",
            "how safe is iryax", "what happens to my data",
            "who has access to my data", "do you share my data",
            "can i delete my data", "gdpr compliant",
            "is iryax iso certified", "security audit",
            "how secure is the platform", "terms and conditions details",
            "cookie policy details", "how long is data stored",
            "where are your servers"
        ],
        "exact_queries": [
            "security", "privacy", "data privacy", "gdpr", "is it secure",
            "data protection", "privacy policy", "terms and conditions",
            "is my data safe", "safe"
        ],
        "url_source": "https://iryax.com/privacy-policy",
        "prompt": "Summarize the Privacy Policy and Security measures. Mention what information is collected, how it is protected, and that payments are secure. Format it cleanly with markdown."
    },
    {
        "name": "contact",
        "keywords": [
            "contact", "email", "phone", "address", "office", "location",
            "reach", "support", "help", "customer service", "headquarters",
            "offices", "map", "directions", "whatsapp", "chat",
            "call us", "message us", "write to us", "get support",
            "technical support", "helpdesk", "ticket", "raise a ticket",
            "complaint", "feedback", "query", "enquiry", "inquiry",
            "contact number", "toll free", "landline", "mobile number",
            "customer care number", "office hours", "business hours",
            "when are you open", "available hours", "response time",
            "how long to reply", "contact form", "contact page",
            "india office", "hyderabad office", "iryax office",
            "nearest office", "visit office", "social media"
        ],
        "phrases": [
            "how to contact", "where is your office", "phone number",
            "email address", "contact info", "contact details",
            "reach out", "get in touch", "support email", "helpdesk",
            "customer care", "office location", "headquarters address",
            "support phone", "whatsapp number", "how to reach iryax",
            "i need help", "who do i contact", "contact for support",
            "how do i raise a complaint", "raise a ticket",
            "how to contact customer support", "speak to someone",
            "talk to a human", "talk to an agent", "live chat",
            "can i call you", "can i whatsapp you",
            "what is your phone number", "what is your email",
            "where are you located", "iryax office address",
            "office hours of iryax", "how quickly do you respond",
            "response time for support", "contact sales team"
        ],
        "exact_queries": [
            "contact", "email", "phone", "address", "office location",
            "support", "whatsapp", "contact details", "how to contact",
            "customer care", "helpdesk"
        ],
        "url_source": "https://iryax.com/contact",
        "prompt": "Extract the contact information (Email, Phone/WhatsApp, and Office Address). Format it cleanly as a markdown bulleted list. Do not add conversational filler. Start directly with 'Here is the contact information for Iryax Global:'."
    },
    {
        "name": "careers",
        "keywords": [
            "job", "jobs", "career", "careers", "opening", "openings",
            "vacancy", "vacancies", "hiring", "recruit", "position",
            "positions", "internship", "internships", "employment", "work",
            "opportunity", "opportunities", "role", "roles", "staff",
            "employee", "talent", "recruitment", "apply", "application",
            "fresher", "experienced", "full time", "part time",
            "remote job", "work from home job", "wfh job",
            "software engineer", "developer", "designer", "marketing",
            "sales", "hr jobs", "finance jobs", "operations jobs",
            "it jobs", "tech jobs", "tech career", "it career",
            "join us", "join iryax", "work at iryax", "iryax jobs",
            "placement", "campus placement", "off campus",
            "lateral hire", "referral", "job referral"
        ],
        "phrases": [
            "job openings", "career opportunities", "work with us",
            "hiring", "tell me about careers", "open positions",
            "current vacancies", "how to apply", "join our team",
            "career page", "employment opportunities", "vacant positions",
            "job vacancies", "internship opportunities", "jobs at iryax",
            "apply for job", "current openings", "are you hiring",
            "do you have any openings", "any job vacancies",
            "i want to work at iryax", "how to apply for jobs",
            "what positions are open", "how to send my resume",
            "software developer job", "internship at iryax",
            "fresher jobs", "experience required for jobs",
            "how to join iryax", "career growth at iryax",
            "work culture at iryax", "employee benefits",
            "does iryax hire remotely", "remote job opportunities",
            "campus recruitment", "can i send my cv"
        ],
        "exact_queries": [
            "jobs", "careers", "vacancies", "hiring", "current openings",
            "job openings", "internship", "apply for job", "career"
        ],
        "url_source": "https://iryax.com/careers",
        "prompt": "Summarize the active job openings across different departments. Format it cleanly with markdown headers. Do not add conversational filler. Start directly with 'Here is information about active careers and job openings at Iryax Global:'."
    },
    {
        "name": "about",
        "keywords": [
            "about", "company", "overview", "history", "mission", "vision",
            "values", "story", "founder", "team", "culture", "background",
            "info", "introduction", "who is iryax", "what is iryax",
            "iryax global", "ingrainsystem", "ingrain system",
            "parent company", "subsidiary", "headquarters", "founded",
            "established", "since when", "how old", "years of experience",
            "company size", "employees count", "team size", "iryax team",
            "leadership", "management", "ceo", "directors",
            "what iryax does", "iryax services", "iryax solutions",
            "iryax technology", "why iryax", "iryax difference",
            "what makes iryax different", "unique about iryax",
            "iryax advantages", "iryax strengths", "company profile"
        ],
        "phrases": [
            "about iryax", "who is iryax", "what is iryax",
            "tell me about iryax", "company overview",
            "what does iryax do", "about your company", "about company",
            "iryax overview", "tell me about company", "explain iryax",
            "what is your company", "background of iryax",
            "company history", "mission and vision",
            "what does your company do", "who founded iryax",
            "when was iryax founded", "where is iryax from",
            "what is the story of iryax", "why was iryax created",
            "what problem does iryax solve", "iryax company details",
            "what industry is iryax in", "iryax headquarters",
            "is iryax an indian company", "tell me about ingrain system",
            "what is ingrainsystem", "iryax company profile"
        ],
        "exact_queries": [
            "about", "overview", "company", "what is iryax", "who are you",
            "about iryax", "iryax global", "company overview", "about company"
        ],
        "url_source": "https://iryax.com/about",
        "prompt": "Summarize what Iryax Global is and what problems it solves based on the text. Format it cleanly with markdown. Do not add conversational filler. Start directly with 'Iryax Global is an enterprise technology company'."
    },
    {
        "name": "customers",
        "keywords": [
            "customer", "customers", "client", "clients", "user", "users",
            "case study", "case studies", "testimonials", "feedback",
            "reviews", "portfolio", "trusted by", "partners",
            "who uses it", "success stories", "reference", "references",
            "existing customers", "current customers", "happy customers",
            "satisfied clients", "industries served", "verticals",
            "which companies use iryax", "company references",
            "client portfolio", "customer base", "user base",
            "ratings", "google review", "trustpilot", "g2",
            "star rating", "how many customers", "how many users",
            "customer count", "client count"
        ],
        "phrases": [
            "who are your customers", "who uses iryax", "client list",
            "our clients", "our customers", "customer reviews",
            "client testimonials", "who do you work with",
            "customer success stories", "show me case studies",
            "any references", "do you have existing customers",
            "who are your clients", "which companies trust you",
            "can you show reviews", "user feedback",
            "what do your customers say", "success stories",
            "any client testimonials", "trusted by whom",
            "how many companies use iryax", "client base",
            "is iryax trusted", "customer satisfaction"
        ],
        "exact_queries": [
            "customers", "clients", "who uses iryax", "testimonials",
            "case studies", "reviews", "references"
        ],
        "static_answer": "Iryax Global serves a diverse group of forward-thinking enterprises, fast-growing startups, and organizations across multiple industries. For specific client references, case studies, or to see how we can help your business, please contact our sales team."
    },
    {
        "name": "greeting",
        "keywords": [
            "hi", "hello", "hlo", "hey", "greetings", "morning", "evening",
            "sup", "yo", "howdy", "hiya", "hey there", "good day", "afternoon",
            "hii", "hiii", "hiiii", "helloo", "hellooo", "hai",
            "namaste", "namasthe", "vanakkam", "salam", "assalam",
            "good night", "gn", "gm", "good morning", "good evening",
            "guten tag", "bonjour", "ola", "hola", "ciao", "wassup",
            "whatsup", "what is up", "how are you", "you there", "anyone there"
        ],
        "phrases": [
            "hi there", "hello there", "good morning", "good evening",
            "how are you", "how are you doing", "hey, hows it going",
            "nice to meet you", "greetings", "good afternoon",
            "hello everyone", "hi everyone", "hey there how are you",
            "good to see you", "glad to be here", "hope you are well",
            "happy to chat", "excited to learn more"
        ],
        "exact_queries": [
            "hi", "hello", "hey", "hii", "helloo", "good morning",
            "good evening", "hai", "namaste", "hola"
        ],
        "static_answer": "Hello! I am Iryax Assistant. I can help you find information about our products, pricing, careers, and contact details. How can I help you today?"
    },
    {
        "name": "identity",
        "keywords": [
            "who are you", "what are you", "are you a bot", "ai",
            "assistant", "your name", "chatbot", "automated",
            "virtual assistant", "ai assistant", "robot", "bot",
            "machine", "algorithm", "artificial intelligence",
            "gpt", "llm", "language model", "openai", "gemini",
            "who made you", "who built you", "who created you",
            "who trained you", "what model are you",
            "are you human", "are you real", "real person",
            "talking to a person", "talk to a human",
            "powered by", "built on", "technology behind",
            "iryax bot", "iryax ai", "iryax chatbot"
        ],
        "phrases": [
            "tell me who you are", "what is your name",
            "are you a real person", "are you automated",
            "whats your name", "tell me about yourself",
            "who am i talking to", "is this a bot",
            "are you an ai", "virtual assistant", "who made you",
            "what ai are you", "are you gpt", "which model are you",
            "are you powered by openai", "are you a robot",
            "is this automated", "who created this chatbot",
            "how were you built", "what is your purpose",
            "what can you help me with", "what are your capabilities"
        ],
        "exact_queries": [
            "who are you", "who are you?", "what are you", "what are you?",
            "are you a bot", "are you a bot?", "whats your name",
            "tell me about yourself", "are you human", "are you an ai"
        ],
        "static_answer": "I am Iryax Assistant, the official AI for Iryax Global. I can help you with products, pricing, careers, and contact info."
    },
]


from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_structured_intents(target_urls: set = None):
    """
    Generate or update structured intents.
    If target_urls is specified, ONLY intents linked to those URLs will be regenerated;
    existing intents for unchanged URLs will be preserved from structured_intents.json.
    """
    print(f"[IntentGen] Starting LLM intent generation using {MODEL_NAME}...")
    if not RAW_PAGES_PATH.exists():
        print("[IntentGen] RAW_PAGES_PATH not found. Aborting.")
        return

    with open(RAW_PAGES_PATH, "r", encoding="utf-8") as f:
        pages = json.load(f)

    # Load existing intents map so we can update in-place if target_urls is provided
    existing_map = {}
    if target_urls and INTENTS_PATH.exists():
        try:
            with open(INTENTS_PATH, "r", encoding="utf-8") as f:
                old_data = json.load(f).get("intents", [])
                for item in old_data:
                    existing_map[item["name"]] = item
        except Exception:
            pass

    # Filter configs to process
    configs_to_process = []
    for config in INTENTS_CONFIG:
        url = config.get("url_source")
        if target_urls and url and url not in target_urls and config["name"] in existing_map:
            # Skip regenerating unchanged intents — preserve existing answer
            continue
        configs_to_process.append(config)

    print(f"[IntentGen] Processing {len(configs_to_process)} intent(s) (target_urls={target_urls or 'ALL'})...")

    results_map = dict(existing_map)

    def process_config(config):
        if "static_answer" in config:
            return config["name"], {
                "name": config["name"],
                "keywords": config["keywords"],
                "phrases": config["phrases"],
                "exact_queries": config["exact_queries"],
                "answer": config["static_answer"],
                "sources": []
            }
            
        url = config["url_source"]
        if url not in pages:
            print(f"[IntentGen] Skipping {config['name']} - URL {url} not scraped.")
            return config["name"], None
            
        markdown = pages[url].get("markdown", "")
        clean_md = re.sub(r'!\[.*?\]\(.*?\)', '', markdown)[:3000]
        
        print(f"[IntentGen] Generating intent: {config['name']}")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant for Iryax Global. Provide output exactly as requested."},
            {"role": "user", "content": f"{config['prompt']}\n\nContext text:\n{clean_md}"}
        ]
        
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                options={"temperature": 0.1, "num_predict": 500}
            )
            answer = response["message"]["content"].strip()
            
            if "For additional information" not in answer:
                answer += f"\n\nFor additional information, please contact us at {CONTACT_EMAIL} or call {CONTACT_PHONE}."
                
            return config["name"], {
                "name": config["name"],
                "keywords": config["keywords"],
                "phrases": config["phrases"],
                "exact_queries": config["exact_queries"],
                "answer": answer,
                "sources": [url]
            }
        except Exception as e:
            print(f"[IntentGen] Failed to generate {config['name']}: {e}")
            return config["name"], existing_map.get(config["name"])

    # Run in parallel with max 3 worker threads
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_config, cfg) for cfg in configs_to_process]
        for future in as_completed(futures):
            name, item = future.result()
            if item:
                results_map[name] = item

    # Preserve order matching INTENTS_CONFIG
    final_intents = []
    for config in INTENTS_CONFIG:
        name = config["name"]
        if name in results_map:
            final_intents.append(results_map[name])

    if final_intents:
        INTENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INTENTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"intents": final_intents}, f, indent=2, ensure_ascii=False)
        print(f"[IntentGen] Successfully saved {len(final_intents)} intents to structured_intents.json.")

if __name__ == "__main__":
    generate_structured_intents()
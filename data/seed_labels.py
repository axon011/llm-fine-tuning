"""One-time script to create label JSON files for existing JDs."""
import json
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"

labels = {
    "anthropic-fde": {
        "title": "Forward Deployed Engineer, Applied AI",
        "company": "Anthropic",
        "location": "Munich, Germany",
        "work_model": "hybrid",
        "seniority": "senior",
        "required_skills": ["Python", "TypeScript", "LLM deployment", "prompt engineering", "agent development", "evaluation frameworks"],
        "nice_to_have": ["financial services", "healthcare", "enterprise IT systems"],
        "salary": "205,000-220,000 EUR",
        "language": "German (C1) + English"
    },
    "appliedai-genai": {
        "title": "Generative AI Engineer",
        "company": "appliedAI Initiative GmbH",
        "location": "Munich / Heilbronn, Germany",
        "work_model": "hybrid",
        "seniority": "mid-level",
        "required_skills": ["Python", "LangChain", "LangGraph", "AutoGen", "RAG", "multi-agent systems", "Transformer architectures", "prompt engineering", "Docker", "CI/CD", "Azure/AWS/GCP"],
        "nice_to_have": ["Vision Language Models", "small language models"],
        "salary": "",
        "language": "German + English"
    },
    "fastino-ai": {
        "title": "ML Engineer - Small Language Models",
        "company": "fastino.ai",
        "location": "Palo Alto, California",
        "work_model": "hybrid/remote",
        "seniority": "mid-level",
        "required_skills": ["Python", "LLMs", "AI/ML engineering", "vector databases", "Kubernetes", "prompt engineering"],
        "nice_to_have": [],
        "salary": "",
        "language": "English"
    },
    "jamie": {
        "title": "AI Engineer",
        "company": "Jamie",
        "location": "Berlin, Germany",
        "work_model": "hybrid/remote",
        "seniority": "mid-level",
        "required_skills": ["ML/AI pipelines", "MLOps", "prompt engineering", "software development"],
        "nice_to_have": ["side projects"],
        "salary": "top 5% salary band in Europe",
        "language": "English"
    },
    "jetbrains-core-engine": {
        "title": "AI Engineer (Core Engine)",
        "company": "JetBrains",
        "location": "Berlin / Munich, Germany",
        "work_model": "distributed",
        "seniority": "senior",
        "required_skills": ["Python", "Kotlin", "NLP", "information retrieval", "LLM prompt engineering", "AI agents", "semantic search"],
        "nice_to_have": ["Java", "code intelligence", "AST", "symbol graphs"],
        "salary": "",
        "language": "English"
    },
    "logicc": {
        "title": "AI Engineer",
        "company": "Logicc",
        "location": "Hamburg, Germany",
        "work_model": "remote/hybrid/onsite",
        "seniority": "mid-level",
        "required_skills": ["Python", "LLM integration", "RAG", "agentic systems", "model evaluation", "coding agents"],
        "nice_to_have": ["C#", "React", "Next.js", "Tailwind"],
        "salary": "55,000-70,000 EUR",
        "language": "German"
    },
    "mistral-fdml": {
        "title": "Forward Deployed Machine Learning Engineer",
        "company": "Mistral AI",
        "location": "Paris / London / Munich / Amsterdam",
        "work_model": "on-site",
        "seniority": "mid-level",
        "required_skills": ["Python", "AI/ML products", "fine-tuning LLMs", "RAG", "agentic systems", "APIs"],
        "nice_to_have": ["open-source contributions", "customer engineering", "PyTorch"],
        "salary": "",
        "language": "English"
    },
    "mistral-fullstack": {
        "title": "Applied AI Engineer, Fullstack Software Engineer",
        "company": "Mistral AI",
        "location": "Paris / Marseille / London / Amsterdam",
        "work_model": "hybrid",
        "seniority": "mid-level",
        "required_skills": ["Python", "TypeScript", "React", "Next.js", "Node.js", "full-stack development"],
        "nice_to_have": ["LLM/GenAI", "open-source contributions", "customer engineering"],
        "salary": "",
        "language": "English"
    },
    "ml-reply-agentic": {
        "title": "Agentic AI Software Engineer",
        "company": "Machine Learning Reply",
        "location": "Munich, Germany",
        "work_model": "hybrid",
        "seniority": "entry-level",
        "required_skills": ["Python", "Java", "agentic AI", "multi-step workflows", "cloud platforms", "FastAPI", "SQL/NoSQL"],
        "nice_to_have": ["LLM frameworks", "Docker", "Kubernetes", "client-facing skills"],
        "salary": "",
        "language": "English"
    },
    "percepta": {
        "title": "Applied AI Engineer",
        "company": "Percepta",
        "location": "London / European Union",
        "work_model": "on-site",
        "seniority": "senior",
        "required_skills": ["AI deployment", "stakeholder collaboration", "production AI systems"],
        "nice_to_have": ["LangGraph", "RAG pipelines", "multi-step agents", "Python", "TypeScript", "AWS/GCP/Azure"],
        "salary": "185,000-230,000 GBP",
        "language": "English"
    },
    "pinnipedia": {
        "title": "AI Engineer (LLMs + Knowledge Graphs)",
        "company": "Pinnipedia Technologies GmbH",
        "location": "Berlin / Ketzin, Germany",
        "work_model": "hybrid",
        "seniority": "mid-level",
        "required_skills": ["Python", "knowledge graphs", "ontology design", "SPARQL/Cypher", "RAG", "evaluation metrics"],
        "nice_to_have": ["ML/LLM observability", "AWS/Azure", "Docker", "CI/CD"],
        "salary": "60,000-85,000 EUR",
        "language": "English"
    },
    "quora-poe": {
        "title": "AI Engineer New Grad 2025-2026",
        "company": "Quora / Poe",
        "location": "Remote",
        "work_model": "remote",
        "seniority": "entry-level",
        "required_skills": ["Python", "TypeScript", "LLM applications", "transformer models", "machine learning"],
        "nice_to_have": ["RAG", "prompt engineering", "agentic workflows"],
        "salary": "107,360-152,900 USD",
        "language": "English"
    },
    "rhesis-ai": {
        "title": "AI Engineer for OSS AI Startup",
        "company": "Rhesis AI GmbH",
        "location": "Berlin / Potsdam, Germany",
        "work_model": "hybrid",
        "seniority": "mid-level",
        "required_skills": ["Python", "LLMs", "PyTorch", "JAX", "TensorFlow", "HuggingFace", "MLOps", "NLP"],
        "nice_to_have": ["deep learning fine-tuning", "cloud-native (GCP/Azure/AWS)", "system design"],
        "salary": "",
        "language": "English"
    },
    "sokra": {
        "title": "Founding Engineer (Full-Stack AI/ML)",
        "company": "Sokra",
        "location": "Berlin, Germany",
        "work_model": "hybrid",
        "seniority": "entry-level",
        "required_skills": ["TypeScript", "React", "Next.js", "Python", "LLM agents", "RAG pipelines"],
        "nice_to_have": [],
        "salary": "0.5-2% equity",
        "language": "English"
    },
    "sunhat": {
        "title": "AI Software Engineer",
        "company": "Sunhat GmbH",
        "location": "Cologne / Remote, Germany",
        "work_model": "remote",
        "seniority": "mid-level",
        "required_skills": ["Python", "LLMs", "RAG", "agentic systems", "FastAPI", "Docker"],
        "nice_to_have": ["Angular", "Nest.js", "Vertex AI"],
        "salary": "65,000-95,000 EUR",
        "language": "English"
    },
    "top-legal": {
        "title": "Junior AI Engineer",
        "company": "top.legal",
        "location": "Germany",
        "work_model": "remote/hybrid",
        "seniority": "junior",
        "required_skills": ["Python", "PyTorch", "LangChain", "NLP", "LLM fine-tuning", "RAG", "PEFT/QLoRA"],
        "nice_to_have": ["JavaScript"],
        "salary": "",
        "language": "English"
    },
    "almedia": {
        "title": "AI Full Stack Engineer",
        "company": "Almedia",
        "location": "Berlin, Germany",
        "work_model": "hybrid",
        "seniority": "mid-level",
        "required_skills": ["Python", "Node.js", "React", "LLM integration", "RAG", "Docker"],
        "nice_to_have": ["TypeScript", "AWS"],
        "salary": "",
        "language": "English"
    },
    "bending-spoons": {
        "title": "Graduate AI Software Engineer",
        "company": "Bending Spoons",
        "location": "Berlin / Remote",
        "work_model": "remote",
        "seniority": "entry-level",
        "required_skills": ["Python", "software engineering", "problem-solving", "AI/ML fundamentals"],
        "nice_to_have": ["Rust", "mobile development"],
        "salary": "",
        "language": "English"
    },
    "code-compass": {
        "title": "AI Software Engineer",
        "company": "Code Compass",
        "location": "Hamburg / Remote, Germany",
        "work_model": "remote",
        "seniority": "mid-level",
        "required_skills": ["Python", "PyTorch", "LLMs", "Docker"],
        "nice_to_have": [],
        "salary": "55,000-85,000 EUR",
        "language": "English"
    },
    "dhl": {
        "title": "Associate Data Scientist AI",
        "company": "DHL",
        "location": "Bonn, Germany",
        "work_model": "hybrid",
        "seniority": "entry-level",
        "required_skills": ["Python", "SQL", "RAG", "agent orchestration", "ML fundamentals"],
        "nice_to_have": ["Java", "AWS"],
        "salary": "",
        "language": "English"
    },
    "ellamind": {
        "title": "AI Engineer",
        "company": "ellamind",
        "location": "Berlin / Bremen, Germany",
        "work_model": "hybrid",
        "seniority": "entry-level",
        "required_skills": ["Python", "LLM evaluation", "RAG", "prompt engineering"],
        "nice_to_have": ["TypeScript", "React"],
        "salary": "60,000-100,000 EUR",
        "language": "English"
    },
    "flank": {
        "title": "AI Engineer",
        "company": "Flank",
        "location": "Berlin, Germany",
        "work_model": "on-site",
        "seniority": "mid-level",
        "required_skills": ["Python", "LLM agents", "RAG", "document processing", "evaluation frameworks"],
        "nice_to_have": ["legal domain", "contract automation"],
        "salary": "",
        "language": "English"
    },
    "integritynext": {
        "title": "AI Engineer",
        "company": "IntegrityNext",
        "location": "Munich / Remote DE",
        "work_model": "remote",
        "seniority": "mid-level",
        "required_skills": ["Python", "LangChain", "LangGraph", "AWS Bedrock", "RAG", "agentic systems", "evaluation frameworks"],
        "nice_to_have": ["React", "Java", "AI observability"],
        "salary": "",
        "language": "English"
    },
    "ionos": {
        "title": "AI Developer with Python",
        "company": "IONOS",
        "location": "Karlsruhe / Berlin, Germany",
        "work_model": "hybrid",
        "seniority": "entry-level",
        "required_skills": ["Python", "FastAPI", "LangChain", "LangGraph", "RAG", "vector databases", "Docker"],
        "nice_to_have": ["Speech-to-Speech", "OAuth2", "Java"],
        "salary": "",
        "language": "English"
    },
    "kapa-ai": {
        "title": "Research Engineer Applied AI",
        "company": "kapa.ai",
        "location": "Remote EU",
        "work_model": "remote",
        "seniority": "mid-level",
        "required_skills": ["Python", "RAG", "LLM evaluation", "information retrieval", "embeddings"],
        "nice_to_have": ["fine-tuning", "TypeScript"],
        "salary": "100,000-170,000 USD",
        "language": "English"
    },
    "knowunity": {
        "title": "AI Engineer",
        "company": "Knowunity",
        "location": "Berlin, Germany",
        "work_model": "hybrid",
        "seniority": "mid-level",
        "required_skills": ["Python", "LLMs", "fine-tuning", "RAG", "evaluation"],
        "nice_to_have": ["EdTech domain"],
        "salary": "",
        "language": "English"
    },
    "lovehoney": {
        "title": "Junior AI Engineer",
        "company": "Lovehoney Group",
        "location": "Berlin, Germany",
        "work_model": "hybrid",
        "seniority": "junior",
        "required_skills": ["Python", "LLMs", "RAG", "LangChain", "LlamaIndex", "SQL", "Docker"],
        "nice_to_have": ["n8n", "BigQuery", "Snowflake"],
        "salary": "",
        "language": "English"
    },
    "maia": {
        "title": "Applied AI Engineer",
        "company": "MAIA",
        "location": "Leipzig / Remote DE",
        "work_model": "remote",
        "seniority": "mid-level",
        "required_skills": ["TypeScript", "Go", "RAG", "agentic workflows", "LLM integrations", "PostgreSQL", "vector databases"],
        "nice_to_have": ["Supabase", "document analysis", "LLM tracing"],
        "salary": "75,000-85,000 EUR",
        "language": "English"
    },
    "mercura": {
        "title": "AI/LLM Engineer",
        "company": "Mercura",
        "location": "Munich, Germany",
        "work_model": "on-site",
        "seniority": "mid-level",
        "required_skills": ["Python", "LLM-powered agentic systems", "RAG", "hybrid search", "evaluation systems", "data pipelines"],
        "nice_to_have": ["React", "TypeScript", "FastAPI"],
        "salary": "",
        "language": "English"
    },
    "netconomy": {
        "title": "AI Engineer",
        "company": "NETCONOMY",
        "location": "Berlin / Dortmund, Germany",
        "work_model": "hybrid",
        "seniority": "mid-level",
        "required_skills": ["Python", "LLMs", "RAG", "Azure", "Docker"],
        "nice_to_have": ["AWS", "GCP"],
        "salary": "50,000-90,000 EUR",
        "language": "English"
    },
    "ommax": {
        "title": "AI/LLM Application Engineer",
        "company": "OMMAX",
        "location": "Munich, Germany",
        "work_model": "hybrid",
        "seniority": "entry-level",
        "required_skills": ["Python", "LangChain", "LangGraph", "RAG", "React", "FastAPI"],
        "nice_to_have": ["Docker", "Kubernetes"],
        "salary": "",
        "language": "English"
    },
    "omnora": {
        "title": "Founding AI Engineer",
        "company": "Omnora",
        "location": "Frankfurt, Germany",
        "work_model": "hybrid",
        "seniority": "mid-level",
        "required_skills": ["Python", "LangGraph", "LangChain", "knowledge graphs", "RAG"],
        "nice_to_have": ["Neo4j", "LiveKit"],
        "salary": "",
        "language": "English"
    },
    "paiqo": {
        "title": "AI Engineer II Agentic AI Platform",
        "company": "Paiqo",
        "location": "Paderborn / Remote DE",
        "work_model": "remote",
        "seniority": "mid-level",
        "required_skills": ["Python", "Azure", "agentic AI", "RAG", "Docker"],
        "nice_to_have": ["Kubernetes", "CI/CD"],
        "salary": "",
        "language": "German (B2) + English"
    },
    "punku-ai": {
        "title": "AI SWE",
        "company": "PUNKU.AI",
        "location": "Heilbronn, Germany",
        "work_model": "on-site",
        "seniority": "entry-level",
        "required_skills": ["Python", "AI agents", "LLMs"],
        "nice_to_have": [],
        "salary": "",
        "language": "English"
    },
    "reflow": {
        "title": "AI Engineer",
        "company": "Reflow",
        "location": "Berlin / Remote",
        "work_model": "remote",
        "seniority": "entry-level",
        "required_skills": ["Python", "agentic AI", "LLMs", "software engineering"],
        "nice_to_have": [],
        "salary": "",
        "language": "English"
    },
    "semorai": {
        "title": "AI Engineer",
        "company": "Semorai GmbH",
        "location": "Heilbronn / Munich, Germany",
        "work_model": "hybrid",
        "seniority": "entry-level",
        "required_skills": ["Python", "LLMs", "multi-agent architectures", "Neo4j", "Qdrant", "gRPC", "Docker"],
        "nice_to_have": ["microservices"],
        "salary": "",
        "language": ""
    },
    "smartbroker": {
        "title": "AI Engineer",
        "company": "Smartbroker AG",
        "location": "Berlin, Germany",
        "work_model": "hybrid",
        "seniority": "mid-level",
        "required_skills": ["Python", "LLMs", "RAG", "agentic systems"],
        "nice_to_have": ["fintech domain"],
        "salary": "",
        "language": "English"
    },
    "tpe": {
        "title": "AI Engineer",
        "company": "TPE Consulting",
        "location": "Germany (Berlin/Munich/Frankfurt/Hamburg/Remote)",
        "work_model": "remote/hybrid",
        "seniority": "entry-level",
        "required_skills": ["Python", "TensorFlow", "PyTorch", "LLMs", "GenAI", "AWS/GCP/Azure", "MLOps", "Docker", "Kubernetes"],
        "nice_to_have": [],
        "salary": "90,000-150,000 EUR",
        "language": "English"
    }
}

for name, label in labels.items():
    path = RAW_DIR / f"{name}.json"
    path.write_text(json.dumps(label, indent=2), encoding="utf-8")
    print(f"  {name}.json")

print(f"\nCreated {len(labels)} label files in {RAW_DIR}/")

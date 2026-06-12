import fitz
from sentence_transformers import SentenceTransformer, util

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ── RAG Knowledge Base ────────────────────────────────────────────────────────
RAG_KNOWLEDGE_BASE = [
    {
        "role": "ML Engineer",
        "desc": "python machine learning tensorflow deep learning scikit-learn model training deployment neural networks pytorch feature engineering statistics mlops",
        "companies": ["Google", "Amazon", "Microsoft", "Razorpay", "CRED", "Flipkart", "NVIDIA"],
        "salary": "₹8L – ₹40L"
    },
    {
        "role": "Data Scientist",
        "desc": "python machine learning statistics research hypothesis testing ab testing visualization feature engineering clustering pandas numpy jupyter",
        "companies": ["Google", "Amazon", "Flipkart", "Uber", "LinkedIn", "Walmart Labs", "Meesho"],
        "salary": "₹7L – ₹32L"
    },
    {
        "role": "Data Analyst",
        "desc": "python sql pandas data analysis statistics visualization matplotlib seaborn excel reporting dashboard power bi tableau looker",
        "companies": ["Flipkart", "Swiggy", "Zomato", "Infosys", "TCS", "Deloitte", "Accenture"],
        "salary": "₹4L – ₹16L"
    },
    {
        "role": "NLP Engineer",
        "desc": "nlp bert transformers text classification sentiment analysis named entity recognition python pytorch huggingface spacy gpt language models",
        "companies": ["Google", "Microsoft", "Amazon", "Sarvam AI", "Krutrim", "AI4Bharat", "OpenAI"],
        "salary": "₹10L – ₹45L"
    },
    {
        "role": "Data Engineer",
        "desc": "python sql etl pipeline apache spark hadoop data warehouse snowflake airflow kafka databricks dbt data lake",
        "companies": ["Airbnb", "Uber", "Swiggy", "Freshworks", "Dunzo", "Razorpay", "PhonePe"],
        "salary": "₹8L – ₹30L"
    },
    {
        "role": "Backend Developer",
        "desc": "python django flask rest api postgresql redis docker microservices java spring golang node.js system design",
        "companies": ["Razorpay", "PhonePe", "CRED", "Zepto", "Meesho", "Groww", "Paytm"],
        "salary": "₹6L – ₹28L"
    },
    {
        "role": "Cloud Engineer",
        "desc": "aws azure gcp docker kubernetes devops ci cd linux serverless pipeline terraform ansible infrastructure",
        "companies": ["Amazon", "Microsoft", "Google", "IBM", "Rackspace", "Infosys", "Wipro"],
        "salary": "₹7L – ₹28L"
    },
    {
        "role": "Full Stack Developer",
        "desc": "react nodejs express mongodb html css javascript python django postgresql rest api graphql next.js",
        "companies": ["Wipro", "Cognizant", "Accenture", "TCS", "Startups", "Capgemini", "HCL"],
        "salary": "₹5L – ₹22L"
    },
    {
        "role": "DevOps Engineer",
        "desc": "docker kubernetes jenkins ci cd git linux bash ansible terraform aws azure monitoring prometheus grafana",
        "companies": ["Amazon", "Microsoft", "Google", "Infosys", "TCS", "Wipro", "HCL"],
        "salary": "₹6L – ₹25L"
    },
    {
        "role": "Business Analyst",
        "desc": "sql excel power bi tableau business intelligence reporting dashboard kpi stakeholder communication requirements agile",
        "companies": ["KPMG", "Deloitte", "Capgemini", "Accenture", "IBM", "EY", "PwC"],
        "salary": "₹5L – ₹20L"
    },
    {
        "role": "Computer Vision Engineer",
        "desc": "opencv python deep learning image processing cnn yolo object detection segmentation pytorch tensorflow cuda",
        "companies": ["Google", "Microsoft", "NVIDIA", "Ola", "Nuro", "Mobileye", "Samsung"],
        "salary": "₹10L – ₹40L"
    },
    {
        "role": "Software Engineer",
        "desc": "java python c++ algorithms data structures problem solving system design object oriented programming git agile",
        "companies": ["TCS", "Infosys", "Wipro", "HCL", "Tech Mahindra", "Capgemini", "Cognizant"],
        "salary": "₹3.5L – ₹15L"
    },
]

SKILL_KEYWORDS = [
    "python", "java", "sql", "machine learning", "deep learning", "tensorflow",
    "pytorch", "pandas", "numpy", "scikit-learn", "tableau", "power bi",
    "excel", "statistics", "data visualization", "nlp", "docker", "aws",
    "git", "flask", "django", "react", "javascript", "spark", "hadoop",
    "kubernetes", "mongodb", "postgresql", "redis", "airflow", "kafka",
    "c++", "golang", "node.js", "next.js", "typescript", "azure", "gcp",
    "linux", "terraform", "jenkins", "ci/cd", "opencv", "huggingface"
]


def extract_text_from_pdf(uploaded_file) -> str:
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception:
        return ""


def rag_retrieve(resume_text: str, top_k: int = 4) -> list:
    """RAG: encode resume → cosine sim → top matching job descriptions."""
    embedder = get_embedder()
    resume_emb = embedder.encode(resume_text[:1200].lower(), convert_to_tensor=True)
    jd_texts = [r["desc"] for r in RAG_KNOWLEDGE_BASE]
    jd_embs = embedder.encode(jd_texts, convert_to_tensor=True)
    sims = util.cos_sim(resume_emb, jd_embs)[0]
    top_idx = sims.argsort(descending=True)[:top_k]

    results = []
    for i in top_idx:
        item = RAG_KNOWLEDGE_BASE[i]
        raw_score = float(sims[i])
        score = min(int(raw_score * 100) + 18, 97)
        results.append({
            "role": item["role"],
            "desc": item["desc"],
            "score": score,
            "companies": item["companies"],
            "salary": item["salary"],
        })
    return results


def extract_skills(text: str):
    text_lower = text.lower()
    found = [s for s in SKILL_KEYWORDS if s in text_lower]
    missing = [s for s in SKILL_KEYWORDS if s not in text_lower]
    return found[:12], missing[:8]


def compute_resume_scores(resume_text: str, found_skills: list) -> dict:
    text_lower = resume_text.lower()
    sections = ["education", "experience", "project", "skill", "certif", "achievement", "summary", "objective"]
    section_count = sum(1 for s in sections if s in text_lower)
    ats = min(len(found_skills) * 5 + 25, 95)
    completeness = min(section_count * 13 + 12, 95)
    keyword_match = min(len(found_skills) * 4 + 18, 92)
    return {"ats": ats, "completeness": completeness, "keyword_match": keyword_match}

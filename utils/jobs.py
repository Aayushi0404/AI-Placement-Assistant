import pandas as pd
from sentence_transformers import SentenceTransformer, util

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


COMPANIES = {
    "ML Engineer":           ["Google", "Amazon", "Microsoft", "Razorpay", "CRED", "NVIDIA", "Flipkart"],
    "Data Scientist":        ["Google", "Amazon", "Flipkart", "Uber", "LinkedIn", "Walmart Labs", "Meesho"],
    "Data Analyst":          ["Flipkart", "Swiggy", "Zomato", "Infosys", "TCS", "Deloitte", "Accenture"],
    "Web Developer":         ["Wipro", "Cognizant", "Accenture", "TCS", "Startups", "Capgemini", "HCL"],
    "Backend Developer":     ["Razorpay", "PhonePe", "CRED", "Zepto", "Meesho", "Groww", "Paytm"],
    "Cloud Engineer":        ["Amazon", "Microsoft", "Google", "IBM", "Rackspace", "Infosys", "Wipro"],
    "Database Administrator":["Oracle", "IBM", "TCS", "Infosys", "Wipro", "Capgemini", "Cognizant"],
    "NLP Engineer":          ["Google", "Microsoft", "Amazon", "Sarvam AI", "Krutrim", "AI4Bharat", "OpenAI"],
    "Data Engineer":         ["Airbnb", "Uber", "Swiggy", "Freshworks", "Razorpay", "PhonePe", "Dunzo"],
    "Business Analyst":      ["KPMG", "Deloitte", "Capgemini", "Accenture", "IBM", "EY", "PwC"],
    "Software Engineer":     ["TCS", "Infosys", "Wipro", "HCL", "Tech Mahindra", "Capgemini", "Cognizant"],
    "Mobile Developer":      ["Paytm", "PhonePe", "CRED", "Meesho", "ShareChat", "Navi", "Ola"],
    "DevOps Engineer":       ["Amazon", "Microsoft", "Google", "Infosys", "TCS", "Wipro", "HCL"],
    "Full Stack Developer":  ["Wipro", "Cognizant", "Accenture", "Startups", "Capgemini", "HCL", "Mphasis"],
    "AI Research Engineer":  ["Google", "Microsoft", "Meta", "OpenAI", "DeepMind", "IISc", "IITs"],
    "Cybersecurity Analyst": ["KPMG", "Deloitte", "IBM", "Wipro", "TCS", "Palo Alto Networks", "CrowdStrike"],
    "Product Manager":       ["Amazon", "Flipkart", "Swiggy", "Razorpay", "CRED", "PhonePe", "Meesho"],
    "QA Engineer":           ["TCS", "Infosys", "Wipro", "Cognizant", "Capgemini", "HCL", "Mphasis"],
    "Blockchain Developer":  ["Polygon", "CoinDCX", "WazirX", "IBM", "TCS", "Deloitte", "Startups"],
    "Computer Vision Engineer":["Google", "Microsoft", "NVIDIA", "Ola", "Mobileye", "Samsung", "Intel"],
}

SALARY = {
    "ML Engineer":            "₹8L – ₹40L",
    "Data Scientist":         "₹7L – ₹32L",
    "Data Analyst":           "₹4L – ₹16L",
    "NLP Engineer":           "₹10L – ₹45L",
    "Data Engineer":          "₹8L – ₹30L",
    "Backend Developer":      "₹6L – ₹28L",
    "Cloud Engineer":         "₹7L – ₹28L",
    "Full Stack Developer":   "₹5L – ₹22L",
    "DevOps Engineer":        "₹6L – ₹25L",
    "Business Analyst":       "₹5L – ₹20L",
    "Computer Vision Engineer":"₹10L – ₹40L",
    "Software Engineer":      "₹3.5L – ₹15L",
    "Mobile Developer":       "₹5L – ₹22L",
    "AI Research Engineer":   "₹12L – ₹60L",
    "Cybersecurity Analyst":  "₹6L – ₹24L",
    "Product Manager":        "₹10L – ₹50L",
    "QA Engineer":            "₹3.5L – ₹14L",
    "Blockchain Developer":   "₹8L – ₹35L",
    "Database Administrator": "₹4L – ₹16L",
    "Web Developer":          "₹4L – ₹18L",
}


def predict_jobs(user_skills_input: str, top_k: int = 5) -> list:
    """RAG: encode user skills → cosine sim → top matching job roles from CSV."""
    df = pd.read_csv("data/jobs.csv")
    user_skills = user_skills_input.lower().strip()

    embedder = get_embedder()
    user_emb = embedder.encode(user_skills, convert_to_tensor=True)
    job_embs = embedder.encode(df["skills"].tolist(), convert_to_tensor=True)
    sims = util.cos_sim(user_emb, job_embs)[0]

    df = df.copy()
    df["score"] = [int(min(float(s) * 100 + 10, 97)) for s in sims]
    top_results = df.sort_values("score", ascending=False).head(top_k)

    output = []
    for _, row in top_results.iterrows():
        role = row["job_role"]
        output.append({
            "job_role": role,
            "match_score": int(row["score"]),
            "companies": COMPANIES.get(role, ["Various companies"]),
            "salary": SALARY.get(role, "₹4L – ₹20L"),
        })
    return output

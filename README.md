# Smart Placement Assistant
RAG-based AI career guidance · SBERT + Groq LLaMA3 · Streamlit

## Quick Setup (3 steps)

### Step 1 — Install
```bash
pip install -r requirements.txt
```
First time takes 5-10 min (downloads SBERT model ~90MB)

### Step 2 — Add Groq API Key
Edit `.env` file:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```
Get free key at https://console.groq.com (free, fast)

### Step 3 — Run
```bash
streamlit run app.py
```
Opens at http://localhost:8501

---

## Project Structure
```
smart_placement/
├── app.py                    ← Main Streamlit app (all 4 pages)
├── .env                      ← Your Groq API key (keep private)
├── requirements.txt
├── data/
│   ├── questions.json        ← RAG KB: 13 domains × 20 questions = 260 questions
│   └── jobs.csv              ← RAG KB: 20 job roles with skills
└── utils/
    ├── interview.py          ← RAG question retrieval + answer analysis
    ├── resume.py             ← RAG resume matching + skill extraction
    └── jobs.py               ← RAG job prediction
```

## RAG Architecture
```
User Input
    ↓
SBERT (all-MiniLM-L6-v2) → 384-dim vector
    ↓
Cosine Similarity against Knowledge Base vectors
    ↓
Top-K most relevant documents retrieved
    ↓
[User Input + Retrieved Context] → Groq LLaMA3-70B
    ↓
Grounded AI Response
```

## Viva Answer (memorize this)
"Our project implements a complete RAG (Retrieval Augmented Generation) pipeline.
We use SBERT's all-MiniLM-L6-v2 model to encode both user input and our domain
knowledge base into 384-dimensional semantic vectors. Cosine similarity retrieves
the most relevant documents from our knowledge base. These retrieved documents are
passed as context to Groq's LLaMA3-70B model for generation.
This RAG architecture is applied across all three features:
interview practice uses it to retrieve diverse domain questions,
resume analyzer uses it to match resumes with job descriptions,
and the job predictor uses it to find semantically similar job roles."

import json
import random
from sentence_transformers import SentenceTransformer, util

# Load knowledge base
with open("data/questions.json", "r") as f:
    QUESTIONS_DB = json.load(f)

DOMAIN_MAP = {
    "Machine Learning": "machine_learning",
    "Deep Learning": "deep_learning",
    "Data Science": "data_science",
    "DSA": "dsa",
    "Operating Systems": "os",
    "DBMS": "dbms",
    "Computer Networks": "cn",
    "OOPs": "oops",
    "SQL": "sql",
    "Artificial Intelligence": "ai",
    "Python": "python",
    "System Design": "system_design",
    "HR Round": "hr"
}

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_rag_question(domain: str, answered_questions: list) -> str:
    key = DOMAIN_MAP.get(domain, "machine_learning")
    all_questions = QUESTIONS_DB.get(key, [])
    available = [q for q in all_questions if q not in answered_questions]

    if not available:
        return random.choice(all_questions)
    if not answered_questions:
        return random.choice(available)

    embedder = get_embedder()
    context = " ".join(answered_questions[-3:])
    context_emb = embedder.encode(context, convert_to_tensor=True)
    avail_embs = embedder.encode(available, convert_to_tensor=True)
    sims = util.cos_sim(context_emb, avail_embs)[0]
    idx = int(sims.argmin())
    return available[idx]


def get_question_difficulty(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["design", "implement", "difference between", "compare", "explain the", "how does", "write a"]):
        return "Hard"
    elif any(k in q for k in ["what is", "when to", "why", "give an example"]):
        return "Medium"
    return "Easy"


def analyze_answer(text: str) -> dict:
    filler_words = ["umm", "uh", "like", "you know", "basically", "literally",
                    "kind of", "sort of", "actually", "i mean", "so basically"]
    tech_keywords = [
        "algorithm", "model", "training", "validation", "accuracy", "precision",
        "recall", "overfitting", "underfitting", "regularization", "gradient",
        "neural", "clustering", "classification", "regression", "feature",
        "dataset", "epoch", "loss", "embedding", "transformer", "cross validation",
        "normalization", "optimization", "backpropagation", "inference", "parameter",
        "layer", "activation", "function", "complexity", "recursion", "iteration",
        "stack", "queue", "tree", "graph", "hash", "binary", "pointer", "memory",
        "cache", "database", "index", "query", "join", "transaction", "schema",
        "protocol", "network", "packet", "thread", "process", "synchronization",
        "deadlock", "semaphore", "mutex", "kernel", "scheduler", "paging",
        "decorator", "generator", "iterator", "lambda", "closure", "inheritance",
        "polymorphism", "encapsulation", "abstraction", "interface", "vector"
    ]

    text_lower = text.lower()
    words = text.split()
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

    filler_count = sum(text_lower.count(f) for f in filler_words)
    tech_words_found = list({w.strip(".,!?") for w in words if w.lower().strip(".,!?") in tech_keywords})[:8]
    tech_count = len(tech_words_found)
    word_count = len(words)
    wpm = min(word_count * 2, 220)

    # Long pauses heuristic
    long_pauses = text_lower.count("...") + text_lower.count(". . .")
    long_pauses = max(long_pauses, max(0, len(sentences) - 4))
    long_pauses = min(long_pauses, 12)

    # Answer Accuracy (0-100)
    accuracy_base = min(40 + tech_count * 6 + min(word_count // 10, 30), 100)
    filler_penalty = min(filler_count * 3, 20)
    answer_accuracy = max(30, accuracy_base - filler_penalty)

    # Technical Vocabulary (0-100)
    if word_count > 0:
        tech_vocab = min(int((tech_count / max(word_count, 1)) * 400 + tech_count * 5), 100)
        tech_vocab = max(20, tech_vocab)
    else:
        tech_vocab = 20

    # Clarity of Speech (0-100)
    avg_sent_len = word_count / max(len(sentences), 1)
    if 10 <= avg_sent_len <= 20:
        clarity = 80
    elif avg_sent_len < 5:
        clarity = 45
    elif avg_sent_len > 35:
        clarity = 50
    else:
        clarity = 65
    clarity = max(20, min(clarity - filler_count * 2, 100))

    # Confidence (0-100)
    hedges = ["maybe", "perhaps", "i think", "i'm not sure", "i guess", "might be",
              "probably", "could be", "not sure", "i believe", "i'm not certain"]
    hedge_count = sum(text_lower.count(h) for h in hedges)
    confidence = max(20, min(100, 80 - hedge_count * 8 - filler_count * 3 + tech_count * 2))

    return {
        "filler_count": filler_count,
        "word_count": word_count,
        "tech_count": tech_count,
        "wpm": wpm,
        "tech_words_found": tech_words_found,
        "long_pauses": long_pauses,
        "answer_accuracy": answer_accuracy,
        "tech_vocab": tech_vocab,
        "clarity": clarity,
        "confidence": confidence,
    }

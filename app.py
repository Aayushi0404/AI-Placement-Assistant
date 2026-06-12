import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq

from utils.interview import get_rag_question, get_question_difficulty, analyze_answer
from utils.resume import extract_text_from_pdf, extract_skills, rag_retrieve, compute_resume_scores
from utils.jobs import predict_jobs

# ── ENV & GROQ ────────────────────────────────────────────────────────────────
load_dotenv()

@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def call_groq(prompt: str, system: str = "You are a helpful AI assistant.") -> str:
    client = get_groq_client()
    if client is None:
        return "ERROR: GROQ_API_KEY not set in .env file"
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"ERROR calling Groq: {e}"


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Placement Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── LIGHT THEME CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset / base ── */
.stApp { background: #f5f6fa; color: #1a1a2e; }
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

/* ── Top navbar ── */
.spa-nav {
    background: #ffffff;
    border-bottom: 1px solid #e2e4ea;
    padding: 14px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.spa-brand { font-size: 1.1rem; font-weight: 800; color: #1a1a2e; }
.spa-brand span { color: #5b4fcf; }
.spa-sub { font-size: 0.7rem; color: #9098b1; margin-top: 2px; }

/* ── Buttons ── */
.stButton > button {
    background: #5b4fcf !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: #4a3fb5 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(91,79,207,0.3) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #ffffff !important;
    border: 1.5px solid #e2e4ea !important;
    border-radius: 10px !important;
    color: #1a1a2e !important;
    font-size: 0.9rem !important;
}
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1.5px solid #e2e4ea !important;
    border-radius: 10px !important;
    color: #1a1a2e !important;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border: 1.5px solid #e8eaf0;
    border-radius: 16px;
    padding: 28px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    height: 100%;
}
.card:hover { border-color: #5b4fcf; box-shadow: 0 6px 24px rgba(91,79,207,0.12); }

.card-icon { font-size: 2.6rem; margin-bottom: 14px; }
.card-title { font-size: 1.1rem; font-weight: 700; color: #1a1a2e; margin-bottom: 8px; }
.card-desc { font-size: 0.82rem; color: #6b7280; line-height: 1.55; }
.card-tag {
    display: inline-block;
    background: #ede9ff;
    color: #5b4fcf;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 12px;
}

/* ── RAG info banner ── */
.rag-banner {
    background: #ede9ff;
    border-left: 4px solid #5b4fcf;
    border-radius: 0 10px 10px 0;
    padding: 10px 16px;
    font-size: 0.8rem;
    color: #4a3fb5;
    margin: 12px 0 20px;
    line-height: 1.5;
}

/* ── Section heading ── */
.sec-title { font-size: 1.65rem; font-weight: 800; color: #1a1a2e; margin-bottom: 4px; }
.sec-sub { font-size: 0.85rem; color: #9098b1; margin-bottom: 18px; }

/* ── Question card ── */
.q-card {
    background: #ffffff;
    border: 1.5px solid #e8eaf0;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 14px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.q-text { font-size: 1.05rem; font-weight: 600; color: #1a1a2e; line-height: 1.55; margin-top: 10px; }

/* ── Badges ── */
.badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
}
.badge-domain  { background: #e0ecff; color: #2563eb; }
.badge-hard    { background: #ffe4e4; color: #dc2626; }
.badge-medium  { background: #fef3cd; color: #d97706; }
.badge-easy    { background: #dcfce7; color: #16a34a; }
.badge-rag     { background: #ede9ff; color: #5b4fcf; }

/* ── Stat boxes ── */
.stat-row { display: flex; gap: 12px; margin: 16px 0; }
.stat-box {
    flex: 1;
    background: #f8f9fc;
    border: 1.5px solid #e8eaf0;
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
}
.stat-val { font-size: 1.6rem; font-weight: 800; color: #5b4fcf; }
.stat-label { font-size: 0.7rem; color: #9098b1; margin-top: 3px; }

/* ── Feedback blocks ── */
.fb-green  { background:#f0fdf4; border-left:4px solid #16a34a; border-radius:0 10px 10px 0; padding:12px 16px; margin:8px 0; font-size:0.88rem; color:#166534; }
.fb-yellow { background:#fffbeb; border-left:4px solid #d97706; border-radius:0 10px 10px 0; padding:12px 16px; margin:8px 0; font-size:0.88rem; color:#92400e; }
.fb-red    { background:#fef2f2; border-left:4px solid #dc2626; border-radius:0 10px 10px 0; padding:12px 16px; margin:8px 0; font-size:0.88rem; color:#991b1b; }
.fb-blue   { background:#eff6ff; border-left:4px solid #2563eb; border-radius:0 10px 10px 0; padding:12px 16px; margin:8px 0; font-size:0.88rem; color:#1e40af; }
.fb-purple { background:#ede9ff; border-left:4px solid #5b4fcf; border-radius:0 10px 10px 0; padding:12px 16px; margin:8px 0; font-size:0.88rem; color:#3730a3; }

/* ── Skill pills ── */
.pill-found   { display:inline-block; background:#dcfce7; color:#16a34a; border:1px solid #bbf7d0; padding:4px 12px; border-radius:20px; font-size:0.78rem; font-weight:600; margin:3px; }
.pill-missing { display:inline-block; background:#fef2f2; color:#dc2626; border:1px solid #fecaca; padding:4px 12px; border-radius:20px; font-size:0.78rem; font-weight:600; margin:3px; }
.pill-company { display:inline-block; background:#f1f5f9; color:#475569; border:1px solid #e2e8f0; padding:3px 10px; border-radius:20px; font-size:0.72rem; margin:3px 2px 0 0; }

/* ── Job card ── */
.job-card {
    background: #ffffff;
    border: 1.5px solid #e8eaf0;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.job-card.top { border-color: #5b4fcf; background: #faf9ff; }
.job-title { font-size: 1rem; font-weight: 700; color: #1a1a2e; }
.job-score-big { font-size: 1.9rem; font-weight: 900; color: #5b4fcf; line-height: 1; }
.job-score-label { font-size: 0.68rem; color: #9098b1; }
.salary-pill { display:inline-block; background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; margin-top:8px; }

/* ── Score bar ── */
.sbar-wrap { background:#e8eaf0; border-radius:20px; height:7px; margin:6px 0; overflow:hidden; }
.sbar-fill { height:100%; border-radius:20px; background:linear-gradient(90deg,#5b4fcf,#a89cf7); transition:width 0.4s; }

/* ── Progress ── */
.prog-wrap { background:#e8eaf0; border-radius:20px; height:6px; margin:6px 0 18px; overflow:hidden; }
.prog-fill { height:100%; border-radius:20px; background:linear-gradient(90deg,#5b4fcf,#a89cf7); }

/* ── Divider ── */
hr.spa-divider { border:none; border-top:1.5px solid #e8eaf0; margin:22px 0; }

/* ── Score circle ── */
.score-circle-wrap { text-align:center; padding:8px 0; }
.score-num { font-size:2.4rem; font-weight:900; color:#5b4fcf; }
.score-label { font-size:0.72rem; color:#9098b1; }

/* ── Resume score row ── */
.rs-row { display:flex; justify-content:space-between; align-items:center; font-size:0.82rem; color:#374151; margin-bottom:4px; }
.rs-pct { font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Session defaults ──────────────────────────────────────────────────────────
_defaults = {
    "page": "home",
    "current_question": "",
    "answered_questions": [],
    "q_number": 0,
    "selected_domain": "Machine Learning",
    "last_feedback": None,
    "last_stats": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── NAVBAR ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="spa-nav">
  <div>
    <div class="spa-brand">Smart <span>Placement</span> Assistant</div>
    <div class="spa-sub">RAG-powered AI career guidance &nbsp;·&nbsp; SBERT + Groq LLaMA3</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Nav buttons row
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("🏠 Home", key="nav_home"):
        st.session_state.page = "home"; st.rerun()
with c2:
    if st.button("🎤 Interview Practice", key="nav_iv"):
        st.session_state.page = "interview"; st.rerun()
with c3:
    if st.button("📄 Resume Analyzer", key="nav_res"):
        st.session_state.page = "resume"; st.rerun()
with c4:
    if st.button("💼 Job Predictor", key="nav_job"):
        st.session_state.page = "jobs"; st.rerun()

st.markdown('<hr class="spa-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
def show_home():
    st.markdown("""
    <div style="text-align:center; padding:40px 20px 32px;">
        <div style="font-size:2.5rem; font-weight:900; color:#1a1a2e;">
            Your AI-Powered <span style="color:#5b4fcf;">Placement Coach</span>
        </div>
        <div style="font-size:0.95rem; color:#6b7280; margin-top:10px;">
            RAG architecture · SBERT semantic embeddings · Groq LLaMA3 generation
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div class="card-icon">🎤</div>
            <div class="card-title">Interview Practice</div>
            <div class="card-desc">RAG retrieves domain questions from knowledge base. Answer &amp; get AI feedback on accuracy, vocabulary, and confidence score.</div>
            <div class="card-tag">13 Domains · 20 Q each · AI Evaluation</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Start Practice →", key="h_iv"):
            st.session_state.page = "interview"; st.rerun()

    with col2:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div class="card-icon">📄</div>
            <div class="card-title">Resume Analyzer</div>
            <div class="card-desc">Upload your PDF. RAG matches it against 12 job descriptions using SBERT cosine similarity. Get ATS score, skill gaps &amp; AI suggestions.</div>
            <div class="card-tag">RAG Match · ATS Score · Skill Gap</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Analyze Resume →", key="h_res"):
            st.session_state.page = "resume"; st.rerun()

    with col3:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div class="card-icon">💼</div>
            <div class="card-title">Job Role Predictor</div>
            <div class="card-desc">Enter your skills. RAG retrieves top matching roles from the job database. See match %, top hiring companies &amp; salary ranges.</div>
            <div class="card-tag">20 Roles · Top Companies · Salary Info</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Predict My Role →", key="h_job"):
            st.session_state.page = "jobs"; st.rerun()

    st.markdown('<hr class="spa-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#9098b1; font-size:0.78rem; padding:8px 0 20px;">
        RAG Flow: User Input → SBERT Encoding → Cosine Similarity Retrieval → Groq LLaMA3 Generation
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# INTERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def show_interview():
    st.markdown('<div class="sec-title">🎤 Interview Practice</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">RAG retrieves the best next question · Record your voice · Get AI feedback</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="rag-banner">
        <strong>RAG Pipeline:</strong> Domain selected → SBERT encodes knowledge base →
        Cosine similarity retrieves most varied question → Your spoken answer (speech-to-text) sent to Groq LLaMA3
    </div>
    """, unsafe_allow_html=True)



    DOMAINS = [
        "Machine Learning", "Deep Learning", "Data Science", "DSA",
        "Operating Systems", "DBMS", "Computer Networks", "OOPs",
        "SQL", "Artificial Intelligence", "Python", "System Design", "HR Round"
    ]
    SHORT = {
        "Machine Learning":"ML","Deep Learning":"DL","Data Science":"DS",
        "DSA":"DSA","Operating Systems":"OS","DBMS":"DBMS",
        "Computer Networks":"CN","OOPs":"OOP","SQL":"SQL",
        "Artificial Intelligence":"AI","Python":"PY","System Design":"SD","HR Round":"HR"
    }

    # ── Step 1: Domain + Question ──────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <div style="width:26px;height:26px;border-radius:50%;background:#5b4fcf;color:#fff;
            font-size:0.75rem;font-weight:700;display:flex;align-items:center;justify-content:center;">1</div>
        <span style="font-weight:700;color:#1a1a2e;font-size:0.95rem;">Select domain & get question</span>
    </div>
    """, unsafe_allow_html=True)

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        domain = st.selectbox("Select Domain", DOMAINS,
            index=DOMAINS.index(st.session_state.selected_domain), key="domain_sel", label_visibility="collapsed")
        st.session_state.selected_domain = domain
    with col_btn:
        if st.button("🎲 Get Question", key="btn_get_q"):
            q = get_rag_question(domain, st.session_state.answered_questions)
            st.session_state.current_question = q
            st.session_state.q_number += 1
            st.session_state.last_feedback = None
            st.session_state.last_stats = None
            st.rerun()

    # Progress
    total = 10
    n = st.session_state.q_number
    pct = min(int(n / total * 100), 100)
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;font-size:0.76rem;color:#9098b1;margin-top:8px;">
        <span>Session Progress</span><span>Q {n} / {total}</span>
    </div>
    <div class="prog-wrap"><div class="prog-fill" style="width:{pct}%"></div></div>
    """, unsafe_allow_html=True)

    if not st.session_state.current_question:
        st.info("👆 Select a domain and click **Get Question** to begin.")
        return

    # ── Question card ──────────────────────────────────────────────────────
    diff = get_question_difficulty(st.session_state.current_question)
    dc = {"Hard":"badge-hard","Medium":"badge-medium","Easy":"badge-easy"}.get(diff,"badge-easy")
    st.markdown(f"""
    <div class="q-card">
        <div style="margin-bottom:8px;">
            <span class="badge badge-domain">{SHORT.get(domain,domain)}</span>
            <span class="badge {dc}">{diff}</span>
            <span class="badge badge-rag">RAG Retrieved</span>
            <span style="font-size:0.76rem;color:#9098b1;">Q{n}</span>
        </div>
        <div class="q-text">{st.session_state.current_question}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 2: Voice Recorder ─────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin:18px 0 6px;">
        <div style="width:26px;height:26px;border-radius:50%;background:#5b4fcf;color:#fff;
            font-size:0.75rem;font-weight:700;display:flex;align-items:center;justify-content:center;">2</div>
        <span style="font-weight:700;color:#1a1a2e;font-size:0.95rem;">Record your voice answer</span>
    </div>
    """, unsafe_allow_html=True)

    # Voice recorder rendered via components.v1.html so JavaScript actually executes
    components.html("""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      * { box-sizing: border-box; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; margin:0; padding:0; }
      body { background: transparent; padding: 0; }
      .voice-card {
        background:#fff;
        border:1.5px solid #e8eaf0;
        border-radius:16px;
        padding:24px 20px;
        box-shadow:0 2px 12px rgba(0,0,0,0.04);
      }
      .rec-btn {
        width:72px; height:72px;
        border-radius:50%;
        border:none;
        cursor:pointer;
        display:flex; align-items:center; justify-content:center;
        margin:0 auto;
        font-size:1.8rem;
        transition:all 0.2s;
      }
      .rec-btn-idle   { background:#ede9ff; }
      .rec-btn-active { background:#fee2e2; animation: pulse 1.2s infinite; }
      @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(220,38,38,0.4); }
        70%  { box-shadow: 0 0 0 12px rgba(220,38,38,0); }
        100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
      }
      .wave-container {
        display:flex; align-items:center; justify-content:center; gap:4px; height:40px; margin:10px 0;
      }
      .wave-bar {
        width:4px; border-radius:2px; background:#5b4fcf;
        animation:wave 0.8s ease-in-out infinite alternate;
      }
      .wave-bar:nth-child(1){height:12px;animation-delay:0s;}
      .wave-bar:nth-child(2){height:24px;animation-delay:0.1s;}
      .wave-bar:nth-child(3){height:18px;animation-delay:0.2s;}
      .wave-bar:nth-child(4){height:32px;animation-delay:0.3s;}
      .wave-bar:nth-child(5){height:20px;animation-delay:0.4s;}
      .wave-bar:nth-child(6){height:28px;animation-delay:0.5s;}
      .wave-bar:nth-child(7){height:14px;animation-delay:0.6s;}
      .wave-bar:nth-child(8){height:30px;animation-delay:0.7s;}
      @keyframes wave { from { transform: scaleY(0.5); } to { transform: scaleY(1.0); } }
      .transcript-area {
        background:#f8f9fc;
        border:1.5px solid #e8eaf0;
        border-radius:12px;
        padding:16px;
        font-size:0.9rem;
        color:#1a1a2e;
        line-height:1.7;
        min-height:80px;
        margin:10px 0;
        word-wrap: break-word;
      }
      .filler-word { color:#dc2626; background:#fef2f2; border-radius:4px; padding:1px 4px; }
      .tech-word   { color:#16a34a; background:#f0fdf4; border-radius:4px; padding:1px 4px; }
      .re-rec-btn {
        background:transparent; border:1.5px solid #e8eaf0; border-radius:8px;
        padding:6px 16px; font-size:0.8rem; color:#6b7280; cursor:pointer; margin-top:8px;
      }
      .re-rec-btn:hover { background:#f8f9fc; }
      .copy-btn {
        background:#5b4fcf; color:#fff; border:none; border-radius:8px;
        padding:8px 20px; font-size:0.85rem; font-weight:600; cursor:pointer; margin-top:10px;
        width:100%;
      }
      .copy-btn:hover { background:#4a3fb5; }
      .copy-success { color:#16a34a; font-size:0.8rem; margin-top:6px; text-align:center; display:none; }
    </style>
    </head>
    <body>
    <div class="voice-card">
      <div style="text-align:center;">
        <button class="rec-btn rec-btn-idle" id="rec-btn" onclick="toggleRec()">🎤</button>
        <div style="font-size:1.6rem;font-weight:800;color:#1a1a2e;margin:10px 0 4px;" id="rec-timer">00:00</div>
        <div style="font-size:0.82rem;color:#9098b1;" id="rec-status">Click the mic to start recording</div>
        <div class="wave-container" id="wave-box" style="display:none;">
          <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
          <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
          <div class="wave-bar"></div><div class="wave-bar"></div>
        </div>
        <div style="font-size:0.75rem;color:#c4c9d8;margin-top:8px;">
          Speak clearly · answer in 1–2 minutes · works best in Chrome
        </div>
      </div>

      <div id="audio-player-row" style="display:none;margin-top:16px;align-items:center;gap:12px;
          background:#f8f9fc;border:1.5px solid #e8eaf0;border-radius:12px;padding:12px 16px;">
        <span style="font-size:0.8rem;color:#6b7280;white-space:nowrap;">🎧 Listen back:</span>
        <audio id="audio-playback" controls style="flex:1;height:36px;width:100%;"></audio>
      </div>

      <div id="transcript-box" style="display:none;margin-top:14px;">
        <div style="font-size:0.78rem;color:#9098b1;margin-bottom:6px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px;">
          <span>📝 Auto-transcribed text (speech-to-text)</span>
          <span style="font-size:0.72rem;">
            <span style="color:#dc2626;">■</span> filler words &nbsp;
            <span style="color:#16a34a;">■</span> tech terms
          </span>
        </div>
        <div class="transcript-area" id="transcript-live"></div>
        <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;">
          <button class="re-rec-btn" onclick="reRecord()">🔄 Re-record</button>
          <button class="copy-btn" onclick="copyTranscript()">📋 Copy transcript to answer box below</button>
        </div>
        <div class="copy-success" id="copy-success">✅ Copied! Paste it in the answer box below.</div>
      </div>
    </div>

    <script>
    var mediaRecorder = null;
    var audioChunks = [];
    var isRecording = false;
    var timerInterval = null;
    var seconds = 0;
    var recognition = null;
    var finalTranscript = "";

    function updateTimer() {
      seconds++;
      var m = Math.floor(seconds / 60);
      var s = seconds % 60;
      document.getElementById("rec-timer").textContent =
        (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
    }

    function highlightTranscript(text) {
      var fillerWords = ["umm","uh","like","you know","basically","literally","kind of","sort of","actually","i mean"];
      var techWords = ["algorithm","model","training","validation","accuracy","precision","recall",
        "overfitting","underfitting","regularization","gradient","neural","clustering",
        "classification","regression","feature","dataset","epoch","loss","embedding",
        "transformer","normalization","backpropagation","recursion","stack","queue",
        "tree","graph","hash","binary","database","index","query","join","transaction",
        "protocol","network","thread","process","deadlock","semaphore","mutex","kernel",
        "polymorphism","encapsulation","abstraction","inheritance"];
      var words = text.split(" ");
      return words.map(function(w) {
        var clean = w.toLowerCase().replace(/[.,!?]/g,"");
        if (fillerWords.includes(clean)) return '<span class="filler-word">' + w + '</span>';
        if (techWords.includes(clean)) return '<span class="tech-word">' + w + '</span>';
        return w;
      }).join(" ");
    }

    function startRecording() {
      if (!navigator.mediaDevices) {
        alert("Microphone not supported. Please use Chrome.");
        return;
      }
      finalTranscript = "";
      audioChunks = [];

      if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SR();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "en-US";
        recognition.onresult = function(e) {
          var interim = "";
          for (var i = e.resultIndex; i < e.results.length; i++) {
            if (e.results[i].isFinal) {
              finalTranscript += e.results[i][0].transcript + " ";
            } else {
              interim += e.results[i][0].transcript;
            }
          }
          var el = document.getElementById("transcript-live");
          if (el) el.innerHTML = highlightTranscript(finalTranscript) +
            '<span style="color:#9098b1;">' + interim + '</span>';
        };
        recognition.onerror = function(e) {
          document.getElementById("rec-status").textContent = "Speech recognition error: " + e.error + ". Type your answer below.";
        };
        recognition.start();
      } else {
        document.getElementById("rec-status").textContent = "Speech recognition not supported. Type your answer below.";
      }

      navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = function(e) { if(e.data.size > 0) audioChunks.push(e.data); };
        mediaRecorder.onstop = function() {
          var audioBlob = new Blob(audioChunks, { type: "audio/webm" });
          var audioUrl = URL.createObjectURL(audioBlob);
          var ap = document.getElementById("audio-playback");
          ap.src = audioUrl;
          document.getElementById("audio-player-row").style.display = "flex";
          document.getElementById("transcript-box").style.display = "block";
          document.getElementById("wave-box").style.display = "none";
          // stop all tracks
          stream.getTracks().forEach(function(t){ t.stop(); });
        };
        mediaRecorder.start();
        isRecording = true;
        seconds = 0;
        timerInterval = setInterval(updateTimer, 1000);
        document.getElementById("rec-btn").className = "rec-btn rec-btn-active";
        document.getElementById("rec-btn").innerHTML = "⏹";
        document.getElementById("rec-status").textContent = "Recording… click ⏹ to stop";
        document.getElementById("wave-box").style.display = "flex";
        document.getElementById("transcript-box").style.display = "none";
        document.getElementById("audio-player-row").style.display = "none";
      }).catch(function(err) {
        alert("Microphone access denied: " + err.message);
      });
    }

    function stopRecording() {
      if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        clearInterval(timerInterval);
        if (recognition) { try { recognition.stop(); } catch(e){} }
        document.getElementById("rec-btn").className = "rec-btn rec-btn-idle";
        document.getElementById("rec-btn").innerHTML = "🎤";
        document.getElementById("rec-status").textContent = "Done! Listen back or copy transcript below.";
      }
    }

    function toggleRec() {
      if (isRecording) stopRecording();
      else startRecording();
    }

    function reRecord() {
      finalTranscript = "";
      document.getElementById("transcript-live").innerHTML = "";
      document.getElementById("transcript-box").style.display = "none";
      document.getElementById("audio-player-row").style.display = "none";
      document.getElementById("rec-status").textContent = "Click the mic to start recording";
      document.getElementById("rec-timer").textContent = "00:00";
      document.getElementById("copy-success").style.display = "none";
    }

    function copyTranscript() {
      var text = finalTranscript.trim();
      if (!text) { alert("Nothing recorded yet."); return; }
      navigator.clipboard.writeText(text).then(function() {
        document.getElementById("copy-success").style.display = "block";
        setTimeout(function(){ document.getElementById("copy-success").style.display = "none"; }, 4000);
      }).catch(function() {
        // fallback
        var ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
        document.getElementById("copy-success").style.display = "block";
        setTimeout(function(){ document.getElementById("copy-success").style.display = "none"; }, 4000);
      });
    }
    </script>
    </body>
    </html>
    """, height=400, scrolling=False)

    # ── Step 3: Submit ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin:18px 0 10px;">
        <div style="width:26px;height:26px;border-radius:50%;background:#5b4fcf;color:#fff;
            font-size:0.75rem;font-weight:700;display:flex;align-items:center;justify-content:center;">3</div>
        <span style="font-weight:700;color:#1a1a2e;font-size:0.95rem;">
            Edit transcript if needed, then submit for AI feedback
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Editable transcript — user can also just type if mic doesn't work
    user_answer = st.text_area(
        "Your answer (auto-filled from voice, or type manually):",
        height=130,
        placeholder="Your spoken answer will appear here after recording stops. You can also type directly if you prefer...",
        key="ans_ta"
    )

    st.markdown("""
    <div style="font-size:0.75rem;color:#9098b1;margin-top:-8px;margin-bottom:12px;">
        💡 <strong>Tip:</strong> If speech-to-text doesn't work, just type your answer above directly.
        Voice recording works best in Google Chrome.
    </div>
    """, unsafe_allow_html=True)

    c_sub, c_skip = st.columns([3, 1])
    with c_sub:
        submitted = st.button("✅ Submit for AI Feedback", key="btn_submit")
    with c_skip:
        if st.button("Skip →", key="btn_skip"):
            st.session_state.current_question = ""
            st.session_state.last_feedback = None
            st.session_state.last_stats = None
            st.rerun()

    if submitted and user_answer.strip():
        with st.spinner("🔍 RAG evaluation pipeline running..."):
            stats = analyze_answer(user_answer)
            prompt = f"""You are an expert technical interview coach evaluating a student's answer.

Question (retrieved from RAG knowledge base): {st.session_state.current_question}
Student's answer: {user_answer}

Evaluate and respond in EXACTLY this format with NO extra text:
SCORE: [number 0-100]
STRENGTH: [one sentence - what was good about the answer]
IMPROVE: [one sentence - most important improvement needed]
MISSING: [one key concept or term they missed]
IDEAL_ANSWER: [2-3 sentences showing what a complete, ideal answer would include - cover the key concepts, technical terms, and points that would make this a perfect answer]
VOCABULARY: [rate their technical vocabulary: Poor / Average / Good / Excellent]"""

            feedback = call_groq(prompt,
                "You are a strict but fair technical interview coach. Always respond in the exact format.")

        st.session_state.last_feedback = feedback
        st.session_state.last_stats = stats
        st.session_state.answered_questions.append(st.session_state.current_question)

    # ── Step 4: AI Feedback ────────────────────────────────────────────────
    if st.session_state.last_feedback and st.session_state.last_stats:
        stats = st.session_state.last_stats
        feedback = st.session_state.last_feedback

        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin:22px 0 10px;">
            <div style="width:26px;height:26px;border-radius:50%;background:#16a34a;color:#fff;
                font-size:0.75rem;font-weight:700;display:flex;align-items:center;justify-content:center;">4</div>
            <span style="font-weight:700;color:#1a1a2e;font-size:0.95rem;">AI Feedback</span>
        </div>
        """, unsafe_allow_html=True)

        # Parse overall score from Groq feedback
        score_val = 70
        for line in feedback.split("\n"):
            if line.strip().startswith("SCORE:"):
                try:
                    score_val = int("".join(c for c in line.split(":")[1] if c.isdigit())[:3])
                except:
                    pass

        # Pull metrics from stats
        ans_acc    = stats.get("answer_accuracy", 70)
        tech_vocab = stats.get("tech_vocab", 60)
        clarity    = stats.get("clarity", 65)
        confidence = stats.get("confidence", 65)

        def bar_color(v):
            if v >= 75: return "#16a34a"
            if v >= 50: return "#d97706"
            return "#dc2626"

        # ── Score circle rendered via components.html (SVG always works here) ──
        dash = int(score_val * 3.015)
        components.html(f"""
        <!DOCTYPE html><html><head>
        <style>
          * {{ margin:0; padding:0; box-sizing:border-box; font-family:'Inter',-apple-system,sans-serif; }}
          body {{ background:transparent; }}
          .wrap {{
            background:#fff; border:1.5px solid #e8eaf0; border-radius:16px;
            padding:24px 28px; box-shadow:0 2px 12px rgba(0,0,0,0.04);
          }}
          .circle-wrap {{ text-align:center; margin-bottom:22px; position:relative; display:inline-block; }}
          .score-inner {{
            position:absolute; top:50%; left:50%;
            transform:translate(-50%,-50%); text-align:center;
          }}
          .score-num {{ font-size:1.6rem; font-weight:900; color:#1a1a2e; line-height:1; }}
          .score-lbl {{ font-size:0.6rem; color:#9098b1; margin-top:2px; }}
          .bar-row {{ margin-bottom:12px; }}
          .bar-header {{ display:flex; justify-content:space-between; font-size:0.82rem; color:#374151; margin-bottom:5px; }}
          .bar-val {{ font-weight:700; }}
          .bar-track {{ background:#e8eaf0; border-radius:20px; height:7px; overflow:hidden; }}
          .bar-fill {{ height:100%; border-radius:20px; }}
        </style>
        </head><body>
        <div class="wrap">
          <div style="text-align:center;">
            <div class="circle-wrap">
              <svg width="110" height="110" viewBox="0 0 110 110">
                <circle cx="55" cy="55" r="48" fill="none" stroke="#e8eaf0" stroke-width="9"/>
                <circle cx="55" cy="55" r="48" fill="none" stroke="#5b4fcf" stroke-width="9"
                  stroke-dasharray="{dash} 301.5" stroke-linecap="round"
                  transform="rotate(-90 55 55)"/>
              </svg>
              <div class="score-inner">
                <div class="score-num">{score_val}</div>
                <div class="score-lbl">overall</div>
              </div>
            </div>
          </div>

          <div class="bar-row">
            <div class="bar-header">
              <span>Answer accuracy</span>
              <span class="bar-val" style="color:{bar_color(ans_acc)};">{ans_acc}%</span>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width:{ans_acc}%;background:{bar_color(ans_acc)};"></div></div>
          </div>

          <div class="bar-row">
            <div class="bar-header">
              <span>Technical vocabulary</span>
              <span class="bar-val" style="color:{bar_color(tech_vocab)};">{tech_vocab}%</span>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width:{tech_vocab}%;background:{bar_color(tech_vocab)};"></div></div>
          </div>

          <div class="bar-row">
            <div class="bar-header">
              <span>Clarity of speech</span>
              <span class="bar-val" style="color:{bar_color(clarity)};">{clarity}%</span>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width:{clarity}%;background:{bar_color(clarity)};"></div></div>
          </div>

          <div class="bar-row">
            <div class="bar-header">
              <span>Confidence</span>
              <span class="bar-val" style="color:{bar_color(confidence)};">{confidence}%</span>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width:{confidence}%;background:{bar_color(confidence)};"></div></div>
          </div>
        </div>
        </body></html>
        """, height=340, scrolling=False)

        # ── Speaking Analysis ─────────────────────────────────────────────
        st.markdown("**🎙️ Speaking Analysis**")
        sa1, sa2, sa3, sa4 = st.columns(4)
        with sa1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val" style="color:#dc2626">{stats['filler_count']}</div>
                <div class="stat-label">filler words</div>
            </div>""", unsafe_allow_html=True)
        with sa2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val" style="color:#5b4fcf">{stats['wpm']}</div>
                <div class="stat-label">words/min</div>
            </div>""", unsafe_allow_html=True)
        with sa3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val" style="color:#16a34a">{stats['tech_count']}</div>
                <div class="stat-label">tech terms</div>
            </div>""", unsafe_allow_html=True)
        with sa4:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val" style="color:#d97706">{stats.get('long_pauses', 0)}</div>
                <div class="stat-label">long pauses</div>
            </div>""", unsafe_allow_html=True)

        if stats["tech_words_found"]:
            pills = " ".join([f'<span class="pill-found">✓ {w}</span>' for w in stats["tech_words_found"]])
            st.markdown(f"<div style='margin:10px 0 6px;'><strong>Tech terms detected:</strong> {pills}</div>", unsafe_allow_html=True)

        # ── Suggestions ───────────────────────────────────────────────────
        st.markdown("<br>**💡 Suggestions**", unsafe_allow_html=True)

        # Robustly parse feedback - handle possible extra spaces or colons
        feedback_data = {}
        for line in feedback.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            for key in ["SCORE", "STRENGTH", "IMPROVE", "MISSING", "IDEAL_ANSWER", "VOCABULARY"]:
                if line.upper().startswith(key + ":"):
                    feedback_data[key] = line[len(key)+1:].strip()
                    break

        if "STRENGTH" in feedback_data:
            st.markdown(f'''<div style="background:#f0fdf4;border-left:4px solid #16a34a;border-radius:0 10px 10px 0;padding:14px 16px;margin:8px 0;">
                <span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:6px;color:#15803d;">✅ Strength</span>
                <span style="font-size:0.88rem;color:#166534;line-height:1.6;">{feedback_data["STRENGTH"]}</span>
            </div>''', unsafe_allow_html=True)

        if "IMPROVE" in feedback_data:
            st.markdown(f'''<div style="background:#fffbeb;border-left:4px solid #d97706;border-radius:0 10px 10px 0;padding:14px 16px;margin:8px 0;">
                <span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:6px;color:#b45309;">⚡ Improve</span>
                <span style="font-size:0.88rem;color:#92400e;line-height:1.6;">{feedback_data["IMPROVE"]}</span>
            </div>''', unsafe_allow_html=True)

        if "MISSING" in feedback_data:
            st.markdown(f'''<div style="background:#fef2f2;border-left:4px solid #dc2626;border-radius:0 10px 10px 0;padding:14px 16px;margin:8px 0;">
                <span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:6px;color:#dc2626;">❗ Missing</span>
                <span style="font-size:0.88rem;color:#991b1b;line-height:1.6;">{feedback_data["MISSING"]}</span>
            </div>''', unsafe_allow_html=True)

        if "IDEAL_ANSWER" in feedback_data:
            st.markdown(f'''<div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);border:1px solid #93c5fd;border-left:4px solid #2563eb;border-radius:0 10px 10px 0;padding:14px 16px;margin:8px 0;">
                <span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:6px;color:#1d4ed8;">💡 Ideal Answer</span>
                <span style="font-size:0.88rem;color:#1e3a5f;line-height:1.6;">{feedback_data["IDEAL_ANSWER"]}</span>
            </div>''', unsafe_allow_html=True)

        if not any(k in feedback_data for k in ["STRENGTH", "IMPROVE", "MISSING"]):
            # Fallback: show raw feedback if parsing failed
            st.markdown(f'<div style="background:#f8f9fc;border-left:4px solid #5b4fcf;border-radius:0 10px 10px 0;padding:14px 16px;margin:8px 0;font-size:0.88rem;color:#374151;">{feedback}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Next Question →", key="btn_next"):
            q = get_rag_question(domain, st.session_state.answered_questions)
            st.session_state.current_question = q
            st.session_state.q_number += 1
            st.session_state.last_feedback = None
            st.session_state.last_stats = None
            st.rerun()

    if st.session_state.q_number > 0:
        st.markdown('<hr class="spa-divider">', unsafe_allow_html=True)
        if st.button("🔄 Reset Session", key="btn_reset"):
            st.session_state.answered_questions = []
            st.session_state.q_number = 0
            st.session_state.current_question = ""
            st.session_state.last_feedback = None
            st.session_state.last_stats = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RESUME
# ══════════════════════════════════════════════════════════════════════════════
def show_resume():
    st.markdown('<div class="sec-title">📄 Resume Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Upload your resume · RAG matches it to job descriptions · Get AI feedback</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="rag-banner">
        <strong>RAG Pipeline:</strong> PDF text extracted → SBERT encodes resume + 12 job descriptions →
        Cosine similarity retrieves top matching roles → Resume + retrieved context sent to Groq LLaMA3
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("📎 Drop your resume PDF here", type=["pdf"])

    if uploaded is not None:
        with st.spinner("🔍 Running RAG pipeline on your resume..."):
            resume_text = extract_text_from_pdf(uploaded)
            if not resume_text.strip():
                st.error("❌ Could not extract text. Make sure it's a text-based PDF (not scanned image).")
                return
            found_skills, missing_skills = extract_skills(resume_text)
            rag_results = rag_retrieve(resume_text)
            scores = compute_resume_scores(resume_text, found_skills)

        st.markdown('<hr class="spa-divider">', unsafe_allow_html=True)

        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown("**🔍 Skills Detected in Resume:**")
            found_html = "".join([f'<span class="pill-found">✓ {s}</span>' for s in found_skills])
            miss_html = "".join([f'<span class="pill-missing">✗ {s}</span>' for s in missing_skills])
            st.markdown(found_html + "<br><br>" + miss_html, unsafe_allow_html=True)

        with col_right:
            st.markdown("**📊 Resume Scores:**")
            for label, val, col_hex in [
                ("ATS Score", scores["ats"], "#2563eb"),
                ("Completeness", scores["completeness"], "#16a34a"),
                ("Keyword Match", scores["keyword_match"], "#d97706"),
            ]:
                st.markdown(f"""
                <div class="rs-row">
                    <span>{label}</span>
                    <span class="rs-pct" style="color:{col_hex}">{val}%</span>
                </div>
                <div class="sbar-wrap">
                    <div class="sbar-fill" style="width:{val}%;background:linear-gradient(90deg,{col_hex},{col_hex}88)"></div>
                </div>
                <br>
                """, unsafe_allow_html=True)

        st.markdown('<hr class="spa-divider">', unsafe_allow_html=True)
        st.markdown("**🎯 RAG Retrieved — Top Matching Job Profiles:**")
        st.markdown('<p style="font-size:0.78rem;color:#9098b1;">Retrieved from knowledge base using SBERT cosine similarity</p>', unsafe_allow_html=True)

        for i, r in enumerate(rag_results):
            top_cls = "job-card top" if i == 0 else "job-card"
            star = "⭐ " if i == 0 else ""
            comp_html = "".join([f'<span class="pill-company">{c}</span>' for c in r["companies"][:5]])
            st.markdown(f"""
            <div class="{top_cls}">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <div class="job-title">{star}{r['role']}</div>
                        <div style="margin-top:6px;">{comp_html}</div>
                        <div class="salary-pill">💰 {r['salary']}</div>
                    </div>
                    <div style="text-align:right;min-width:75px;">
                        <div class="job-score-big">{r['score']}%</div>
                        <div class="job-score-label">semantic match</div>
                    </div>
                </div>
                <div class="sbar-wrap" style="margin-top:10px;">
                    <div class="sbar-fill" style="width:{r['score']}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="spa-divider">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
            <div style="font-size:1.3rem;">🤖</div>
            <div>
                <div style="font-weight:800;font-size:1.05rem;color:#1a1a2e;">AI Suggestions</div>
                <div style="font-size:0.76rem;color:#9098b1;">Groq LLaMA3 · RAG augmented generation</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        retrieved_ctx = "\n".join([f"- {r['role']}: {r['desc'][:90]}" for r in rag_results])
        prompt = f"""You are a professional resume expert. Analyze this resume using the retrieved job descriptions as reference context.

Retrieved job descriptions from RAG knowledge base:
{retrieved_ctx}

Resume text (first 1400 chars):
{resume_text[:1400]}

Respond in EXACTLY this format with no extra text:
BEST_FIT_ROLE: [best role and one-sentence reason why]
STRENGTH: [what is strong about this resume in one sentence]
IMPROVE: [one specific improvement to make]
MISSING: [one important missing skill or section]"""

        with st.spinner("Generating AI feedback..."):
            feedback = call_groq(prompt, "You are a professional resume expert. Respond in the exact format given.")

        for line in feedback.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEST_FIT_ROLE:"):
                val = line[14:].strip()
                st.markdown(f"""
                <div style="background:#ede9ff;border-left:4px solid #5b4fcf;border-radius:0 12px 12px 0;
                    padding:14px 18px;margin:8px 0;">
                    <div style="font-size:0.72rem;font-weight:700;color:#5b4fcf;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">🎯 Best Fit Role</div>
                    <div style="font-size:0.92rem;color:#3730a3;line-height:1.55;">{val}</div>
                </div>
                """, unsafe_allow_html=True)
            elif line.startswith("STRENGTH:"):
                val = line[9:].strip()
                st.markdown(f"""
                <div style="background:#f0fdf4;border-left:4px solid #16a34a;border-radius:0 12px 12px 0;
                    padding:14px 18px;margin:8px 0;">
                    <div style="font-size:0.72rem;font-weight:700;color:#16a34a;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">✅ Strength</div>
                    <div style="font-size:0.92rem;color:#166534;line-height:1.55;">{val}</div>
                </div>
                """, unsafe_allow_html=True)
            elif line.startswith("IMPROVE:"):
                val = line[8:].strip()
                st.markdown(f"""
                <div style="background:#fffbeb;border-left:4px solid #d97706;border-radius:0 12px 12px 0;
                    padding:14px 18px;margin:8px 0;">
                    <div style="font-size:0.72rem;font-weight:700;color:#d97706;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">⚡ Improve</div>
                    <div style="font-size:0.92rem;color:#92400e;line-height:1.55;">{val}</div>
                </div>
                """, unsafe_allow_html=True)
            elif line.startswith("MISSING:"):
                val = line[8:].strip()
                st.markdown(f"""
                <div style="background:#fef2f2;border-left:4px solid #dc2626;border-radius:0 12px 12px 0;
                    padding:14px 18px;margin:8px 0;">
                    <div style="font-size:0.72rem;font-weight:700;color:#dc2626;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">❗ Missing</div>
                    <div style="font-size:0.92rem;color:#991b1b;line-height:1.55;">{val}</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# JOBS
# ══════════════════════════════════════════════════════════════════════════════
def show_jobs():
    st.markdown('<div class="sec-title">💼 Job Role Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Enter your skills · RAG finds best-fit roles · See companies &amp; salary</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="rag-banner">
        <strong>RAG Pipeline:</strong> Skills entered → SBERT encodes skills + 20 job roles →
        Cosine similarity retrieves top matches → Skills + retrieved roles sent to Groq LLaMA3 for career advice
    </div>
    """, unsafe_allow_html=True)

    skills_input = st.text_input(
        "Your Skills (comma separated):",
        placeholder="python, machine learning, sql, pandas, statistics, tensorflow..."
    )

    if st.button("🔍 Find My Best Roles", key="btn_find"):
        if not skills_input.strip():
            st.warning("Please enter at least one skill.")
            return

        with st.spinner("🔍 Running RAG retrieval on job database..."):
            results = predict_jobs(skills_input, top_k=5)

        st.markdown('<hr class="spa-divider">', unsafe_allow_html=True)
        st.markdown("**🎯 RAG Retrieved — Best Matching Roles:**")

        for i, job in enumerate(results):
            score = job["match_score"]
            top_cls = "job-card top" if i == 0 else "job-card"
            star = "⭐ " if i == 0 else ""
            comp_html = "".join([f'<span class="pill-company">{c}</span>' for c in job["companies"]])

            st.markdown(f"""
            <div class="{top_cls}">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div style="flex:1;">
                        <div class="job-title">{star}{job['job_role']}</div>
                        <div style="margin-top:7px;">{comp_html}</div>
                        <div class="salary-pill">💰 {job['salary']}</div>
                    </div>
                    <div style="text-align:right;min-width:75px;">
                        <div class="job-score-big">{score}%</div>
                        <div class="job-score-label">match</div>
                    </div>
                </div>
                <div class="sbar-wrap" style="margin-top:12px;">
                    <div class="sbar-fill" style="width:{score}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="spa-divider">', unsafe_allow_html=True)
        st.markdown("**🤖 Personalized Career Advice (Groq LLaMA3):**")

        retrieved_roles = "\n".join([f"- {j['job_role']} ({j['match_score']}% match)" for j in results])
        prompt = f"""You are an expert career counselor for engineering students in India.

Student's skills: {skills_input}

Job roles retrieved from RAG database:
{retrieved_roles}

Give a personalized, encouraging career explanation covering:
1. Which role is the BEST fit and exactly why (mention specific skills)
2. Which 2-3 skills they have are most in-demand
3. ONE specific skill to learn next to boost their profile the most

Keep it to 4-5 sentences. Be warm, specific, and actionable. Mention salary potential if relevant."""

        with st.spinner("Generating career advice..."):
            advice = call_groq(prompt, "You are an encouraging career counselor for Indian engineering students. Be specific and actionable.")

        st.markdown(f'<div class="fb-purple" style="font-size:0.92rem;line-height:1.75;">{advice}</div>', unsafe_allow_html=True)


# ── ROUTER ────────────────────────────────────────────────────────────────────
page = st.session_state.page
if page == "home":
    show_home()
elif page == "interview":
    show_interview()
elif page == "resume":
    show_resume()
elif page == "jobs":
    show_jobs()
else:
    show_home()

import os
import json
import hashlib
import html
import fitz
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# set_page_config must be the first Streamlit call (the original called st.error before it).
st.set_page_config(
    page_title="JobFit AI – AI Resume & Job Description Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "JobFit AI is an AI-powered resume and job description matcher that analyzes your "
                 "skills, experience, and ATS compatibility to help you understand how well your "
                 "resume matches a job."
    },
)

load_dotenv()

# ======================= ORIGINAL AI LOGIC (unchanged) =======================
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0, max_output_tokens=5000)

prompt = PromptTemplate(
    input_variables=["resume", "job_description"],
    template="""
You are an expert ATS Resume Analyzer and Technical Recruiter.

Compare the candidate's resume ONLY against the provided job description.

IMPORTANT RULES:
- Use ONLY information present in the resume and job description.
- Do NOT invent skills, experience, projects, education, or achievements.
- Give a realistic match score from 0 to 100.
- Analyze skills, experience, projects, education, and ATS keywords.
- Keep the analysis specific to this job.
- Return ONLY valid JSON.
- Do NOT use Markdown.
- Do NOT add ```json or ``` around the response.
- Do NOT add any explanation before or after the JSON.

RESUME:
{resume}

JOB DESCRIPTION:
{job_description}

Return JSON using EXACTLY this structure:

{{
    "match_score": 0,
    "verdict": "Strong Match",

    "breakdown": {{
        "skills": 0,
        "experience": 0,
        "ats": 0,
        "education": 0
    }},

    "matching_skills": ["skill1", "skill2"],
    "missing_skills": ["skill1", "skill2"],

    "partial_skills": [
        {{"skill": "skill name", "reason": "short explanation"}}
    ],

    "experience_match": {{"score": 0, "explanation": "short explanation"}},
    "education_match": {{"score": 0, "explanation": "short explanation"}},

    "strengths": ["strength1", "strength2", "strength3"],
    "gaps": ["gap1", "gap2", "gap3"],
    "ats_keywords": ["keyword1", "keyword2", "keyword3"],
    "recommendations": ["recommendation1", "recommendation2", "recommendation3"],
    "skills_to_learn": ["skill1", "skill2", "skill3"],

    "final_verdict": {{"category": "Good Match", "explanation": "short explanation of the overall result"}}
}}

SCORING GUIDELINES:

match_score:
- 85-100 = Strong Match
- 70-84 = Good Match
- 50-69 = Moderate Match
- 0-49 = Weak Match

For the breakdown scores:
- skills: How well the candidate's skills match the job requirements.
- experience: How well the candidate's experience/projects match the job.
- ats: How well the resume contains relevant job-specific keywords.
- education: How well the candidate's education matches the job requirement.

Return ONLY the JSON object.
"""
)
# =============================================================================

# ------------------------------- THEMES ---------------------------------------
THEMES = {
    "dark": {
        "bg": "#070B16", "glow1": "rgba(79,124,255,.18)", "glow2": "rgba(139,92,246,.16)",
        "surface": "rgba(255,255,255,.045)", "solid": "#0F1526", "border": "rgba(255,255,255,.11)",
        "text": "#F1F5F9", "muted": "#A8B3C7", "a1": "#5B8CFF", "a2": "#8B5CF6", "track": "#222B45",
        "good": "#4ADE80", "good_bg": "rgba(34,197,94,.13)", "ok": "#A3E635",
        "bad": "#F87171", "bad_bg": "rgba(239,68,68,.13)", "warn": "#FBBF24", "warn_bg": "rgba(245,158,11,.13)",
        "info": "#A5B4FC", "info_bg": "rgba(99,102,241,.16)", "shadow": "0 10px 30px rgba(0,0,0,.35)",
    },
    "light": {
        "bg": "#F5F7FB", "glow1": "rgba(79,124,255,.12)", "glow2": "rgba(139,92,246,.10)",
        "surface": "rgba(255,255,255,.85)", "solid": "#FFFFFF", "border": "#DDE3EE",
        "text": "#0F172A", "muted": "#475569", "a1": "#3B6BF5", "a2": "#7C3AED", "track": "#E2E8F0",
        "good": "#15803D", "good_bg": "#DCFCE7", "ok": "#4D7C0F",
        "bad": "#B91C1C", "bad_bg": "#FEE2E2", "warn": "#B45309", "warn_bg": "#FEF3C7",
        "info": "#4338CA", "info_bg": "#E0E7FF", "shadow": "0 8px 24px rgba(15,23,42,.08)",
    },
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root { __VARS__ }
html, body { overflow-x: hidden; }
.stApp { font-family: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; color: var(--text);
  background: radial-gradient(900px 420px at 12% -8%, var(--glow1), transparent),
              radial-gradient(800px 400px at 92% 0%, var(--glow2), transparent), var(--bg); }
.stApp p, .stApp label, .stApp li, .stApp span, .stApp h1, .stApp h2, .stApp h3 { color: var(--text); }
#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] { visibility: hidden; display: none; }
[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { max-width: 1100px; padding: 1.2rem 1.5rem 3rem; }
a { color: var(--a1); }
:focus-visible { outline: 3px solid var(--a1) !important; outline-offset: 2px; }

/* header */
.jf-top { display: flex; align-items: center; gap: 28px; flex-wrap: wrap; padding: 6px 0; }
.jf-brand { font-weight: 800; font-size: 22px; color: var(--text); text-decoration: none; }
.jf-brand b { background: linear-gradient(90deg, var(--a1), var(--a2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.jf-nav { display: flex; gap: 20px; flex-wrap: wrap; }
.jf-nav a { color: var(--muted); text-decoration: none; font-weight: 600; font-size: 15px; }
.jf-nav a:hover { color: var(--text); }

/* hero */
.jf-hero { text-align: center; padding: 56px 8px 36px; }
.jf-hero h1 { font-size: clamp(38px, 7vw, 66px); font-weight: 800; letter-spacing: -1.5px; margin: 0; line-height: 1.05;
  background: linear-gradient(90deg, var(--a1), var(--a2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.jf-hero h2 { font-size: clamp(22px, 4vw, 36px); font-weight: 700; margin: 14px 0 0; }
.jf-hero p { max-width: 680px; margin: 18px auto 28px; color: var(--muted) !important; font-size: 18px; line-height: 1.6; }
.jf-cta { display: inline-block; padding: 15px 30px; border-radius: 14px; font-weight: 700; font-size: 17px; color: #fff !important;
  text-decoration: none; background: linear-gradient(90deg, var(--a1), var(--a2)); box-shadow: 0 8px 24px rgba(91,92,246,.35); }
.jf-cta:hover { filter: brightness(1.08); }

/* cards + grids */
.jf-h { font-size: 26px; font-weight: 800; margin: 44px 0 6px; letter-spacing: -.4px; }
.jf-sub { color: var(--muted) !important; margin: 0 0 18px; }
.jf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; }
.jf-grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
.jf-card { background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: 20px;
  box-shadow: var(--shadow); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); }
.jf-card h3 { font-size: 17px; font-weight: 700; margin: 0 0 10px; }
.jf-card p { color: var(--muted) !important; margin: 0; font-size: 14.5px; line-height: 1.55; }
.jf-step .ic { font-size: 30px; }
.jf-step .no { color: var(--muted); font-size: 13px; font-weight: 600; margin-top: 8px; }
.jf-step .t { font-weight: 700; font-size: 17px; }
.jf-label { font-size: 18px; font-weight: 700; margin: 0 0 2px; }
.jf-hint { color: var(--muted) !important; font-size: 14px; margin: 0 0 10px; }

/* streamlit widgets */
[data-testid="stFileUploaderDropzone"] { background: var(--surface); border: 2px dashed var(--a1); border-radius: 18px; padding: 28px 18px; }
[data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button { color: var(--text) !important; background: var(--solid); border: 1px solid var(--border); border-radius: 10px; }
[data-testid="stFileUploaderFile"] * { color: var(--text) !important; }
[data-baseweb="textarea"], [data-baseweb="base-input"] { background: var(--solid) !important; border-radius: 16px !important; }
.stTextArea textarea { background: var(--solid) !important; color: var(--text) !important; -webkit-text-fill-color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 16px !important; font-size: 15px; padding: 14px; }
.stTextArea textarea::placeholder { color: var(--muted) !important; -webkit-text-fill-color: var(--muted) !important; }
[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border) !important; border-radius: 14px; }
[data-testid="stExpander"] summary * { color: var(--text) !important; }
button[kind="primary"], button[data-testid="stBaseButton-primary"] { width: 100%; min-height: 54px; border: none; border-radius: 14px;
  font-size: 17px; font-weight: 700; color: #fff !important; background: linear-gradient(90deg, var(--a1), var(--a2)); }
button[kind="primary"] *, button[data-testid="stBaseButton-primary"] * { color: #fff !important; }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] { min-height: 44px; border-radius: 12px; font-weight: 600;
  background: var(--surface); border: 1px solid var(--border); }
button[kind="secondary"] *, button[data-testid="stBaseButton-secondary"] * { color: var(--text) !important; }

/* results */
.jf-ring { width: 176px; height: 176px; border-radius: 50%; margin: 6px auto 12px; display: grid; place-items: center;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), var(--track) 0); }
.jf-ring-in { width: 140px; height: 140px; border-radius: 50%; background: var(--solid); display: flex; flex-direction: column; align-items: center; justify-content: center; }
.jf-ring-in b { font-size: 40px; font-weight: 800; color: var(--c); line-height: 1; }
.jf-ring-in span { font-size: 12px; font-weight: 700; color: var(--muted); margin-top: 6px; letter-spacing: .5px; }
.jf-verdict { font-size: 24px; font-weight: 800; text-align: center; }
.jf-big { font-size: 30px; font-weight: 800; }
.jf-bar { background: var(--track); border-radius: 99px; height: 10px; overflow: hidden; margin-top: 10px; }
.jf-bar i { display: block; height: 100%; border-radius: 99px; }
.jf-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.jf-chip { padding: 7px 13px; border-radius: 99px; font-size: 14px; font-weight: 600; border: 1px solid currentColor; }
.jf-chip.good { color: var(--good); background: var(--good_bg); }
.jf-chip.bad { color: var(--bad); background: var(--bad_bg); }
.jf-chip.info { color: var(--info); background: var(--info_bg); border-radius: 10px; }
.jf-row { display: flex; gap: 12px; align-items: flex-start; padding: 12px 14px; border-radius: 12px; margin-bottom: 8px; border: 1px solid currentColor; font-size: 14.5px; line-height: 1.5; }
.jf-row.good { color: var(--good); background: var(--good_bg); }
.jf-row.bad { color: var(--bad); background: var(--bad_bg); }
.jf-row span { color: var(--text) !important; }
.jf-num { min-width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center; font-weight: 700; color: #fff; background: linear-gradient(135deg, var(--a1), var(--a2)); }

/* footer */
.jf-footer { margin-top: 64px; padding: 28px 0 8px; border-top: 1px solid var(--border); text-align: center; }
.jf-footer .n { font-weight: 800; font-size: 20px; }
.jf-footer p { color: var(--muted) !important; margin: 4px 0; }
.jf-links { display: flex; justify-content: center; flex-wrap: wrap; gap: 12px; margin: 14px 0; }
.jf-links a { display: inline-flex; align-items: center; gap: 8px; padding: 9px 16px; border-radius: 12px; border: 1px solid var(--border);
  background: var(--surface); color: var(--text) !important; text-decoration: none; font-weight: 600; }
.jf-links a:hover { border-color: var(--a1); }
.jf-links svg { fill: currentColor; }

@media (max-width: 640px) {
  .block-container { padding: .8rem 1rem 2rem; }
  .jf-hero { padding: 36px 0 24px; }
  .jf-hero p { font-size: 16px; }
  .jf-grid2 { grid-template-columns: 1fr; }
  .jf-cta, button[kind="primary"] { width: 100%; text-align: center; }
  .jf-h { margin-top: 32px; font-size: 22px; }
}
"""


# ------------------------------- helpers ---------------------------------------
def block(s):
    """Render HTML without indentation (leading spaces would turn it into a code block)."""
    st.markdown(" ".join(line.strip() for line in s.splitlines() if line.strip()), unsafe_allow_html=True)


def esc(v):
    return html.escape(str(v))


def pct(v):
    try:
        return max(0, min(100, int(float(v))))
    except (TypeError, ValueError):
        return 0


def as_list(v):
    return v if isinstance(v, list) else []


def score_var(s):
    return "--good" if s >= 85 else "--ok" if s >= 70 else "--warn" if s >= 50 else "--bad"


def bar(v, var="--a1"):
    return f'<div class="jf-bar" role="progressbar" aria-valuenow="{v}" aria-valuemin="0" aria-valuemax="100"><i style="width:{v}%;background:var({var})"></i></div>'


def chips(items, kind, icon):
    return '<div class="jf-chips">' + "".join(f'<span class="jf-chip {kind}">{icon} {esc(i)}</span>' for i in items) + "</div>"


def friendly_api_error(e):
    msg = str(e).lower()
    if "429" in msg or "quota" in msg or "resource_exhausted" in msg or "rate" in msg:
        return "The AI service is busy or the free quota has been used up. Please wait a minute and try again."
    if "api key" in msg or "api_key" in msg or "401" in msg or "403" in msg or "permission" in msg:
        return "The AI service rejected the API key. Please check that the Gemini API key is valid."
    return "The AI service couldn't complete the analysis. Please try again in a moment."


# ------------------------------- UI sections -----------------------------------
def apply_custom_css():
    vars_css = ";".join(f"--{k}:{v}" for k, v in THEMES[st.session_state.theme].items())
    block(f"<style>{CSS.replace('__VARS__', vars_css)}</style>")


def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


def render_header():
    left, right = st.columns([5, 1.4], vertical_alignment="center")
    with left:
        block("""<div class="jf-top"><a class="jf-brand" href="#top">🎯 JobFit <b>AI</b></a>
        <nav class="jf-nav" aria-label="Main"><a href="#analyze">Home / Analyze</a><a href="#how-it-works">How it works</a></nav></div>""")
    with right:
        dark = st.session_state.theme == "dark"
        st.button("☀️ Light mode" if dark else "🌙 Black mode", on_click=toggle_theme, key="theme_btn",
                  use_container_width=True, help="Switch between Black and Light theme")


def render_hero():
    block("""<div class="jf-hero" id="top"><h1>JobFit AI</h1><h2>Your Resume. Their Job. Know Your Match.</h2>
    <p>AI-powered resume and job description matching that helps you understand your ATS compatibility,
    identify missing skills, and improve your chances of getting noticed.</p>
    <a class="jf-cta" href="#analyze">Analyze My Resume →</a></div>""")


def render_how_it_works():
    steps = [("📄", "Upload Resume", "Add your resume as a PDF."), ("💼", "Add Job Description", "Paste the role you want."),
             ("🤖", "AI Analysis", "Your resume is compared with the job."), ("🎯", "Get Your Match Score", "See your score, gaps and fixes.")]
    cards = "".join(f'<div class="jf-card jf-step"><div class="ic">{ic}</div><div class="no">Step {i}</div><div class="t">{t}</div><p>{d}</p></div>'
                    for i, (ic, t, d) in enumerate(steps, 1))
    block(f'<div id="how-it-works" class="jf-h">How it works</div><p class="jf-sub">Four steps from resume to match score.</p><div class="jf-grid">{cards}</div>')


def render_upload_section():
    block('<div id="analyze" class="jf-h">Resume job match analyzer</div><p class="jf-sub">Upload your resume, paste the job description, then run the analysis.</p>')
    c1, c2 = st.columns(2, gap="large")
    with c1:
        block('<div class="jf-label">📄 Upload your resume</div><div class="jf-hint">Accepted format: PDF (text-based PDFs work best).</div>')
        resume = st.file_uploader("Upload Resume", type=["pdf"], label_visibility="collapsed")
    with c2:
        block('<div class="jf-label">💼 Job description</div><div class="jf-hint">Paste the full job posting you are applying for.</div>')
        jd = st.text_area("Job Description", height=210, label_visibility="collapsed",
                          placeholder="Paste the job description here: responsibilities, required skills, qualifications...")
    st.write("")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        clicked = st.button("Analyze Resume", type="primary", use_container_width=True)
    return resume, jd, clicked


def run_analysis(resume, jd):
    """Original pipeline (PDF text -> prompt -> Gemini -> JSON), with friendlier errors and no repeat calls."""
    data = resume.getvalue()
    key = hashlib.sha256(data + jd.strip().encode()).hexdigest()
    if st.session_state.get("analysis_key") == key and st.session_state.get("analysis"):
        return  # same inputs: reuse stored result, no extra API call
    with st.spinner("Reading resume and running analysis..."):
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            resume_text = "".join(page.get_text() for page in doc)
            doc.close()
        except Exception:
            st.error("That file couldn't be read as a PDF. Please upload a valid, non-corrupted PDF.")
            return
        if not resume_text.strip():
            st.error("Couldn't extract any text from that PDF (it may be a scanned image). Try a text-based PDF.")
            return
        final_prompt = prompt.format(resume=resume_text, job_description=jd)
        try:
            response = llm.invoke(final_prompt)
        except Exception as e:
            st.error(friendly_api_error(e))
            return
        if isinstance(response.content, str):
            result = response.content
        else:
            result = "".join(item.get("text", "") for item in response.content
                             if isinstance(item, dict) and item.get("type") == "text")
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()
        try:
            analysis = json.loads(result)
            if not isinstance(analysis, dict):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            st.error("The AI returned an unexpected response. Please try again.")
            return
    st.session_state.analysis, st.session_state.analysis_key = analysis, key


def render_results(a):
    score = pct(a.get("match_score", 0))
    verdict = esc(a.get("verdict", "Unknown"))
    var = score_var(score)
    bd = a.get("breakdown", {}) if isinstance(a.get("breakdown"), dict) else {}
    block('<div class="jf-h" id="results">🎯 Your resume analysis</div><p class="jf-sub">AI-powered compatibility analysis for this job.</p>')

    block(f"""<div class="jf-card"><div class="jf-ring" style="--p:{score};--c:var({var})" role="img" aria-label="ATS match {score} percent">
    <div class="jf-ring-in"><b>{score}%</b><span>ATS MATCH</span></div></div><div class="jf-verdict">{verdict}</div></div>""")

    st.write("")
    labels = [("skills", "💻 Skills"), ("experience", "💼 Experience"), ("ats", "🤖 ATS Compatibility"), ("education", "🎓 Education")]
    cards = ""
    for k, lab in labels:
        v = pct(bd.get(k, 0))
        cards += f'<div class="jf-card"><div class="jf-hint">{lab}</div><div class="jf-big" style="color:var({score_var(v)})">{v}%</div>{bar(v, score_var(v))}</div>'
    block(f'<div class="jf-h" style="margin-top:8px">Match breakdown</div><div class="jf-grid">{cards}</div>')

    ms, mis = as_list(a.get("matching_skills")), as_list(a.get("missing_skills"))
    block(f"""<div class="jf-grid2" style="margin-top:16px">
    <div class="jf-card"><h3>✓ Matched skills</h3>{chips(ms, "good", "✓") if ms else "<p>No strong matching skills found.</p>"}</div>
    <div class="jf-card"><h3>✕ Missing skills</h3>{chips(mis, "bad", "✕") if mis else "<p>No major missing skills found.</p>"}</div></div>""")

    st.markdown("#### Partially matching skills")
    partial = [p for p in as_list(a.get("partial_skills")) if isinstance(p, dict)]
    if partial:
        for p in partial:
            with st.expander(str(p.get("skill", "Unknown"))):
                st.write(p.get("reason", ""))
    else:
        st.caption("No partially matching skills identified.")

    def detail(title, d):
        d = d if isinstance(d, dict) else {}
        s = pct(d.get("score", 0))
        return f'<div class="jf-card"><h3>{title}</h3><div class="jf-big" style="color:var({score_var(s)})">{s}% match</div>{bar(s, score_var(s))}<p style="margin-top:12px">{esc(d.get("explanation", ""))}</p></div>'
    block(f'<div class="jf-grid2" style="margin-top:16px">{detail("💼 Experience match", a.get("experience_match"))}{detail("🎓 Education match", a.get("education_match"))}</div>')

    rows = lambda items, kind, ic: "".join(f'<div class="jf-row {kind}"><b>{ic}</b><span>{esc(i)}</span></div>' for i in items) or "<p>Nothing listed.</p>"
    block(f"""<div class="jf-grid2" style="margin-top:16px">
    <div class="jf-card"><h3>💪 Resume strengths</h3>{rows(as_list(a.get("strengths")), "good", "✓")}</div>
    <div class="jf-card"><h3>🔍 Resume gaps</h3>{rows(as_list(a.get("gaps")), "bad", "⚠")}</div></div>""")

    kws = as_list(a.get("ats_keywords"))
    if kws:
        block(f'<div class="jf-card" style="margin-top:16px"><h3>🔑 ATS keywords</h3>{chips(kws, "info", "#")}</div>')

    recs = "".join(f'<div class="jf-row" style="border-color:var(--border);align-items:center"><div class="jf-num">{i}</div><span>{esc(r)}</span></div>'
                   for i, r in enumerate(as_list(a.get("recommendations")), 1))
    if recs:
        block(f'<div class="jf-card" style="margin-top:16px"><h3>🛠️ Improvement suggestions</h3>{recs}</div>')

    learn = as_list(a.get("skills_to_learn"))
    if learn:
        block(f'<div class="jf-card" style="margin-top:16px"><h3>🚀 Recommended skills to learn</h3>{chips(learn, "info", "🚀")}</div>')

    fv = a.get("final_verdict", {}) if isinstance(a.get("final_verdict"), dict) else {}
    block(f"""<div class="jf-card" style="margin-top:16px;text-align:center;border-color:var(--a1)"><div class="jf-hint">FINAL VERDICT</div>
    <div class="jf-verdict" style="color:var(--info)">{esc(fv.get("category", a.get("verdict", "")))}</div>
    <p style="margin-top:8px">{esc(fv.get("explanation", ""))}</p></div>""")


def render_about():
    block("""<div class="jf-h">AI resume matcher and ATS resume checker</div>
    <p class="jf-sub" style="max-width:760px">JobFit AI is an AI-powered resume and job description matcher that analyzes your skills,
    experience, and ATS compatibility to help you understand how well your resume matches a job. Use it as a resume analyzer to see which
    skills match, which are missing, and what to change for better resume optimization before you apply.</p>""")


def render_footer():
    gh = '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 .5C5.73.5.5 5.73.5 12c0 5.08 3.29 9.38 7.86 10.9.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.36-3.88-1.36-.52-1.33-1.28-1.69-1.28-1.69-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.02 1.75 2.69 1.24 3.35.95.1-.74.4-1.24.73-1.53-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.28 1.18-3.09-.12-.29-.51-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.62 1.59.23 2.76.11 3.05.74.81 1.18 1.83 1.18 3.09 0 4.42-2.69 5.39-5.25 5.68.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 23.5 12C23.5 5.73 18.27.5 12 .5z"/></svg>'
    li = '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M20.45 20.45h-3.56v-5.57c0-1.33-.03-3.04-1.85-3.04-1.86 0-2.14 1.45-2.14 2.95v5.66H9.34V9h3.42v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12zM7.12 20.45H3.56V9h3.56v11.45zM22.23 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 24 1.77 24h20.46c.98 0 1.77-.77 1.77-1.73V1.73C24 .77 23.21 0 22.23 0z"/></svg>'
    block(f"""<div class="jf-footer"><div class="n">JobFit AI</div><p>AI-powered resume and job matching.</p><p>Built by <b>Harish Shimpi</b></p>
    <div class="jf-links"><a href="https://github.com/HarishCodess" target="_blank" rel="noopener noreferrer">{gh} GitHub</a>
    <a href="https://www.linkedin.com/in/harish-shimpi-25188a337" target="_blank" rel="noopener noreferrer">{li} LinkedIn</a></div>
    <p>© 2026 JobFit AI</p></div>""")


# --------------------------------- main -----------------------------------------
st.session_state.setdefault("theme", "dark")
apply_custom_css()

if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    st.error("No Gemini API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) in a `.env` file "
             "locally, or in Secrets on Streamlit Community Cloud.")
    st.stop()

render_header()
render_hero()
render_how_it_works()
resume, job_description, analyze = render_upload_section()

if analyze:
    if resume is None:
        st.warning("Please upload your resume (PDF).")
    elif not job_description.strip():
        st.warning("Please enter the job description.")
    else:
        run_analysis(resume, job_description)

# Results live in session_state, so switching theme (a rerun) never re-calls the AI or loses them.
if st.session_state.get("analysis"):
    render_results(st.session_state.analysis)

render_about()
render_footer()
import os
import json
import fitz  
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    st.error(
        "No Gemini API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) "
        "in a `.env` file — see `.env.example`."
    )
    st.stop()

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


st.set_page_config(
    page_title="JOBFIT AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #0F1117; }
    .block-container { padding-top: 3em; padding-left: 5%; padding-right: 5%; }
    .main-title { font-size: 52px; font-weight: 700; text-align: center; margin-bottom: 10px; color: #FFFFFF; }
    .gradient-text {
        background: linear-gradient(90deg, #6366F1, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle { text-align: center; color: #9CA3AF; font-size: 20px; margin-bottom: 45px; }
    .card-title { font-size: 20px; font-weight: 600; color: #FFFFFF; margin-bottom: 6px; }
    .card-description { color: #9CA3AF; font-size: 14px; margin-bottom: 16px; }
    .stButton > button {
        width: 100%; border-radius: 10px; height: 50px; font-size: 17px;
        font-weight: 600; background-color: #6366F1; color: white; border: none;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
    Your resume. Their job.
    <br>
    <span class="gradient-text">Know your match.</span>
</div>
<div class="subtitle">AI-powered resume and job compatibility analysis</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="card-title"> Upload your resume</div>
        <div class="card-description">Upload your latest resume in PDF format.</div>
    """, unsafe_allow_html=True)
    resume = st.file_uploader("Upload Resume", type=["pdf"], label_visibility="collapsed")

with col2:
    st.markdown("""
        <div class="card-title">Job description</div>
        <div class="card-description">Paste the job description you're applying for.</div>
    """, unsafe_allow_html=True)
    job_description = st.text_area(
        "Job Description",
        placeholder="Paste the job description here...",
        height=200,
        label_visibility="collapsed"
    )

st.write("")
_, button_col, _ = st.columns([1, 2, 1])
with button_col:
    analyze = st.button(" Analyze My Match", use_container_width=True)

# ---------------- Button logic (everything below is now correctly scoped) ----------------
if analyze:
    if resume is None:
        st.warning("Please upload your resume.")
    elif not job_description.strip():
        st.warning("Please enter the job description.")
    else:
        with st.spinner("Reading resume and running analysis..."):
            # Extract PDF text
            doc = fitz.open(stream=resume.read(), filetype="pdf")
            resume_text = ""
            for page in doc:
                resume_text += page.get_text()
            doc.close()

            if not resume_text.strip():
                st.error("Couldn't extract any text from that PDF (it may be a scanned image). Try a text-based PDF.")
                st.stop()

            final_prompt = prompt.format(resume=resume_text, job_description=job_description)

            try:
                response = llm.invoke(final_prompt)
            except Exception as e:
                st.error(f"Gemini API call failed: {e}")
                st.stop()

            # Response content can be a plain string or a list of content blocks
            if isinstance(response.content, str):
                result = response.content
            else:
                result = "".join(
                    item.get("text", "")
                    for item in response.content
                    if isinstance(item, dict) and item.get("type") == "text"
                )

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
            except json.JSONDecodeError:
                st.error("AI returned an invalid response. Please try again.")
                st.code(result, language="text")
                st.stop()

        
        st.markdown("---")
        st.markdown("<h1 style='text-align:center;'>🎯 Resume Analysis</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center;color:#9CA3AF;'>AI-powered compatibility analysis</p>",
            unsafe_allow_html=True
        )
        st.write("")

        # Match score
        score = analysis.get("match_score", 0)
        verdict = analysis.get("verdict", "Unknown")

        if score >= 85:
            score_color = "#22C55E"
        elif score >= 70:
            score_color = "#84CC16"
        elif score >= 50:
            score_color = "#F59E0B"
        else:
            score_color = "#EF4444"

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #181B25, #202432); padding: 30px;
                border-radius: 20px; text-align: center; border: 1px solid #2D3342; margin-bottom: 25px;">
                <div style="color:#9CA3AF; font-size:16px; margin-bottom:10px;">OVERALL MATCH SCORE</div>
                <div style="font-size:64px; font-weight:700; color:{score_color};">{score}%</div>
                <div style="font-size:20px; font-weight:600; color:white; margin-bottom:18px;">{verdict}</div>
                <div style="background:#303543; border-radius:20px; height:12px; width:80%; margin:auto; overflow:hidden;">
                    <div style="background:{score_color}; width:{score}%; height:100%; border-radius:20px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        
        st.markdown("### 📊 Match Breakdown")
        breakdown = analysis.get("breakdown", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("💻 Skills", f"{breakdown.get('skills', 0)}%")
        with c2:
            st.metric("💼 Experience", f"{breakdown.get('experience', 0)}%")
        with c3:
            st.metric("🤖 ATS", f"{breakdown.get('ats', 0)}%")
        with c4:
            st.metric("🎓 Education", f"{breakdown.get('education', 0)}%")

        st.write("")

        
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("### Matching Skills")
            matching_skills = analysis.get("matching_skills", [])
            if matching_skills:
                html = "".join(
                    f'<span style="display:inline-block;background:#123524;color:#4ADE80;'
                    f'padding:8px 14px;margin:4px;border-radius:20px;font-size:14px;'
                    f'border:1px solid #166534;">✓ {s}</span>'
                    for s in matching_skills
                )
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("No strong matching skills found.")

        with sc2:
            st.markdown("###  Missing Skills")
            missing_skills = analysis.get("missing_skills", [])
            if missing_skills:
                html = "".join(
                    f'<span style="display:inline-block;background:#3A1717;color:#F87171;'
                    f'padding:8px 14px;margin:4px;border-radius:20px;font-size:14px;'
                    f'border:1px solid #7F1D1D;">✕ {s}</span>'
                    for s in missing_skills
                )
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.success("No major missing skills found.")

        st.write("")

        st.markdown("###  Partially Matching Skills")
        partial_skills = analysis.get("partial_skills", [])
        if partial_skills:
            for item in partial_skills:
                skill = item.get("skill", "Unknown")
                reason = item.get("reason", "")
                with st.expander(f"{skill}"):
                    st.write(reason)
        else:
            st.success("No partially matching skills identified.")

     
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("###  Experience Match")
            experience = analysis.get("experience_match", {})
            exp_score = experience.get("score", 0)
            st.progress(min(exp_score, 100) / 100)
            st.markdown(f"**{exp_score}% Match**")
            st.write(experience.get("explanation", ""))

        with ec2:
            st.markdown("###  Education Match")
            education = analysis.get("education_match", {})
            edu_score = education.get("score", 0)
            st.progress(min(edu_score, 100) / 100)
            st.markdown(f"**{edu_score}% Match**")
            st.write(education.get("explanation", ""))

        st.write("")

       
        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown("###  Resume Strengths")
            for strength in analysis.get("strengths", []):
                st.markdown(
                    f'<div style="background:#151E19;border:1px solid #166534;padding:12px 15px;'
                    f'border-radius:10px;margin-bottom:8px;color:#D1FAE5;">✓ {strength}</div>',
                    unsafe_allow_html=True
                )
        with gc2:
            st.markdown("### Resume Gaps")
            for gap in analysis.get("gaps", []):
                st.markdown(
                    f'<div style="background:#241919;border:1px solid #7F1D1D;padding:12px 15px;'
                    f'border-radius:10px;margin-bottom:8px;color:#FECACA;">⚠ {gap}</div>',
                    unsafe_allow_html=True
                )

        st.write("")

        
        st.markdown("###  ATS Keywords")
        keywords = analysis.get("ats_keywords", [])
        if keywords:
            html = "".join(
                f'<span style="display:inline-block;background:#1E1B4B;color:#A5B4FC;'
                f'padding:8px 14px;margin:4px;border-radius:8px;border:1px solid #4338CA;">{k}</span>'
                for k in keywords
            )
            st.markdown(html, unsafe_allow_html=True)

        st.write("")

        
        st.markdown("###  Resume Improvement Suggestions")
        for i, rec in enumerate(analysis.get("recommendations", []), start=1):
            st.markdown(
                f"""
                <div style="display:flex;gap:15px;align-items:center;background:#181B25;
                    border:1px solid #2D3342;padding:15px;border-radius:12px;margin-bottom:10px;">
                    <div style="background:#6366F1;width:30px;height:30px;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;font-weight:bold;color:white;">{i}</div>
                    <div style="color:#E5E7EB;">{rec}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        
        st.markdown("###  Recommended Skills to Learn")
        skills_to_learn = analysis.get("skills_to_learn", [])
        if skills_to_learn:
            cols = st.columns(len(skills_to_learn))
            for col, skill in zip(cols, skills_to_learn):
                with col:
                    st.markdown(
                        f"""
                        <div style="background:linear-gradient(135deg,#1E1B4B,#312E81);padding:18px;
                            border-radius:12px;text-align:center;border:1px solid #4338CA;
                            color:#C7D2FE;font-weight:600;">🚀 {skill}</div>
                        """,
                        unsafe_allow_html=True
                    )

        st.write("")

        
        final_verdict = analysis.get("final_verdict", {})
        category = final_verdict.get("category", verdict)
        explanation = final_verdict.get("explanation", "")

        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#181B25,#202432);border:1px solid #6366F1;
                padding:25px;border-radius:18px;text-align:center;margin-top:20px;">
                <div style="color:#9CA3AF;font-size:14px;margin-bottom:8px;">FINAL VERDICT</div>
                <div style="color:#A5B4FC;font-size:30px;font-weight:700;margin-bottom:10px;">{category}</div>
                <div style="color:#D1D5DB;font-size:15px;">{explanation}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

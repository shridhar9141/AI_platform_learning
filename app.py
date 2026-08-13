import streamlit as st
import pandas as pd
import json
import plotly.express as px
from groq import Groq

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AdaptiLearn AI - Adaptive Learning Platform",
    page_icon="🎓",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "name": "Learner",
        "subject": "Python Data Structures & Algorithms",
        "level": "Intermediate",
        "learning_style": "Practical / Code-First",
        "mastery_score": 65
    }

if "quiz_history" not in st.session_state:
    st.session_state.quiz_history = [
        {"topic": "Core Concepts", "score": 80, "difficulty": "Medium", "date": "2026-08-10"},
        {"topic": "Advanced Applications", "score": 50, "difficulty": "Medium", "date": "2026-08-11"},
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("🎓 AdaptiLearn AI")
st.sidebar.markdown("---")

groq_api_key = st.sidebar.text_input("Enter Groq API Key:", type="password", help="Get key from console.groq.com")
if not groq_api_key:
    # Fallback to Streamlit Secrets if available
    groq_api_key = st.secrets.get("GROQ_API_KEY", "")

selected_model = st.sidebar.selectbox(
    "Select Model Engine:",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    index=0
)

# --- HELPER FUNCTIONS & GROQ LLM INTEGRATION ---
def get_groq_client(api_key):
    if not api_key:
        st.error("⚠️ Please provide a Groq API Key in the sidebar or via Streamlit Secrets!")
        st.stop()
    return Groq(api_key=api_key)

def generate_adaptive_assessment(subject, level, style, mastery_score, api_key):
    """Generates an assessment question tailored to ANY subject dynamically."""
    client = get_groq_client(api_key)
    prompt = f"""
    Act as an adaptive EdTech assessment engine.
    Generate 1 multiple-choice quiz question tailored to the following parameters:
    - Target Subject / Field of Study: {subject}
    - Proficiency Level: {level}
    - Current Mastery Score: {mastery_score}%
    - Preferred Learning Style: {style}

    Return ONLY a valid JSON object matching this schema:
    {{
        "question": "The quiz question text",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "answer": "Option A",
        "explanation": "Detailed step-by-step reasoning explaining why the answer is correct.",
        "difficulty": "Easy/Medium/Hard",
        "topic": "Subtopic Name"
    }}
    Do NOT include markdown code block markers like ```json.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You output strict raw JSON without formatting markup."},
                {"role": "user", "content": prompt}
            ],
            model=selected_model,
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        st.error(f"Error generating assessment: {str(e)}")
        return None

# --- AI AGENT 1: CONCEPT ANALYZER & MISCONCEPTION AGENT ---
def run_concept_analyzer_agent(user_answer, question_data, subject, api_key):
    """Autonomous AI Agent that analyzes learner mistakes and suggests interventions."""
    client = get_groq_client(api_key)
    is_correct = (user_answer == question_data["answer"])
    
    agent_system_prompt = f"""
    You are the 'Concept Diagnostic & Remediation Agent', an autonomous AI agent in an EdTech ecosystem.
    Your job is to analyze learner submissions in '{subject}' and diagnose cognitive gaps.

    Structure your response strictly as follows:
    ### 🕵️ Agent Diagnosis
    - **Accuracy State:** {'Correct' if is_correct else 'Incorrect'}
    - **Identified Misconception / Key Insight:** [Analyze why the learner chose this option or praise their reasoning]
    
    ### 🎯 Targeted Remediation
    - **Actionable Focus:** [1-2 sentences directing the student on what specific concept in {subject} to review next]
    """

    user_payload = f"""
    Subject: {subject}
    Question: {question_data['question']}
    Options: {question_data['options']}
    Learner Selected: {user_answer}
    Correct Answer: {question_data['answer']}
    Explanatory Context: {question_data['explanation']}
    """
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": agent_system_prompt},
            {"role": "user", "content": user_payload}
        ],
        model=selected_model,
        temperature=0.4
    )
    return is_correct, response.choices[0].message.content

def get_recommended_pathway(subject, history, api_key):
    """Generates a dynamic learning pathway for any given subject."""
    client = get_groq_client(api_key)
    prompt = f"""
    The learner is studying '{subject}' and has the following performance history:
    {json.dumps(history)}

    Based on this data, recommend a structured 3-step personalized adaptive learning pathway for '{subject}'.
    For each step, include:
    1. Topic / Module Name
    2. Reason for Recommendation (gap analysis)
    3. Suggested Resource / Activity Type (e.g., Interactive Lab, Case Study, Practice Quiz)
    """
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=selected_model,
        temperature=0.4
    )
    return response.choices[0].message.content

# --- MAIN APP INTERFACE ---
st.title("🧠 Universal Adaptive AI Learning Engine")
st.caption("Personalized pathways, dynamic assessments, and Agentic diagnostic feedback for ANY subject powered by Groq LLMs.")

tabs = st.tabs(["📊 Learner Dashboard", "🧩 Adaptive Assessment", "🗺️ Dynamic Pathway", "🤖 AI Tutor"])

# --- TAB 1: LEARNER DASHBOARD & SUBJECT SELECTION ---
with tabs[0]:
    st.subheader("Learner Profile & Dynamic Subject Selection")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Subject & Personalization Settings**")
        st.session_state.user_profile["name"] = st.text_input("Learner Name", st.session_state.user_profile["name"])
        
        # Any subject selection feature
        subject_mode = st.radio("Subject Selection Mode:", ["Preset List", "Custom Subject"])
        if subject_mode == "Preset List":
            st.session_state.user_profile["subject"] = st.selectbox(
                "Target Subject", 
                [
                    "Python Data Structures & Algorithms",
                    "Precision Agriculture & Smart Farming",
                    "Machine Learning Fundamentals",
                    "Cybersecurity & Network Defense",
                    "Organic Chemistry",
                    "Financial Markets & Economics"
                ]
            )
        else:
            st.session_state.user_profile["subject"] = st.text_input(
                "Enter ANY Custom Subject:",
                value=st.session_state.user_profile["subject"],
                help="Type any topic e.g., Astrophysics, Marine Biology, World History"
            )

        st.session_state.user_profile["level"] = st.select_slider(
            "Proficiency Level", 
            options=["Beginner", "Intermediate", "Advanced"],
            value=st.session_state.user_profile["level"]
        )
        st.session_state.user_profile["learning_style"] = st.selectbox(
            "Learning Preference",
            ["Practical / Problem-Solving", "Visual / Intuitive", "Theoretical / Deep Dive"]
        )
        
    with col2:
        df_history = pd.DataFrame(st.session_state.quiz_history)
        avg_score = int(df_history["score"].mean()) if not df_history.empty else 0
        st.session_state.user_profile["mastery_score"] = avg_score
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Mastery", f"{avg_score}%", delta=f"{avg_score - 50}% baseline")
        m2.metric("Assessments Taken", len(df_history))
        m3.metric("Current Track", st.session_state.user_profile["level"])
        
        st.info(f"📍 **Active Learning Domain:** `{st.session_state.user_profile['subject']}`")
        
        if not df_history.empty:
            fig = px.line(
                df_history, 
                x="date", 
                y="score", 
                color="topic",
                title=f"Mastery Progression in {st.session_state.user_profile['subject']}",
                markers=True,
                range_y=[0, 100]
            )
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: ADAPTIVE ASSESSMENT WITH AI AGENT ---
with tabs[1]:
    st.subheader(f"Adaptive Assessment: {st.session_state.user_profile['subject']}")
    st.markdown("Questions and evaluation are dynamically tuned to your chosen subject.")
    
    if st.button("🎲 Generate Next Question", type="primary"):
        with st.spinner(f"Generating adaptive question for '{st.session_state.user_profile['subject']}' via Groq..."):
            st.session_state.current_q = generate_adaptive_assessment(
                st.session_state.user_profile["subject"],
                st.session_state.user_profile["level"],
                st.session_state.user_profile["learning_style"],
                st.session_state.user_profile["mastery_score"],
                groq_api_key
            )
            st.session_state.submitted = False

    if "current_q" in st.session_state and st.session_state.current_q:
        q_data = st.session_state.current_q
        
        st.markdown(f"**Subject:** `{st.session_state.user_profile['subject']}` | **Topic:** `{q_data.get('topic', 'General')}` | **Difficulty:** `{q_data.get('difficulty', 'Medium')}`")
        st.info(f"**Question:** {q_data['question']}")
        
        selected_option = st.radio("Choose your answer:", q_data["options"], key="quiz_options")
        
        if st.button("Submit Answer") and not st.session_state.get("submitted", False):
            st.session_state.submitted = True
            with st.spinner("🤖 Running Concept Analyzer AI Agent..."):
                is_correct, agent_feedback = run_concept_analyzer_agent(
                    selected_option, 
                    q_data, 
                    st.session_state.user_profile["subject"], 
                    groq_api_key
                )
                
                score = 100 if is_correct else 30
                st.session_state.quiz_history.append({
                    "topic": q_data.get("topic", "General"),
                    "score": score,
                    "difficulty": q_data.get("difficulty", "Medium"),
                    "date": "2026-08-13"
                })
                
                if is_correct:
                    st.success("🎉 Correct Answer!")
                else:
                    st.error(f"❌ Incorrect. Correct Answer: **{q_data['answer']}**")
                
                st.markdown("---")
                st.markdown(agent_feedback)
                with st.expander("Full Explanation Breakdown"):
                    st.write(q_data["explanation"])

# --- TAB 3: DYNAMIC PATHWAY ---
with tabs[2]:
    st.subheader(f"Recommended Pathway for {st.session_state.user_profile['subject']}")
    
    if st.button("🔄 Generate Custom Pathway"):
        with st.spinner(f"Analyzing skill gaps in {st.session_state.user_profile['subject']}..."):
            pathway_res = get_recommended_pathway(
                st.session_state.user_profile["subject"],
                st.session_state.quiz_history,
                groq_api_key
            )
            st.session_state.recommended_pathway = pathway_res

    if "recommended_pathway" in st.session_state:
        st.markdown(st.session_state.recommended_pathway)
    else:
        st.info("Click the button above to generate a customized learning roadmap for your active subject.")

# --- TAB 4: AI TUTOR CHAT ---
with tabs[3]:
    st.subheader(f"AI Tutor for {st.session_state.user_profile['subject']}")
    st.caption("Ask questions, request examples, or explore complex topics within your chosen subject.")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if user_prompt := st.chat_input(f"Ask a question about {st.session_state.user_profile['subject']}..."):
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                client = get_groq_client(groq_api_key)
                system_prompt = f"You are an expert AI tutor teaching '{st.session_state.user_profile['subject']}' to an '{st.session_state.user_profile['level']}' level student."
                
                messages = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history
                response = client.chat.completions.create(
                    messages=messages,
                    model=selected_model,
                    temperature=0.6
                )
                bot_reply = response.choices[0].message.content
                st.markdown(bot_reply)
                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

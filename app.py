import streamlit as st
from agents.coach import InterviewCoach
import uuid

st.set_page_config(page_title="AI Interview Coach", page_icon="🎯", layout="wide")

# Initialize session state
if "coach" not in st.session_state:
    st.session_state.coach = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "interview_complete" not in st.session_state:
    st.session_state.interview_complete = False

# Sidebar configuration
with st.sidebar:
    st.header("🎯 Interview Setup")

    position = st.text_input("Position", "Senior Python Developer")
    level = st.selectbox("Level", ["junior", "mid", "senior", "staff"])
    interview_type = st.selectbox("Type", ["technical", "behavioral", "system_design"])

    job_desc = st.text_area(
        "Job Description (optional)",
        placeholder="Paste job description for targeted questions..."
    )

    num_questions = st.slider("Number of Questions", 3, 10, 5)

    if st.button("Start Interview", type="primary"):
        st.session_state.coach = InterviewCoach(
            job_descriptions_dir=None,
            interview_type=interview_type,
            level=level,
            position=position,
            max_questions=num_questions,
        )
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.interview_complete = False

        # Get first question
        topics = ["core skills", "system design", "problem solving", "experience", "culture fit"]
        welcome = st.session_state.coach.start_interview(
            st.session_state.session_id,
            topics[:num_questions]
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome})
        st.rerun()

# Main content
st.title("🎯 AI Interview Coach")

if st.session_state.coach is None:
    st.info("👈 Configure your interview in the sidebar and click 'Start Interview'")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "feedback" in message:
                with st.expander("View Feedback"):
                    fb = message["feedback"]
                    st.metric("Score", f"{fb.score}/10")
                    st.write(f"**Understanding:** {fb.understanding}")
                    if fb.improvements:
                        st.write("**Tips:**")
                        for tip in fb.improvements:
                            st.write(f"- {tip}")

    # Chat input
    if not st.session_state.interview_complete:
        if prompt := st.chat_input("Your answer..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get response
            result = st.session_state.coach.submit_answer(
                st.session_state.session_id,
                prompt
            )

            if result["is_complete"]:
                st.session_state.interview_complete = True
                # Generate report
                report = st.session_state.coach.generate_report(st.session_state.session_id)

                report_content = f"""
## Interview Complete! 🎉

**Overall Score: {report.overall_score}/10**
**Recommendation: {report.recommendation.upper()}**

### Summary
{report.summary}

### Strengths
{chr(10).join('- ' + s for s in report.strengths)}

### Areas to Improve
{chr(10).join('- ' + a for a in report.areas_to_improve)}

### Suggested Topics to Study
{chr(10).join('- ' + t for t in report.suggested_topics_to_study)}
"""
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": report_content
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["next_question"],
                    "feedback": result["feedback"]
                })

            st.rerun()
    else:
        st.success("Interview complete! Check the report above.")
        if st.button("Start New Interview"):
            st.session_state.coach = None
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.interview_complete = False
            st.rerun()
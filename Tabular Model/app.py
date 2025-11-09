import streamlit as st
import joblib
import pandas as pd
import numpy as np

bundle = joblib.load("svm_with_scaler.pkl")
model = bundle["model"]
scaler = bundle["scaler"]

emotion_map = {
    "happy": 0,
    "neutral": 1,
    "sad": 2,
    "tired": 3
}

pose_cols = ["pose_center", "pose_down", "pose_left", "pose_right", "pose_up"]

feature_order = [
    "blink_rate", "yawn_count", "gaze_on_screen", "head_movement",
    "emotion_encoded", "pose_center", "pose_down", "pose_left",
    "pose_right", "pose_up"
]

# Quiz questions
quiz_questions = [
    {
        "question": "What is the output of: print(2 ** 3)?",
        "options": ["5", "6", "8", "9"],
        "answer": 2
    },
    {
        "question": "Which data type is used to store True/False values?",
        "options": ["int", "str", "bool", "float"],
        "answer": 2
    },
    {
        "question": "What does 'len([1, 2, 3])' return?",
        "options": ["1", "2", "3", "6"],
        "answer": 2
    }
]


def predict_attention(blink_count, yawn_count, gaze_on_screen, head_movement_count,
                      emotion, head_pose):
    blink_rate = blink_count / 10
    yawn_rate = yawn_count / 10
    head_movement_rate = head_movement_count / 10

    emotion_encoded = emotion_map.get(emotion.lower(), 1)

    pose_vector = {col: 0 for col in pose_cols}
    pose_key = f"pose_{head_pose.lower()}"
    if pose_key in pose_vector:
        pose_vector[pose_key] = 1

    input_data = pd.DataFrame([{
        "blink_rate": blink_rate,
        "yawn_count": yawn_rate,
        "gaze_on_screen": gaze_on_screen,
        "head_movement": head_movement_rate,
        "emotion_encoded": emotion_encoded,
        **pose_vector
    }])[feature_order]

    scaled_input = scaler.transform(input_data)
    pred = model.predict(scaled_input)[0]

    label = "bore" if pred == 0 else "engaged"

    return label


st.set_page_config(page_title="Boredom Detection System", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: #0f0f1e;
        background-image: 
            radial-gradient(at 20% 30%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
            radial-gradient(at 80% 70%, rgba(168, 85, 247, 0.15) 0px, transparent 50%),
            radial-gradient(at 50% 50%, rgba(236, 72, 153, 0.1) 0px, transparent 50%);
        animation: gradientShift 15s ease infinite;
    }

    @keyframes gradientShift {
        0%, 100% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(20deg); }
    }

    .main .block-container {
        padding: 3rem;
        max-width: 1400px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .custom-header {
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeInDown 0.8s ease;
    }

    .custom-header h1 {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .custom-header p {
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 300;
    }

    .neuro-card {
        background: rgba(30, 30, 46, 0.6);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        padding: 2.5rem;
        box-shadow: 
            20px 20px 60px rgba(0, 0, 0, 0.5),
            -20px -20px 60px rgba(255, 255, 255, 0.03),
            inset 0 0 0 1px rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.8s ease;
    }

    .neuro-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            25px 25px 80px rgba(0, 0, 0, 0.6),
            -25px -25px 80px rgba(255, 255, 255, 0.04),
            inset 0 0 0 1px rgba(255, 255, 255, 0.08);
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .section-header {
        color: #e2e8f0;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-left: 1rem;
        border-left: 4px solid #667eea;
    }

    .stNumberInput label, .stSlider label, .stSelectbox label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
    }

    .stNumberInput input, .stSelectbox select {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(100, 116, 139, 0.3) !important;
        border-radius: 15px !important;
        color: #e2e8f0 !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: inset 2px 2px 5px rgba(0, 0, 0, 0.3) !important;
    }

    .stNumberInput input:hover, .stSelectbox select:hover {
        border-color: rgba(102, 126, 234, 0.5) !important;
        background: rgba(15, 23, 42, 0.8) !important;
    }

    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #667eea !important;
        box-shadow: 
            inset 2px 2px 5px rgba(0, 0, 0, 0.3),
            0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        background: rgba(15, 23, 42, 0.9) !important;
    }

    .stSlider {
        padding: 1rem 0;
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        height: 6px !important;
        border-radius: 10px !important;
    }

    .stSlider > div > div > div > div > div {
        background: #ffffff !important;
        border: 3px solid #667eea !important;
        width: 20px !important;
        height: 20px !important;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.4) !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 1rem 2rem;
        border: none;
        border-radius: 20px;
        box-shadow: 
            0 10px 30px rgba(102, 126, 234, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-top: 2rem;
        position: relative;
        overflow: hidden;
    }

    .stButton > button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }

    .stButton > button:hover:before {
        left: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 15px 40px rgba(102, 126, 234, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }

    .stButton > button:active {
        transform: translateY(-1px) scale(1);
    }

    .stAlert {
        border-radius: 20px;
        border: none !important;
        padding: 2rem;
        margin-top: 2rem;
        animation: resultPop 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        position: relative;
        overflow: hidden;
    }

    /* WHITE TEXT FOR ALERTS */
    .stAlert > div {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    @keyframes resultPop {
        0% {
            opacity: 0;
            transform: scale(0.8);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }

    .stSuccess {
        background: rgba(16, 185, 129, 0.15) !important;
        border: 2px solid rgba(16, 185, 129, 0.5) !important;
        box-shadow: 0 10px 40px rgba(16, 185, 129, 0.3);
    }

    .stError {
        background: rgba(239, 68, 68, 0.15) !important;
        border: 2px solid rgba(239, 68, 68, 0.5) !important;
        box-shadow: 0 10px 40px rgba(239, 68, 68, 0.3);
    }

    .stInfo {
        background: rgba(59, 130, 246, 0.15) !important;
        border: 2px solid rgba(59, 130, 246, 0.5) !important;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);
    }

    .stWarning {
        background: rgba(245, 158, 11, 0.15) !important;
        border: 2px solid rgba(245, 158, 11, 0.5) !important;
        box-shadow: 0 10px 40px rgba(245, 158, 11, 0.3);
    }

    [data-testid="column"] {
        padding: 0 1rem;
    }

    h3 {
        color: #e2e8f0;
        font-weight: 700;
        font-size: 1.5rem;
        text-align: center;
        margin: 2rem 0 1rem 0;
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
    }

    .metric-box {
        background: rgba(30, 30, 46, 0.4);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }

    .metric-box:hover {
        background: rgba(30, 30, 46, 0.6);
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateX(5px);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .metric-value {
        color: #e2e8f0;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    @keyframes popIn {
        0% {
            opacity: 0;
            transform: scale(0.9) translateY(20px);
        }
        100% {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }

    .quiz-header {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .quiz-subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    .quiz-question {
        color: #e2e8f0;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
        padding: 1rem;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 15px;
        border-left: 4px solid #667eea;
    }

    .stRadio > label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        display: none;
    }

    .stRadio > div {
        background: rgba(15, 23, 42, 0.4);
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid rgba(100, 116, 139, 0.2);
    }

    .stRadio > div > label {
        color: #e2e8f0 !important;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        cursor: pointer;
        margin: 0.3rem 0;
        background: rgba(30, 30, 46, 0.3);
        border: 1px solid rgba(100, 116, 139, 0.2);
    }

    .stRadio > div > label:hover {
        background: rgba(102, 126, 234, 0.15);
        border-color: rgba(102, 126, 234, 0.4);
        transform: translateX(5px);
    }

    .quiz-score {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin: 2rem 0;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
        border-radius: 20px;
        border: 2px solid rgba(102, 126, 234, 0.6);
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        animation: scoreReveal 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }

    @keyframes scoreReveal {
        0% {
            opacity: 0;
            transform: scale(0.5) rotate(-10deg);
        }
        100% {
            opacity: 1;
            transform: scale(1) rotate(0deg);
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-header">
    <h1>🧠 Boredom Detection System</h1>
    <p>Real-time boredom detection powered by machine learning</p>
</div>
""", unsafe_allow_html=True)

if 'show_quiz' not in st.session_state:
    st.session_state.show_quiz = False
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = [None, None, None]
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="neuro-card">', unsafe_allow_html=True)
    blink_count = st.number_input("Blink Count", min_value=0, max_value=100, value=10, key="blink")
    yawn_count = st.number_input("Yawn Count", min_value=0, max_value=50, value=2, key="yawn")
    gaze_on_screen = st.slider("Gaze on Screen (%)", min_value=0, max_value=100, value=75, key="gaze")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="neuro-card">', unsafe_allow_html=True)
    head_movement = st.number_input("Head Movements", min_value=0, max_value=100, value=5, key="head")
    emotion = st.selectbox("Emotional State", ["happy", "neutral", "sad", "tired"], key="emotion")
    head_pose = st.selectbox("Head Position", ["center", "down", "left", "right", "up"], key="pose")
    st.markdown('</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b:
    predict_btn = st.button("🔍 Detect Boredom Level")

if predict_btn:
    with st.spinner("🧠 Analyzing behavioral patterns..."):
        result = predict_attention(
            blink_count=blink_count,
            yawn_count=yawn_count,
            gaze_on_screen=gaze_on_screen,
            head_movement_count=head_movement,
            emotion=emotion,
            head_pose=head_pose
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if result == "bore":
        st.error("⚠️ BOREDOM DETECTED - User is feeling bored")
        st.session_state.show_quiz = True
        st.session_state.quiz_answers = [None, None, None]
        st.session_state.quiz_submitted = False
    else:
        st.success("✅ ENGAGED - User is actively focused and attentive")
        st.session_state.show_quiz = False

# QUIZ POPUP
if st.session_state.show_quiz:
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Quiz container with close button
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(30, 30, 46, 0.98) 0%, rgba(20, 20, 36, 0.98) 100%);
                backdrop-filter: blur(30px);
                border-radius: 30px;
                padding: 3rem;
                border: 2px solid rgba(102, 126, 234, 0.4);
                box-shadow: 0 30px 80px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(255, 255, 255, 0.1);
                position: relative;
                animation: popIn 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);">
        <div class="quiz-header">📚 Quick Programming Quiz</div>
        <div class="quiz-subtitle">Answer these questions to re-engage your focus!</div>
    </div>
    """, unsafe_allow_html=True)

    col_close1, col_close2 = st.columns([10, 1])
    with col_close2:
        if st.button("✕", key="close_quiz", help="Close quiz"):
            st.session_state.show_quiz = False
            st.session_state.quiz_submitted = False
            st.rerun()

    # Questions
    for i, q in enumerate(quiz_questions):
        st.markdown(f'<div class="quiz-question">Question {i + 1}: {q["question"]}</div>', unsafe_allow_html=True)
        st.session_state.quiz_answers[i] = st.radio(
            f"Select answer for Q{i + 1}",
            options=q["options"],
            key=f"q{i}",
            label_visibility="collapsed"
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # Submit button
    col_submit1, col_submit2, col_submit3 = st.columns([1, 2, 1])
    with col_submit2:
        if st.button("📝 Submit Quiz", key="submit_quiz", use_container_width=True):
            score = 0
            for i, q in enumerate(quiz_questions):
                if st.session_state.quiz_answers[i] == q["options"][q["answer"]]:
                    score += 1

            st.session_state.quiz_submitted = True

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="quiz-score">🎯 Your Score: {score}/3</div>', unsafe_allow_html=True)

            if score == 3:
                st.success("🎉 Perfect score! Stay Focused!")
            elif score >= 2:
                st.info("👍 Good job! Keep focusing!")
            else:
                st.warning("💪 Please Focus!")

    st.markdown("<br><br>", unsafe_allow_html=True)

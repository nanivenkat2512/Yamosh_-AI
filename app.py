import os
import json
import re
import textwrap
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Yamosh AI | Smart Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"
CHUNK_SIZE = 14000


# =====================================================
# 2. SESSION STATE
# =====================================================

DEFAULT_STATE = {
    "study_data": None,
    "source_notes": "",
    "quiz_submitted": False,
    "quiz_attempt": 0,
    "submitted_answers": []
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# 3. GEMINI CONNECTION
# =====================================================

# Load API key from local .env file
api_key = os.getenv("GEMINI_API_KEY")

# If not found, try Streamlit Cloud Secrets
if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None

# Check API key
if not api_key:
    st.error(
        "Gemini API key not found. "
        "Please configure GEMINI_API_KEY in your .env file "
        "or Streamlit Cloud Secrets."
    )
    st.stop()

# Connect to Gemini
try:
    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": 120000}
    )

except Exception as error:
    st.error(f"Gemini connection failed: {error}")
    st.stop()
# =====================================================
# 4. HELPER FUNCTIONS
# =====================================================

def ask_gemini(prompt):
    """Send a prompt to Gemini and return its response."""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text


def parse_json_response(response_text):
    """Parse JSON safely, including responses wrapped in Markdown fences."""

    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    cleaned = response_text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Gemini did not return valid JSON.")

    return json.loads(cleaned[start:end + 1])


def split_text_into_chunks(text, max_chars=CHUNK_SIZE):
    """Split long study material into manageable sections."""

    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        while len(paragraph) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            chunks.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]

        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) + 2 > max_chars:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = paragraph

        else:
            if current_chunk:
                current_chunk += "\n\n"

            current_chunk += paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def validate_key_points(key_points):
    """Check that key points have titles and explanations."""

    if not isinstance(key_points, list) or not key_points:
        raise ValueError("No important key points were generated.")

    for point in key_points:
        if not isinstance(point, dict):
            raise ValueError("A key point has an invalid format.")

        title = point.get("title")
        detail = point.get("detail")

        if not isinstance(title, str) or not title.strip():
            raise ValueError("A key point is missing its title.")

        if not isinstance(detail, str) or not detail.strip():
            raise ValueError("A key point is missing its explanation.")


def validate_questions(questions):
    """Check that the quiz contains exactly 10 valid questions."""

    if not isinstance(questions, list) or len(questions) != 10:
        raise ValueError("The quiz must contain exactly 10 questions.")

    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"Question {index} has an invalid format.")

        if not isinstance(question.get("question"), str):
            raise ValueError(f"Question {index} is missing its question text.")

        options = question.get("options")

        if (
            not isinstance(options, list)
            or len(options) != 4
            or not all(
                isinstance(option, str) and option.strip()
                for option in options
            )
            or len(set(options)) != 4
        ):
            raise ValueError(
                f"Question {index} must have four different options."
            )

        if question.get("answer") not in options:
            raise ValueError(
                f"Question {index} has an answer outside its options."
            )

        explanation = question.get("explanation")

        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(
                f"Question {index} is missing its explanation."
            )


# =====================================================
# 5. GENERATE COMPREHENSIVE STUDY MATERIAL
# =====================================================

def generate_study_content(notes):
    """Analyze study material in chunks and generate a quiz."""

    chunks = split_text_into_chunks(notes)

    if not chunks:
        raise ValueError("No readable study material was found.")

    all_summary_points = []
    all_key_points = []

    progress = st.progress(0)
    status = st.empty()

    total_chunks = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        status.write(
            f"📖 Analyzing section {index} of {total_chunks}..."
        )

        prompt = """
You are an expert academic study assistant.

Carefully analyze the complete study section.

SUMMARY REQUIREMENTS:

1. Include all major topics and subtopics.
2. Include important definitions and concepts.
3. Include technical terminology.
4. Include principles, methods and processes.
5. Include algorithms and formulas when present.
6. Include examples and applications.
7. Include advantages and disadvantages.
8. Include limitations and challenges.
9. Include comparisons and differences.
10. Preserve the original logical order.
11. Use clear and simple academic language.
12. Do not invent facts.
13. Do not skip important information.
14. Generate as many summary points as necessary.
15. There is no fixed point limit.

KEY POINT REQUIREMENTS:

1. Extract all important exam-related concepts.
2. Include definitions and technical terms.
3. Include important facts and formulas.
4. Include methods and applications.
5. Include comparisons and examples.
6. Give every key point a title and explanation.
7. Do not impose a fixed number of key points.
8. Avoid unnecessary repetition.

Return ONLY valid JSON in this structure:

{
    "summary": [
        "Summary point"
    ],
    "key_points": [
        {
            "title": "Concept title",
            "detail": "Complete explanation"
        }
    ]
}

The entries above are examples of the required structure.
Replace them with actual information from the study section.

STUDY SECTION:

""" + chunk

        response_text = ask_gemini(prompt)
        section_data = parse_json_response(response_text)

        if not isinstance(section_data, dict):
            raise ValueError(
                f"Invalid response in section {index}."
            )

        summary = section_data.get("summary")
        key_points = section_data.get("key_points")

        if (
            not isinstance(summary, list)
            or not summary
            or not all(
                isinstance(point, str) and point.strip()
                for point in summary
            )
        ):
            raise ValueError(
                f"Summary generation failed in section {index}."
            )

        validate_key_points(key_points)

        all_summary_points.extend(summary)
        all_key_points.extend(key_points)

        progress.progress(index / total_chunks)

    status.write("🧠 Creating your 10-question quiz...")

    study_reference = {
        "summary": all_summary_points,
        "key_points": all_key_points
    }

    quiz_prompt = """
You are an expert academic quiz creator.

Create exactly 10 multiple-choice questions using the
complete study reference.

REQUIREMENTS:

1. Generate exactly 10 questions.
2. Cover different major topics and subtopics.
3. Include important definitions and concepts.
4. Include technical terminology and principles.
5. Include applications and processes.
6. Include comparisons and differences.
7. Include reasoning-based questions where appropriate.
8. Avoid unnecessary repetition.
9. Use four different options for each question.
10. Only one answer must be correct.
11. The answer must exactly match one option.
12. Explain the correct answer clearly.
13. Do not invent facts.
14. Use only the supplied study reference.

Return ONLY valid JSON:

{
    "questions": [
        {
            "question": "Question text?",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": "Option A",
            "explanation": "Explanation"
        }
    ]
}

Return exactly 10 questions.

STUDY REFERENCE:

""" + json.dumps(study_reference, ensure_ascii=False)

    quiz_data = parse_json_response(ask_gemini(quiz_prompt))

    if not isinstance(quiz_data, dict):
        raise ValueError("Invalid quiz response.")

    questions = quiz_data.get("questions")
    validate_questions(questions)

    progress.progress(1.0)
    status.write("✅ Your study material is ready!")

    return {
        "summary": all_summary_points,
        "key_points": all_key_points,
        "questions": questions
    }


# =====================================================
# 6. GENERATE A FRESH QUIZ
# =====================================================

def generate_new_quiz(study_data, previous_questions):
    """Generate 10 new questions without intentionally repeating old ones."""

    study_reference = {
        "summary": study_data["summary"],
        "key_points": study_data["key_points"]
    }

    previous_question_text = [
        question["question"]
        for question in previous_questions
    ]

    prompt = """
You are an expert academic quiz creator.

Generate exactly 10 NEW multiple-choice questions.

REQUIREMENTS:

1. Use only the supplied study reference.
2. Generate exactly 10 questions.
3. Cover different topics and subtopics.
4. Do not repeat previous questions.
5. Do not simply reword previous questions.
6. Test different concepts, facts and definitions.
7. Include reasoning-based questions where appropriate.
8. Give four distinct options per question.
9. Only one option must be correct.
10. The answer must exactly match one option.
11. Include an explanation for every answer.
12. Do not invent facts.

Return ONLY valid JSON:

{
    "questions": [
        {
            "question": "New question?",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": "Option A",
            "explanation": "Explanation"
        }
    ]
}

Return exactly 10 new questions.

PREVIOUS QUESTIONS - DO NOT REPEAT:

""" + json.dumps(previous_question_text, ensure_ascii=False) + """

STUDY REFERENCE:

""" + json.dumps(study_reference, ensure_ascii=False)

    result = parse_json_response(ask_gemini(prompt))

    if not isinstance(result, dict):
        raise ValueError("Invalid new quiz response.")

    questions = result.get("questions")
    validate_questions(questions)

    return questions


# =====================================================
# 7. PREMIUM APPLICATION DESIGN
# =====================================================

st.markdown(
    textwrap.dedent("""
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99,102,241,0.15),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #0b1020,
                #111a33,
                #17132b
            );
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    h2, h3 {
        color: #e2e8f0 !important;
    }

    p, label {
        color: #cbd5e1;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #9333ea);
        color: white;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 0.7rem 1.3rem;
        font-weight: 700;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        color: white;
        border-color: #c4b5fd;
        transform: translateY(-2px);
    }

    .stTextArea textarea {
        background: #151d33 !important;
        color: white !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(99,102,241,0.08);
        border: 1px dashed #818cf8;
        border-radius: 14px;
        padding: 15px;
    }

    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #c084fc);
    }

    footer {
        visibility: hidden;
    }
    </style>
    """),
    unsafe_allow_html=True
)


# =====================================================
# 8. APPLICATION HEADER
# =====================================================

header_html = textwrap.dedent("""
<div style="
    background: linear-gradient(120deg, #25205a, #312e81, #4c1d95);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 22px;
    padding: 35px;
    margin: 15px 0 30px 0;
">
    <div style="font-size:48px;">📚</div>

    <div style="
        font-size:42px;
        font-weight:800;
        color:white;
        margin-top:5px;
    ">
        Yamosh <span style="color:#c4b5fd;">AI</span>
    </div>

    <div style="
        font-size:19px;
        font-weight:600;
        color:#c4b5fd;
        margin-top:8px;
    ">
        Your Personal AI Study Assistant
    </div>

    <div style="
        font-size:15px;
        color:#e2e8f0;
        margin-top:10px;
    ">
        Transform your study material into comprehensive summaries,
        important key points and practice quizzes.
    </div>
</div>
""")

# Use st.html so HTML is rendered, not displayed as code.
st.html(header_html)


# =====================================================
# 9. STUDY MATERIAL INPUT
# =====================================================

st.header("📖 1. Upload or Paste Study Material")

input_method = st.radio(
    "Choose your input method:",
    ["Paste Notes", "Upload PDF"],
    horizontal=True,
    key="input_method"
)

notes = ""

if input_method == "Paste Notes":
    notes = st.text_area(
        "Enter your study notes:",
        height=250,
        placeholder="Paste your complete study material here...",
        key="study_notes_input"
    )

else:
    uploaded_file = st.file_uploader(
        "Upload your PDF document",
        type=["pdf"],
        key="study_pdf"
    )

    if uploaded_file is not None:
        try:
            pdf_reader = PdfReader(uploaded_file)
            extracted_pages = []

            for page_number, page in enumerate(
                pdf_reader.pages,
                start=1
            ):
                page_text = page.extract_text() or ""

                if page_text.strip():
                    extracted_pages.append(
                        f"\n--- Page {page_number} ---\n{page_text}"
                    )

            notes = "\n".join(extracted_pages)

            if notes.strip():
                st.success(
                    f"PDF uploaded successfully! "
                    f"{len(pdf_reader.pages)} pages detected."
                )

                with st.expander("👁️ Preview Extracted PDF Text"):
                    st.text(notes)

            else:
                st.warning(
                    "No readable text was found. "
                    "This PDF may contain scanned images. "
                    "OCR is required for scanned PDFs."
                )

        except Exception as error:
            st.error(f"PDF reading failed: {error}")


# =====================================================
# 10. GENERATE STUDY MATERIAL
# =====================================================

if st.button(
    "✨ Generate Complete Study Material",
    type="primary",
    use_container_width=True
):
    if not notes.strip():
        st.warning(
            "Please enter your study notes or upload a readable PDF."
        )

    else:
        try:
            with st.spinner(
                "Yamosh AI is preparing your study material..."
            ):
                new_data = generate_study_content(notes)

                st.session_state["study_data"] = new_data
                st.session_state["source_notes"] = notes

                st.session_state["quiz_submitted"] = False
                st.session_state["submitted_answers"] = []

                # A new attempt creates completely new radio widget keys.
                st.session_state["quiz_attempt"] += 1

                st.success(
                    "Your complete study material is ready!"
                )

                st.rerun()

        except Exception as error:
            st.error(
                f"Study material generation failed: {error}"
            )


# =====================================================
# 11. DISPLAY SUMMARY AND KEY POINTS
# =====================================================

data = st.session_state.get("study_data")

if data is not None:
    st.divider()

    st.header("📝 2. Complete Study Summary")

    st.caption(
        f"📌 {len(data['summary'])} summary points generated"
    )

    for index, point in enumerate(data["summary"], start=1):
        st.markdown(f"**{index}.** {point}")

    st.divider()

    st.header("⭐ 3. All Important Key Points")

    st.caption(
        f"📌 {len(data['key_points'])} important key points identified"
    )

    for index, point in enumerate(data["key_points"], start=1):
        with st.expander(f"🔹 {index}. {point['title']}"):
            st.write(point["detail"])


# =====================================================
# 12. QUIZ SECTION
# =====================================================

st.divider()

st.header("🧠 4. Test Your Knowledge")

st.write(
    "Challenge yourself with 10 questions based on your study material."
)

if data is None:
    st.info(
        "📚 Generate your study material first to unlock the quiz."
    )

else:
    questions = data["questions"]
    total_questions = len(questions)
    attempt = st.session_state["quiz_attempt"]

    st.caption(
        f"Quiz Attempt {attempt} | {total_questions} Questions"
    )

    # Display each question with a unique key for this attempt.
    for index, question in enumerate(questions):
        st.markdown(
            f"### Question {index + 1} of {total_questions}"
        )

        st.radio(
            question["question"],
            question["options"],
            index=None,
            key=f"quiz_attempt_{attempt}_question_{index}",
            disabled=st.session_state["quiz_submitted"]
        )

    # Read selected answers from the current quiz attempt.
    current_answers = [
        st.session_state.get(
            f"quiz_attempt_{attempt}_question_{index}"
        )
        for index in range(total_questions)
    ]

    answered_count = sum(
        answer is not None
        for answer in current_answers
    )

    st.progress(answered_count / total_questions)

    st.caption(
        f"✍️ {answered_count}/{total_questions} questions answered"
    )

    # Submit quiz.
    if not st.session_state["quiz_submitted"]:
        if st.button(
            "🚀 Submit My Quiz",
            type="primary",
            use_container_width=True
        ):
            if answered_count != total_questions:
                st.warning(
                    "Please answer all 10 questions before submitting."
                )

            else:
                st.session_state["submitted_answers"] = (
                    current_answers.copy()
                )

                st.session_state["quiz_submitted"] = True
                st.rerun()

    # =================================================
    # 13. QUIZ RESULTS
    # =================================================

    if st.session_state["quiz_submitted"]:
        submitted_answers = st.session_state["submitted_answers"]

        score = sum(
            1
            for index, question in enumerate(questions)
            if submitted_answers[index] == question["answer"]
        )

        percentage = (score / total_questions) * 100

        if percentage >= 80:
            result_message = "Excellent work! Outstanding performance."
        elif percentage >= 60:
            result_message = "Good job! Keep improving your knowledge."
        elif percentage >= 40:
            result_message = "Nice effort! A little more revision will help."
        else:
            result_message = (
                "Keep practising! Review the explanations below."
            )

        st.divider()
        st.header("🏆 Your Quiz Results")

        # Render the score card as HTML directly.
        # This prevents Streamlit from displaying HTML tags as code.
        score_html = textwrap.dedent(f"""
        <div style="
            background: linear-gradient(120deg, #312e81, #6d28d9, #7c3aed);
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 22px;
            padding: 35px 20px;
            text-align: center;
            color: white;
            margin: 20px 0;
            box-shadow: 0 15px 45px rgba(76,29,149,0.25);
        ">
            <div style="
                color: white;
                font-size: 26px;
                font-weight: 700;
                margin-bottom: 12px;
            ">
                🏆 Your Final Score
            </div>

            <div style="
                color: white;
                font-size: 64px;
                font-weight: 800;
                line-height: 1.2;
                margin: 10px 0;
            ">
                {score}/{total_questions}
            </div>

            <div style="
                color: #ede9fe;
                font-size: 22px;
                font-weight: 600;
            ">
                {percentage:.0f}% Accuracy
            </div>

            <div style="
                color: #ddd6fe;
                font-size: 15px;
                margin-top: 12px;
            ">
                {result_message}
            </div>
        </div>
        """)

        st.html(score_html)

        # Score metrics.
        col1, col2, col3 = st.columns(3)

        col1.metric("✅ Correct", score)
        col2.metric("❌ Incorrect", total_questions - score)
        col3.metric("📊 Accuracy", f"{percentage:.0f}%")

        # Detailed answer review.
        st.divider()
        st.subheader("📋 Detailed Answer Review")

        for index, question in enumerate(questions):
            user_answer = submitted_answers[index]
            correct_answer = question["answer"]

            with st.expander(
                f"Question {index + 1}: {question['question']}"
            ):
                st.write(f"**Your answer:** {user_answer}")
                st.write(f"**Correct answer:** {correct_answer}")

                if user_answer == correct_answer:
                    st.success("Correct answer! Well done.")
                else:
                    st.error(
                        "Incorrect answer. Review the explanation."
                    )

                st.markdown("**💡 Explanation**")
                st.info(question["explanation"])

        # Generate a fresh quiz.
        st.divider()
        st.subheader("🔄 Practice Again")

        st.write(
            "Generate a fresh set of 10 questions "
            "from the same study material."
        )

        if st.button(
            "✨ Generate New Quiz",
            type="primary",
            use_container_width=True
        ):
            try:
                with st.spinner(
                    "Gemini is creating 10 fresh questions..."
                ):
                    new_questions = generate_new_quiz(
                        data,
                        questions
                    )

                    data["questions"] = new_questions
                    st.session_state["study_data"] = data

                    # Increment attempt to create new radio keys.
                    st.session_state["quiz_attempt"] += 1

                    st.session_state["quiz_submitted"] = False
                    st.session_state["submitted_answers"] = []

                    st.rerun()

            except Exception as error:
                st.error(
                    f"Could not generate a new quiz: {error}"
                )


# =====================================================
# 14. DOWNLOAD REVISION PACK
# =====================================================

if data is not None:
    st.divider()

    st.header("📥 5. Download Complete Revision Pack")

    revision_pack = (
        "YAMOSH AI - COMPLETE REVISION PACK\n"
        "====================================\n\n"
    )

    revision_pack += "COMPLETE STUDY SUMMARY\n"
    revision_pack += "----------------------\n\n"

    for index, point in enumerate(data["summary"], start=1):
        revision_pack += f"{index}. {point}\n\n"

    revision_pack += "\nALL IMPORTANT KEY POINTS\n"
    revision_pack += "-----------------------\n\n"

    for index, point in enumerate(data["key_points"], start=1):
        revision_pack += (
            f"{index}. {point['title']}\n"
            f"{point['detail']}\n\n"
        )

    revision_pack += "\nPRACTICE QUIZ\n"
    revision_pack += "-------------\n\n"

    for index, question in enumerate(data["questions"], start=1):
        revision_pack += f"{index}. {question['question']}\n"

        for option_index, option in enumerate(question["options"]):
            revision_pack += (
                f"   {chr(65 + option_index)}. {option}\n"
            )

        revision_pack += (
            f"\nCorrect Answer: {question['answer']}\n"
        )

        revision_pack += (
            f"Explanation: {question['explanation']}\n\n"
        )

    st.download_button(
        label="📥 Download My Complete Revision Pack",
        data=revision_pack,
        file_name="Yamosh_AI_Complete_Revision.txt",
        mime="text/plain",
        use_container_width=True
    )


# =====================================================
# 15. FOOTER
# =====================================================

st.divider()

footer_html = textwrap.dedent("""
<div style="
    text-align: center;
    padding: 20px 0;
    color: #94a3b8;
    font-size: 13px;
">
    <div style="font-size: 22px;">📚</div>

    <b style="color: #c4b5fd;">Yamosh AI</b>

    <div>
        Your Personal AI Study Assistant
    </div>

    <div style="margin-top: 8px;">
        Powered by Google Gemini
    </div>
</div>
""")

st.html(footer_html)
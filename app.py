import streamlit as st
import pandas as pd
import os
from pathlib import Path
from openai import OpenAI
import json


# Custom styling
st.markdown("""
    <style>
    /* Global background color */
    .stApp {
        background-color: #f0f7f7 !important;
    }

    /* Main content container */
    .main .block-container {
        background-color: #f0f7f7 !important;
    }

    /* Ensure background color on mobile */
    @media (max-width: 768px) {
        .main .block-container {
            background-color: #f0f7f7 !important;
            padding: 1rem !important;
        }
        
        .stApp {
            background-color: #f0f7f7 !important;
        }
    }

    /* Make the title more prominent */
    .title {
        font-size: 42px !important;
        font-weight: 600 !important;
        color: #03707d !important;
        padding-bottom: 1rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1) !important;
    }
    
    /* Style the header sections */
    .section-header {
        padding-top: 1rem !important;
        font-size: 24px !important;
        font-weight: 500 !important;
        color: #03707d !important;
    }
    
    /* Style the buttons */
    .stButton>button {
        background-color: #03707d !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        border: none !important;
        transition: all 0.2s !important;
    }
    
    .stButton>button:hover {
        background-color: #27bec2 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    /* Style the upload box */
    .uploadedFile {
        background-color: #ffffff !important;
        border: 2px dashed #27bec2 !important;
        border-radius: 6px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }

    .stUploadedFileMsg {
        color: #03707d !important;
    }
    
    /* Style download buttons in sidebar differently */
    .sidebar .stButton>button {
        background-color: #27bec2 !important;
    }
    
    .sidebar .stButton>button:hover {
        background-color: #03707d !important;
    }
    
    /* Style for expandable sections and specifically target the label */
    .streamlit-expanderHeader {
        background-color: #ffffff !important;
        border-radius: 6px !important;
        border: 1px solid #27bec2 !important;
    }

    /* Target the expander label and force color */
    .streamlit-expanderHeader label {
        color: #03707d !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }

    /* Handle hover state */
    .streamlit-expanderHeader:hover label {
        color: #27bec2 !important;
    }

    /* Style the text areas */
    .stTextArea textarea {
        border-radius: 6px !important;
        border-color: #27bec2 !important;
        background-color: #ffffff !important;
        color: #03707d !important;
    }
    
    /* Style for text selection in text areas */
    .stTextArea textarea::selection {
        background-color: #a7e7e1 !important;
        color: #03707d !important;
    }

    /* Style the selectboxes for better readability */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border-radius: 6px !important;
        color: #03707d !important;
        border-color: #27bec2 !important;
    }

    .stSelectbox > div > div > div {
        color: #03707d !important;
    }

    /* Style the dropdown options */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #03707d !important;
    }

    /* File uploader styling */
    .stFileUploader {
        background-color: #ffffff !important;
        padding: 1rem !important;
        border-radius: 6px !important;
        border: 2px dashed #27bec2 !important;
    }

    /* Add some padding to the sidebar */
    .sidebar .block-container {
        padding: 2rem 1rem !important;
        background-color: #e8e6e4 !important;
    }

    /* Success message styling */
    .element-container .stSuccess {
        background-color: #a7e7e1 !important;
        color: #03707d !important;
        border-color: #27bec2 !important;
    }

    /* Warning message styling */
    .element-container .stWarning {
        background-color: #ffffff !important;
        color: #03707d !important;
        border-color: #27bec2 !important;
    }

    /* Error message styling */
    .element-container .stError {
        background-color: #ffffff !important;
        color: #03707d !important;
        border-color: #27bec2 !important;
    }
    </style>
""", unsafe_allow_html=True)

def setup_openai():
    """Configure OpenAI API using Streamlit secrets"""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.session_state['OPENAI_API_KEY'] = api_key
        return True
    except Exception as e:
        st.error("Error loading OpenAI API key from secrets. Please ensure it's properly configured in .streamlit/secrets.toml")
        return False

def setup_directories():
    """Create the necessary directories if they don't exist"""
    base_dir = Path(__file__).parent
    input_dir = base_dir / "data" / "input"
    gold_dir = base_dir / "data" / "gold_standard"
    
    input_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    
    return input_dir, gold_dir

def load_csv_files(directory):
    """Load all CSV files from the specified directory"""
    csv_files = []
    directory = Path(directory)
    if directory.exists():
        for file in directory.glob('*.csv'):
            csv_files.append(file.name)
    return csv_files

def load_pdf(filename):
    """Load PDF file from data directory"""
    try:
        pdf_path = Path(__file__).parent / "data" / filename
        return pdf_path.read_bytes()
    except Exception as e:
        return b"Error loading PDF"

def call_openai_api(prompt):
    """Call OpenAI API with the generated prompt"""
    try:
        client = OpenAI(api_key=st.session_state['OPENAI_API_KEY'])
        
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {"role": "system", "content": "You are a medical education assistant helping students review and learn from their exam results. Provide constructive, supportive feedback and specific recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"

def analyse_mcq_answers(input_file, gold_standard_file):
    """Analyse multiple choice answers against the gold standard"""
    try:
        # Load the files
        input_path = input_dir / input_file
        gold_path = gold_dir / gold_standard_file
        
        # Read CSVs with explicit handling of whitespace in headers
        student_df = pd.read_csv(input_path, skipinitialspace=True)
        gold_df = pd.read_csv(gold_path, skipinitialspace=True)
        
        # Clean column names by removing any extra whitespace
        student_df.columns = student_df.columns.str.strip()
        gold_df.columns = gold_df.columns.str.strip()
        
        # Merge student answers with gold standard
        merged_df = student_df.merge(gold_df, on='question_number', how='left')
        
        # Calculate overall score
        total_questions = len(merged_df)
        correct_answers = (merged_df['answer_student'] == merged_df['answer_correct']).sum()
        score_percentage = (correct_answers / total_questions) * 100
        
        # Identify incorrect answers and their topics
        incorrect_df = merged_df[merged_df['answer_student'] != merged_df['answer_correct']]
        
        # Get unique topic-subtopic pairs and sort by topic
        topic_summary = incorrect_df[['topic', 'subtopic']].drop_duplicates()
        topic_summary = topic_summary.sort_values(['topic', 'subtopic'])
        
        # Group by main topic
        grouped_topics = {}
        for _, row in topic_summary.iterrows():
            if row['topic'] not in grouped_topics:
                grouped_topics[row['topic']] = []
            grouped_topics[row['topic']].append(row['subtopic'])
        
        # Prepare the analysis result
        analysis_result = f"""
Overall Performance Summary:
--------------------------
Total Questions: {total_questions}
Correct Answers: {correct_answers}
Overall Score: {score_percentage:.1f}%

Topics for Review:
----------------"""
        
        # Add grouped topics
        for topic in sorted(grouped_topics.keys()):
            analysis_result += f"\n• {topic}"
            for subtopic in sorted(grouped_topics[topic]):
                analysis_result += f"\n  - {subtopic}"
            analysis_result += "\n"
        
        # Store detailed results and analysis result in session state
        st.session_state['detailed_results'] = {
            'score': score_percentage,
            'grouped_topics': grouped_topics,
            'total_questions': total_questions,
            'correct_answers': correct_answers
        }
        st.session_state['analysis_result'] = analysis_result
        
        return analysis_result
    
    except Exception as e:
        error_message = f"Error during analysis: {str(e)}\nPlease check your CSV files match these formats:\n\nStudent CSV:\nquestion_number,answer_student\n1,A\n2,B\n\nModel Answers CSV:\nquestion_number,answer_correct,topic,subtopic\n1,E,Urology,Urethral Stricture"
        st.session_state['analysis_result'] = error_message
        return error_message

def generate_feedback(analysis_result):
    """Generate LLM prompt and get OpenAI response"""
    if 'detailed_results' not in st.session_state:
        return "Please run analysis first."
    
    results = st.session_state['detailed_results']
    
    topics_text = "Topics for Review:\n"
    for topic in sorted(results['grouped_topics'].keys()):
        topics_text += f"\n• {topic}"
        for subtopic in sorted(results['grouped_topics'][topic]):
            topics_text += f"\n  - {subtopic}"

    prompt = f"""The following is a list of subjects and subtopics that a medical student has answered incorrectly in a recent exam.

{topics_text}

Provide a supportive and constructive revision plan for the student to assist in learning these topics. You have access to a RAG/vector database. This contains lecture notes, slides, tutorial information and case studies produced by the medical school. It is being built at present so pretend these files exist and produce example links to relevant resources for the subjects. Do not overwhelm the student with information. Keep your answer succinct, accurate and detailed."""

    return prompt, call_openai_api(prompt)

# Set up directories
input_dir, gold_dir = setup_directories()

# Main app interface with styled title
st.markdown('<p class="title">Medical MCQ Performance & Study Tool</p>', unsafe_allow_html=True)

# Check for OpenAI API configuration
if not setup_openai():
    st.warning("Please configure your OpenAI API key in the sidebar to enable feedback generation.")
    st.stop()

# Upload section in expander
with st.expander("Upload New Files", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="section-header">Upload Student Answers</p>', unsafe_allow_html=True)
        input_upload = st.file_uploader("Upload student answers CSV", type=['csv'], key="input_upload")
        if input_upload:
            with open(os.path.join(input_dir, input_upload.name), "wb") as f:
                f.write(input_upload.getbuffer())
            st.success(f"Saved {input_upload.name}")

    with col2:
        st.markdown('<p class="section-header">Upload Model Answers</p>', unsafe_allow_html=True)
        gold_upload = st.file_uploader("Upload model answers CSV", type=['csv'], key="gold_upload")
        if gold_upload:
            with open(os.path.join(gold_dir, gold_upload.name), "wb") as f:
                f.write(gold_upload.getbuffer())
            st.success(f"Saved {gold_upload.name}")

# File selection section
st.markdown('<p class="section-header">File Selection</p>', unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    input_files = load_csv_files(input_dir)
    selected_input = st.selectbox(
        "Select student answers file",
        input_files,
        key="input_select"
    )

with col4:
    gold_files = load_csv_files(gold_dir)
    selected_gold = st.selectbox(
        "Select model answers file",
        gold_files,
        key="gold_select"
    )

# Analysis section
st.markdown('<p class="section-header">Performance Summary</p>', unsafe_allow_html=True)
if st.button("Analyse"):
    if selected_input and selected_gold:
        analysis_result = analyse_mcq_answers(selected_input, selected_gold)
        st.session_state['analysis_result'] = analysis_result
    else:
        st.error("Please select both student answers and model answers files")

# Display analysis results if they exist in session state
if 'analysis_result' in st.session_state:
    st.text_area("Analysis Results", st.session_state['analysis_result'], height=300)

# Generation section
st.markdown('<p class="section-header">AI Feedback Generation</p>', unsafe_allow_html=True)
if st.button("Generate Feedback"):
    if 'detailed_results' in st.session_state:
        with st.spinner('Generating personalised feedback...'):
            prompt, ai_response = generate_feedback(None)
            
            tab1, tab2 = st.tabs(["AI Response", "Generated Prompt"])
            
            with tab1:
                st.text_area("Personalised Feedback", ai_response, height=400)
            
            with tab2:
                st.text_area("LLM Prompt", prompt, height=300)
    else:
        st.warning("Please run analysis first")

# Sidebar content
st.sidebar.markdown('<p class="section-header">Medical MCQ Performance & Study Tool</p>', unsafe_allow_html=True)
st.sidebar.markdown("""
Welcome to the Medical School's MCQ Analysis and Learning Support Tool. This application helps you review your exam performance and creates personalised study recommendations using the medical school's comprehensive learning resources.

**Getting Started:**
1. To download a copy of the exam use the links below:
""")

# Add download buttons for both exam PDFs
exam_pdf = load_pdf("Practice-paper-1.pdf")
answers_pdf = load_pdf("Practice-paper-1-answers.pdf")

col1, col2 = st.sidebar.columns(2)

with col1:
    st.download_button(
        label="📄 Exam Paper",
        data=exam_pdf,
        file_name="Practice-paper-1.pdf",
        mime="application/pdf"
    )

with col2:
    st.download_button(
        label="📄 With Answers",
        data=answers_pdf,
        file_name="Practice-paper-1-answers.pdf",
        mime="application/pdf"
    )

st.sidebar.markdown("""
2. Select your answer sheet from the dropdown menu
3. Click 'Analyse' to see your performance summary
4. Use 'Generate Feedback' to receive a personalised learning plan with links to relevant medical school resources

Our AI-powered system will analyse your answers and create targeted recommendations based on the medical school's lecture materials, clinical cases, and study resources.""")

st.markdown("---")
st.markdown("*Medical MCQ Performance & Study Tool v2.0*")

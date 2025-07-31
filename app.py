import streamlit as st
import pandas as pd
import requests
import json
import time
import docx2txt
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import random

# ---- Custom CSS (minimal Perplexity polish, not overstyled) ----
# ---- Custom CSS for Perplexity-inspired look ----
st.markdown("""
    <style>
    body {
        background-color: #f9fafb;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700;
        margin-bottom: 0.75rem;
        text-align: center;
    }
    .main-area {
        background: #fff;
        padding: 1.5rem 2.5rem;
        border-radius: 18px;
        box-shadow: 0 2px 18px rgba(0,0,0,0.08);
        margin-top: 1.5rem;
    }
    .stTextArea, .stTextInput, .stSelectbox, .stButton, .stFileUploader {
        font-size: 1.1rem !important;
    }
    .stSelectbox label, .stTextArea label, .stTextInput label, .stFileUploader label {
        font-size: 1.05rem !important;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="AKAMAI TEST GEN", layout="wide")
st.markdown('<div class="main-header">🤖 AKAMAI Test Gen', unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1, 2.3, 1])

with col_center:
    st.markdown('<div class="main-area">', unsafe_allow_html=True)

MAX_FILE_SIZE_MB = 5
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 5

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")
model_embed = load_embedder()

def generate_test_case_prompt(
    df,
    specific_user_story=None,
    user_story_col=None,
    func_req_col=None
):
    def normalize_col(col):
        return ''.join(e for e in col.lower().strip() if e.isalnum())
    def find_col(possible_names, df):
        norm_cols = {normalize_col(c): c for c in df.columns}
        for name in possible_names:
            norm_name = normalize_col(name)
            if norm_name in norm_cols:
                return norm_cols[norm_name]
        return None
    if user_story_col is None:
        user_story_col = find_col(
            ["user story", "userstory", "story", "usserstories", "use case", "business requirement/userstory"],
            df
        )
    if func_req_col is None:
        func_req_col = find_col([
            "functional requirement",
            "functional requirements",
            "function requirement",
            "function requirements",
            "requirement",
            "requirements",
            "funcreq",
            "businessrequirements",
            "test case"
        ], df)
    user_stories = {}
    current_story = None
    if user_story_col and func_req_col:
        for _, row in df.iterrows():
            user_story = str(row.get(user_story_col, '')).strip()
            func_req = str(row.get(func_req_col, '')).strip()
            if user_story and user_story.lower() != 'nan':
                current_story = user_story
                user_stories.setdefault(current_story, [])
            if func_req and func_req.lower() != 'nan' and current_story:
                reqs = [r.strip() for r in func_req.split('\n') if r.strip()]
                user_stories[current_story].extend(reqs)
        user_stories = {k: v for k, v in user_stories.items() if k and v}
        if not user_stories:
            user_stories = {"Default User Story": ["No valid data found in the specified columns."]}
    else:
        user_stories = {"Default User Story": ["Either User Story or Functional Requirement column not specified or found."]}
    if not user_stories:
        user_stories = {"Default User Story": ["No functional requirements or user stories found in the input file."]}
    if specific_user_story and specific_user_story in user_stories:
        selected_story = specific_user_story
    else:
        selected_story = random.choice(list(user_stories.keys()))
    prompt = [
        "QUERY: Hi, Can you please generate the test cases for the below requirement IN THE FORM OF A TABLE:\n",
        f"User Story: {selected_story}",
        "Functional Requirements:"
    ]
    for i, req in enumerate(user_stories[selected_story], 1):
        prompt.append(f"{i}. {req}")
    return selected_story, "\n".join(prompt), list(user_stories.keys())

def split_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.create_documents([text])
    return [doc.page_content for doc in docs]

def read_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        return ''.join([page.extract_text() or "" for page in reader.pages])
    elif uploaded_file.name.endswith(".docx"):
        return docx2txt.process(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".csv"):
        return ""
    return ""

def build_faiss_index(chunks, embedder):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve_context(query, chunks, embedder, index, top_k=TOP_K):
    query_embedding = embedder.encode([query])[0]
    D, I = index.search(np.array([query_embedding]), k=top_k)
    top_chunks = [chunks[i] for i in I[0] if i >= 0]
    return top_chunks

def construct_prompt(query, retrieved_chunks):
    context_block = ""
    for idx, chunk in enumerate(retrieved_chunks, 1):
        context_block += f"\n-- Context {idx} --\n{chunk}\n"
    instruction = (
        "You are a highly skilled QA developer. Use only the provided context sections below to answer the question. "
        "If you do not find the answer in the context, say 'Not available from context.'\n"
        "Provide your answer in a clear and structured manner.\n"
    )
    return f"{instruction}{context_block}\n-- Question --\n{query.strip()}\n-- End --\n"

# ======== UPLOAD & PROCESS FILE ========
uploaded_file = st.file_uploader("📤 Upload requirements (.csv, .pdf, .docx, .xlsx)", type=["pdf", "docx", "xlsx", "csv"])
file_chunks, chunk_sources, index = [], [], None
xlsx_prompt, story_prompt_text = None, ""
user_story_col, func_req_col = "-- Select --", "-- Select --"  # Initialize with default
df = None

if uploaded_file:
    file_size_MB = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_MB > MAX_FILE_SIZE_MB:
        st.error(f"❌ File exceeds {MAX_FILE_SIZE_MB} MB. Upload a smaller file.")
    else:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file, sheet_name=0)
            elif uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = None
            if df is not None:
                df = df.dropna(how='all').fillna('')
                st.write("### Preview of uploaded data (top 5 rows):")
                st.write("DEBUG: Columns detected in uploaded file:", df.columns.tolist())
                st.dataframe(df.head())
                columns = df.columns.tolist()
                user_story_col = st.selectbox("📘 Select the column for User Story:", options=["-- Select --"] + columns)
                func_req_col = st.selectbox("📄 Select the column for Functional Requirement:", options=["-- Select --"] + columns)
                if user_story_col != "-- Select --" and func_req_col != "-- Select --":
                    selected_story, xlsx_prompt, available_stories = generate_test_case_prompt(
                        df,
                        user_story_col=user_story_col,
                        func_req_col=func_req_col
                    )
                    selected = st.selectbox("📌 Select a User Story (or Random):", ["Random"] + available_stories)
                    if selected != "Random":
                        selected_story, xlsx_prompt, _ = generate_test_case_prompt(
                            df,
                            user_story_col=user_story_col,
                            func_req_col=func_req_col,
                            specific_user_story=selected
                        )
                    else:
                        selected_story = random.choice(available_stories)
                    st.success(f"✅ Generated Prompt for: {selected_story}")
                    st.code(xlsx_prompt)
                else:
                    st.info("Please select both User Story and Functional Requirement columns to generate the prompt.")
            else:
                # Handle PDF / DOCX for RAG
                file_text = read_uploaded_file(uploaded_file)
                if file_text:
                    file_chunks = split_into_chunks(file_text)
                    chunk_sources = file_chunks
                    index, _ = build_faiss_index(file_chunks, model_embed)
        except Exception as e:
            st.error(f"❌ Error reading file:\n{e}")

st.markdown("---")

prompt_input = st.text_area(
    "📥 Enter a custom question or prompt manually:",
    value=xlsx_prompt if xlsx_prompt else "",
    height=180,
    placeholder="Type your test generation prompt here..."
    )


model_map = {
    "Demo Table Only": None,
    "LLaMA2 7B": "llama2-uncensored",
    "Phi-4 Mini": "phi4-mini",
    "Mistral 7B": "mistral",
    "LLaMA3.3 70B": "llama3.3",
    "smollama": "smollm2:1.7b"
}
model_choice = st.selectbox("🧠 Choose a model to respond:", list(model_map.keys()))

run_clicked = st.button("🚀 RUN")

if run_clicked and (prompt_input or xlsx_prompt or file_chunks):
    st.subheader(f"📤 Output from: {model_choice}")
    if xlsx_prompt:
        final_prompt = xlsx_prompt
    elif file_chunks and prompt_input and index:
        top_chunks = retrieve_context(prompt_input, chunk_sources, model_embed, index)
        final_prompt = construct_prompt(prompt_input, top_chunks)
    else:
        final_prompt = prompt_input.strip()
    with st.spinner("⏳ Generating response..."):
        if model_map[model_choice] is None:
            dummy_data = pd.DataFrame({
                "Step": ["Login", "Create Ticket"],
                "Pre Condition": ["User should have valid credentials", "Environment Setup is done"],
                "Expected Result": ["Success", "Ticket Created"]
            })
            st.write("Here's a dummy response:")
            st.table(dummy_data)
        else:
            try:
                start_time = time.time()
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model_map[model_choice], "prompt": final_prompt},
                    stream=True,
                    timeout=60,
                )
                full_response = ""
                output_placeholder = st.empty()
                for line in response.iter_lines():
                    if line:
                        data = line.decode("utf-8")
                        json_data = json.loads(data)
                        text_chunk = json_data.get("response", "")
                        full_response += text_chunk
                        output_placeholder.markdown(f"**Ollama Response:**\n\n{full_response}")
                        if json_data.get("done"):
                            break
                duration = time.time() - start_time
                st.success(f"✅ Response generated successfully in {duration:.2f} seconds!")
            except Exception as e:
                st.error(f"❌ Failed to get response from Ollama API: {e}")
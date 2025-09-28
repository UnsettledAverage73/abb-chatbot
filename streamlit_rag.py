import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics import accuracy_score
import difflib


# Function to calculate similarity between two strings
def string_similarity(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def evaluate_chatbot(eval_df, qa_pipeline):
    results = []
    for _, row in eval_df.iterrows():
        question = row["question"]
        true_answer = row["answer"]
        pred_answer = chatbot_query(question, qa_pipeline)
        similarity = string_similarity(pred_answer, true_answer)
        results.append(similarity)
    return sum(results) / len(results)  # average similarity (0-1)
# -----------------------
# File paths (update these before running)
# -----------------------
file1 = "3439_RMU_StockObso_2507A-1(1).csv"
file2 = "sustainbility2(1).csv"

# -----------------------
# 1. Data Loading and Preprocessing
# -----------------------
@st.cache_resource
def load_and_preprocess_data():
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    # Combine df1
    df1_text_columns = ['Manual', 'Jan 2025', 'Dec 2024']
    df1['combined_text'] = df1[df1_text_columns].astype(str).agg(' '.join, axis=1)

    # Combine df2
    df2_text_columns = [
        'Date', 'Plastic Reduced (kg)', 'Cumulative (kg)',
        'Daily Plastic Consumption (kg)', 'Wood Consumption (kg)',
        'Energy Consumption (kWh)', 'E-Waste (kg)', 'SF6 Consumption (kg)',
        'Argon Consumption (kg)', 'Helium Consumption (kg)',
        'CO2 Emission (kg)', 'Hazardous Waste (kg)'
    ]
    existing_df2_text_columns = [col for col in df2_text_columns if col in df2.columns]

    def format_row_for_llm(row):
        return ", ".join([f"{col}: {row[col]}" for col in existing_df2_text_columns])

    df2['combined_text'] = df2.apply(format_row_for_llm, axis=1)
    
    return df1, df2

# -----------------------
# 2. Embedding Generation
# -----------------------
@st.cache_resource
def generate_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = list(model.encode(df['combined_text'].tolist()))
    df['embeddings'] = embeddings
    return df, model

# -----------------------
# 3. FAISS Index
# -----------------------
def create_faiss_index(df):
    embeddings_array = np.array(df['embeddings'].tolist()).astype('float32')
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    return index

# -----------------------
# 4. Retrieval
# -----------------------
def retrieve_relevant_data(query_text, model, index, df, k=3):
    query_embedding = model.encode([query_text]).astype('float32')
    D, I = index.search(query_embedding, k=k)
    retrieved_contexts = [df.loc[idx, 'combined_text'] for idx in I[0]]
    return " ".join(retrieved_contexts)

# -----------------------
# 5. Generation (LLM)
# -----------------------
@st.cache_resource
def get_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def generate_answer(question, context, qa_pipeline):
    llm_response = qa_pipeline(question=question, context=context)
    return llm_response['answer']

# -----------------------
# 6. Chatbot Query
# -----------------------
def chatbot_query(question, qa_pipeline):
    df1, df2 = load_and_preprocess_data()
    df2, embedding_model = generate_embeddings(df2)
    faiss_index = create_faiss_index(df2)

    context = retrieve_relevant_data(question, embedding_model, faiss_index, df2)
    answer = generate_answer(question, context, qa_pipeline)
    return answer

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG-based Chatbot")
st.write("Ask me anything about the sustainability or stock data!")

user_question = st.text_input("Enter your question:")
qa_pipeline = get_qa_pipeline()

if st.button("Ask") and user_question.strip() != "":
    with st.spinner("Thinking..."):
        answer = chatbot_query(user_question, qa_pipeline)
    st.success(f"**Answer:** {answer}")

st.subheader("📊 Evaluate Chatbot Accuracy")

if st.button("Run Evaluation"):
    eval_df = pd.read_csv("eval_data.csv")
    accuracy = evaluate_chatbot(eval_df, qa_pipeline)
    st.metric("Chatbot Accuracy", f"{accuracy*100:.2f}%")


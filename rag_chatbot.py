import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# File paths
file1 = "/home/unsettledaverage73/ABB-chatbot/abb-chatbot/3439_RMU_StockObso_2507A-1(1).csv"
file2 = "/home/unsettledaverage73/ABB-chatbot/abb-chatbot/sustainbility2(1).csv"

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data():
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    # Prepare combined text for df1
    df1_text_columns = ['Manual', 'Jan 2025', 'Dec 2024']
    df1['combined_text'] = df1[df1_text_columns].astype(str).agg(' '.join, axis=1)

    # Prepare combined text for df2 with improved formatting
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

# --- 2. Embedding Generation ---
def generate_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = list(model.encode(df['combined_text'].tolist()))
    df['embeddings'] = embeddings
    return df, model

# --- 3. Vector Store (FAISS) ---
def create_faiss_index(df):
    embeddings_array = np.array(df['embeddings'].tolist()).astype('float32')
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    return index

# --- 4. Retrieval ---
def retrieve_relevant_data(query_text, model, index, df, k=3):
    query_embedding = model.encode([query_text]).astype('float32')
    D, I = index.search(query_embedding, k=k)
    retrieved_contexts = [df.loc[idx, 'combined_text'] for idx in I[0]]
    return " ".join(retrieved_contexts)

# --- 5. Generation (LLM) ---
def generate_answer(question, context):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    llm_response = qa_pipeline(question=question, context=context)
    return llm_response['answer']

# --- Main Chatbot Functionality ---
def chatbot_query(question):
    df1, df2 = load_and_preprocess_data()
    # We'll focus on df2 for sustainability data for this example
    df2, embedding_model = generate_embeddings(df2)
    faiss_index = create_faiss_index(df2)

    context = retrieve_relevant_data(question, embedding_model, faiss_index, df2)
    answer = generate_answer(question, context)
    return answer

if __name__ == "__main__":
    print("Initializing RAG Chatbot...")
    # Example usage
    user_question = "What is the total plastic reduced in August?"
    response = chatbot_query(user_question)
    print(f"\nQuestion: {user_question}")
    print(f"Answer: {response}")

    user_question2 = "What was the energy consumption on 05-Aug-25?"
    response2 = chatbot_query(user_question2)
    print(f"\nQuestion: {user_question2}")
    print(f"Answer: {response2}")

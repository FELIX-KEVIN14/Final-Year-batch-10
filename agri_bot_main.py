import streamlit as st

# ⛳ MUST be the first Streamlit command!
st.set_page_config(page_title="Construction Chatbot", page_icon="🏗️", layout="centered")
def redirect(url):
    st.markdown(f"""
        <meta http-equiv="refresh" content="0; url={url}" />
    """, unsafe_allow_html=True)
import pandas as pd
import faiss
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# Load saved components
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon GPU
    print("Using MPS (Apple GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")
    
if st.button("Exit"):
    redirect("http://localhost:8502") # Stops the Streamlit script from running further

@st.cache_resource
def load_resources():
    tokenizer = BertTokenizer.from_pretrained("bert_qa_tokenizer")
    model = BertModel.from_pretrained("bert_qa_model")
    index = faiss.read_index("qa_faiss.index")
    df = pd.read_csv("qa_dataset.csv")
    return tokenizer, model, index, df

# Text encoder using BERT
def encode_text(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Get answer from FAISS
def retrieve_answer(query, tokenizer, model, index, df, top_k=1):
    query_embedding = encode_text([query], tokenizer, model)
    _, I = index.search(query_embedding, top_k)
    return df.iloc[I[0][0]]["Answer"]

# Load resources
tokenizer, model, index, df = load_resources()

# Streamlit UI setup
st.title("🏗️ Construction Site Chatbot")
st.markdown("Ask me anything related to construction!")

# Initialize chat history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
query = st.chat_input("Ask a construction-related question...")

# Process input and generate response
if query:
    answer = retrieve_answer(query, tokenizer, model, index, df)
    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("bot", answer))

# Display chat history
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

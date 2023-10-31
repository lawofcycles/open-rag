from langchain.vectorstores import Epsilla
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import subprocess
from typing import List

# embed model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
db = FAISS.load_local("faiss_index", embeddings)

st.title("💬 Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Answer user question upon receiving
if question := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": question})

    context = '\n'.join(map(lambda doc: doc.page_content, db.similarity_search(question, k = 5)))

    st.chat_message("user").write(question)

    # Here we use prompt engineering to ingest the most relevant pieces of chunks from knowledge into the prompt.


    # Append the response
    msg = { 'role': 'assistant', 'content': content }
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg['content'])
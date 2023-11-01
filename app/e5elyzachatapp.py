# Internal usage
from time import  sleep
#### IMPORTS FOR AI PIPELINES 
import requests 
import streamlit as st

# #AVATARS
# av_us = './man.png'  #"🦖"  #A single emoji, e.g. "🧑‍💻", "🤖", "🦖". Shortcodes are not supported.
# av_ass = './lamini.png'

# FUNCTION TO LOG ALL CHAT MESSAGES INTO chathistory.txt
def writehistory(text):
    with open('chathistory.txt', 'a') as f:
        f.write(text)
        f.write('\n')
    f.close()


st.title("OSS RAG ChatBot")
st.subheader("Using model intfloat/multilingual-e5-large and elyza/ELYZA-japanese-Llama-2-7b-fast-instruct")

# Set a default model
# if "hf_model" not in st.session_state:
#     st.session_state["hf_model"] = "MBZUAI/LaMini-Flan-T5-77M"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Accept user input
if myprompt := st.chat_input("ご質問をどうぞ"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": myprompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(myprompt)
        usertext = f"user: {myprompt}"
        writehistory(usertext)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        apiresponse = requests.get(f'http://127.0.0.1:8000/model?question={myprompt}')
        risposta = apiresponse.content.decode("utf-8")
        res  =  risposta[1:-1]
        response = res.split(" ")
        for r in response:
            full_response = full_response + r + " "
            message_placeholder.markdown(full_response + "▌")
            sleep(0.1)
        message_placeholder.markdown(full_response)
        asstext = f"assistant: {full_response}"
        writehistory(asstext)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

uploaded_file = st.file_uploader('Choose a source pdf')

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import argparse
import os
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
    st.write(file_details)
    with open(os.path.join("resource",uploaded_file.name),"wb") as f: 
        f.write(uploaded_file.getbuffer())
        st.success("Saved File")

    # loader = UnstructuredFileLoader(uploaded_file.getvalue())
    # documents = loader.load()

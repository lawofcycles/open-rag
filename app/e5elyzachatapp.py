from time import  sleep
import requests 
import streamlit as st

st.title("OSS RAG ChatBot")
st.markdown("""##### intfloat/multilingual-e5-largeとelyza/ELYZA-japanese-Llama-2-7b-fast-instructを使ったMUFG FAQ(https://faq01.bk.mufg.jp/?site_domain=default)のRAGです""")

# 履歴を保存するsession_state
# Streamlitはユーザが画面を操作するたびにスクリプト全体が再実行されるが、session_stateの値は保持される
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    # role ごとに表示を変えたい場合は編集
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# チャット入力欄
if myprompt := st.chat_input("ご質問をどうぞ"):
    # ユーザメッセージを履歴に追加
    st.session_state.messages.append({"role": "user", "content": myprompt})
    # ユーザメッセージを画面に表示
    with st.chat_message("user"):
        st.markdown(myprompt)
        usertext = f"user: {myprompt}"
    # Chat Botの応答を画面に表示
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        res = requests.get(f'http://127.0.0.1:8000/query?question={myprompt}', timeout=120)
        message = res.content.decode("utf-8")
        message  =  message[1:-1]
        message = message.replace("\\n\\n", "\n")
        message = message.replace("\\n", "\n")
        for r in message:
            full_response = full_response + r
            message_placeholder.markdown(full_response + "▌")
            sleep(0.01)
        message_placeholder.markdown(full_response)
        asstext = f"assistant: {full_response}"
        st.session_state.messages.append({"role": "assistant", "content": full_response})
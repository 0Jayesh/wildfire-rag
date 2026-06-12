import streamlit as st
import requests

st.set_page_config(page_title="Wildfire Research Assistant", page_icon="🔥")

st.title(" Wildfire Research Assistant")
st.caption("Ask questions about wildfire detection research — backed by published papers and MTech thesis")

API_URL = "http://127.0.0.1:8000/query"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about wildfire detection research..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(API_URL, json={"question": question})
            answer = response.json()["answer"]
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
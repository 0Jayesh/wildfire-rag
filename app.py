import streamlit as st
import requests

st.set_page_config(page_title="Wildfire Research Assistant", page_icon="🔥")

st.title(" Wildfire Research Assistant")
st.caption("Ask questions about wildfire detection research.")

st.markdown("""
    <style>
    .stButton > button[kind="primary"] {
        background-color: #f0f2f680;
        border-color: #f0f2f680;
    }
    .stButton > button[kind="secondary"] {
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/query"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            col1, col2, _ = st.columns([2, 2, 20])
            feedback = msg.get("feedback")
            with col1:
                if st.button("👍", key=f"up_{i}", type="primary" if feedback == "satisfied" else "secondary"):
                    st.session_state.messages[i]["feedback"] = "satisfied"
                    st.rerun()
            with col2:
                if st.button("👎", key=f"down_{i}", type="primary" if feedback == "unsatisfied" else "secondary"):
                    st.session_state.messages[i]["feedback"] = "unsatisfied"
                    st.rerun()

            if feedback == "unsatisfied":
                st.caption("Sorry about that. Try rephrasing your question for a better answer.")

# Chat input
if question := st.chat_input("Ask a question about wildfire detection research..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Build history from session state
            history = []
            msgs = st.session_state.messages[:-1]  # exclude current question
            for i in range(0, len(msgs)-1, 2):
                if msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
                    # history.append({"question": msgs[i]["content"], "answer": msgs[i+1]["content"]})
                    history.append({
                        "question": msgs[i]["content"], 
                        "answer": msgs[i+1]["content"],
                        "feedback": msgs[i+1].get("feedback")
                    })
            
            response = requests.post(API_URL, json={"question": question, "chat_history": history})
            answer = response.json()["answer"]
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

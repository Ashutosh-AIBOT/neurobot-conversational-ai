import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# ---------------------- PAGE HEADER ----------------------
st.title("🤖 NeuroBot – Intelligent Chat Assistant - 1 ")
st.markdown("""
Welcome to **NeuroBot**, your intelligent conversational AI assistant.  
Type your message below to start chatting.  

---
""")

# ---------------------- SESSION MEMORY ----------------------
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Render previous conversation
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# ---------------------- USER INPUT ----------------------
user_input = st.chat_input("Type your message here...")

if user_input:
    # Show user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # ---------------------- SAFE API CALL ----------------------
    try:
        response = chatbot.invoke(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG
        )

        ai_message = response['messages'][-1].content

    except Exception as e:

        # Friendly clean error message
        error_text = (
            "⚠️ **Error Occurred**\n\n"
            "Something went wrong while connecting to the AI Model.\n\n"
            "**Possible Reasons:**\n"
            "- Invalid or missing API Key\n"
            "- Network failure\n"
            "- Backend server not reachable\n"
            "- Authentication error (401)\n\n"
            "**Technical Details:**\n"
            f"`{str(e)}`"
        )

        ai_message = error_text

    # ---------------------- SHOW AI MESSAGE ----------------------
    st.session_state['message_history'].append(
        {'role': 'assistant', 'content': ai_message}
    )

    with st.chat_message("assistant"):
        st.text(ai_message)

import streamlit as st
import requests

st.set_page_config(
    page_title="Federated AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Federated Meta-Learning AI Assistant")
st.write("Ask me anything!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_message = st.chat_input("Type your question here...")

if user_message:

    st.session_state.messages.append({
        "role": "user",
        "content": user_message
    })

    with st.chat_message("user"):
        st.write(user_message)

    try:
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"message": user_message},
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        ai_response = data.get(
            "response",
            "Sorry, I could not generate an answer."
        )

    except requests.exceptions.ConnectionError:
        ai_response = (
            "Unable to connect to the AI backend. "
            "Please make sure FastAPI is running."
        )

    except requests.exceptions.Timeout:
        ai_response = (
            "The AI model is taking longer than expected. "
            "Please try again."
        )

    except Exception as e:
        ai_response = f"Error: {str(e)}"

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response
    })

    with st.chat_message("assistant"):
        st.write(ai_response)
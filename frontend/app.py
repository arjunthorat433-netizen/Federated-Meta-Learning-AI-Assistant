import streamlit as st
import requests

st.set_page_config(
    page_title="Personal AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
}

.dashboard-card {
    background-color: white;
    padding: 22px;
    border-radius: 15px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.card-title {
    font-size: 18px;
    font-weight: bold;
}

.card-value {
    font-size: 28px;
    font-weight: bold;
    margin-top: 8px;
}

.info-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "",
        "email": ""
    }

if "preferences" not in st.session_state:
    st.session_state.preferences = {
        "language": "English",
        "response_style": "Simple"
    }


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("🤖 AI Assistant")
st.sidebar.caption("Federated Meta-Learning System")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "💬 AI Chat",
        "👤 Profile",
        "⚙️ Preferences"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Personalized AI Assistant\n\n"
    "Powered by Federated Meta-Learning"
)


# -----------------------------
# Dashboard
# -----------------------------

if page == "📊 Dashboard":

    st.title("🤖 Personal AI Assistant")
    st.subheader("Dashboard")

    st.write(
        "Welcome to your personalized AI assistant."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">🤖 AI Model</div>
            <div class="card-value">Ready</div>
            <p>AI model integration status</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">🎯 Personalization</div>
            <div class="card-value">Active</div>
            <p>User personalization enabled</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">🌐 Federated Clients</div>
            <div class="card-value">3</div>
            <p>Virtual edge clients</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("System Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>🧠 Meta-Learning</h3>
        <p>
        The system learns user preferences and adapts
        responses for personalized assistance.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>🔐 Federated Learning</h3>
        <p>
        User data can remain on edge devices while
        model knowledge is collaboratively improved.
        </p>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------
# AI Chat
# -----------------------------

elif page == "💬 AI Chat":

    st.title("💬 AI Chat")

    st.caption(
        "Chat with your personalized AI assistant"
    )

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_message = st.chat_input(
        "Ask your AI assistant..."
    )

    if user_message:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        with st.chat_message("user"):
            st.write(user_message)

        try:

            api_response = requests.post(
                "http://127.0.0.1:8000/chat",
                json={
                    "message": user_message
                },
                timeout=10
            )

            if api_response.status_code == 200:

                data = api_response.json()

                response = data["response"]

            else:

                response = "Backend returned an error."

        except requests.exceptions.RequestException:

            response = (
                "Unable to connect to the AI backend. "
                "Please make sure FastAPI is running."
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        with st.chat_message("assistant"):
            st.write(response)


# -----------------------------
# Profile
# -----------------------------

elif page == "👤 Profile":

    st.title("👤 User Profile")

    st.write(
        "Manage your personal information."
    )

    name = st.text_input(
        "Name",
        value=st.session_state.profile["name"]
    )

    email = st.text_input(
        "Email",
        value=st.session_state.profile["email"]
    )

    if st.button(
        "💾 Save Profile",
        use_container_width=True
    ):

        st.session_state.profile["name"] = name
        st.session_state.profile["email"] = email

        st.success(
            "Profile saved successfully!"
        )


# -----------------------------
# Preferences
# -----------------------------

elif page == "⚙️ Preferences":

    st.title("⚙️ User Preferences")

    st.write(
        "Customize your AI assistant."
    )

    language_options = [
        "English",
        "Marathi",
        "Hindi"
    ]

    response_options = [
        "Short",
        "Detailed",
        "Simple"
    ]

    language = st.selectbox(
        "Preferred Language",
        language_options,
        index=language_options.index(
            st.session_state.preferences["language"]
        )
    )

    response_style = st.selectbox(
        "Response Style",
        response_options,
        index=response_options.index(
            st.session_state.preferences["response_style"]
        )
    )

    if st.button(
        "💾 Save Preferences",
        use_container_width=True
    ):

        st.session_state.preferences["language"] = language

        st.session_state.preferences["response_style"] = response_style

        st.success(
            "Preferences saved successfully!"
        )
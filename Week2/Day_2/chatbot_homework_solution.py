import streamlit as st
import requests
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set page configuration
st.set_page_config(page_title="Chatbot Interface", layout="wide")

# App title
st.title("Hugging Face Chatbot")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "hf_key" not in st.session_state:
    # Fall back to HF_TOKEN from the environment / .env so the key can be pre-set
    st.session_state.hf_key = os.getenv("HF_TOKEN", "")

# Models currently served through the HuggingFace router.
# The old "mistralai/Mistral-7B-Instruct-v0.3" was retired by the provider — it now
# returns HTTP 410 ("model is deprecated and no longer supported").
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "google/gemma-3-12b-it",
]

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    hf_key = st.text_input(
        "Enter your HuggingFace API key:",
        type="password",
        value=st.session_state.hf_key,
    )
    if hf_key:
        st.session_state.hf_key = hf_key

    model = st.selectbox("Model:", MODELS, index=0)

    # Button to start a new conversation
    if st.button("New Conversation"):
        st.session_state.messages = []
        st.rerun()

# Configure API details
# Unified router endpoint — HuggingFace picks an available provider for the model.
API_URL = "https://router.huggingface.co/v1/chat/completions"

def query(payload, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()

        # Surface the API's own message — it explains *why* the call failed
        # (bad key, deprecated model, rate limit, out of credits, ...)
        try:
            detail = response.json().get("error", response.text)
            if isinstance(detail, dict):
                detail = detail.get("message", detail)
        except ValueError:
            detail = response.text

        st.error(f"Error {response.status_code}: {detail}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {e}")
        return None

# Display the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask something..."):
    if not st.session_state.hf_key:
        st.error("Please enter your HuggingFace API key in the sidebar.")
    else:
        # Add user message to the chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display the user message
        with st.chat_message("user"):
            st.write(prompt)

        # Display a spinner while waiting for the API response
        with st.spinner("Thinking..."):
            # Prepare the payload from the conversation history
            payload = {
                "messages": st.session_state.messages,
                "max_tokens": 512,
                "model": model
            }

            # Query the API
            response = query(payload, st.session_state.hf_key)

            bot_response = None
            if response and response.get("choices"):
                bot_response = response["choices"][0]["message"].get("content")

            if bot_response:
                # Add assistant response to the chat history
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

                # Display the assistant response
                with st.chat_message("assistant"):
                    st.write(bot_response)
            else:
                # Drop the unanswered user turn so the next try doesn't send
                # two user messages in a row (providers reject that).
                st.session_state.messages.pop()
                st.error("Failed to get a valid response from the API.")


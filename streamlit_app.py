# Import required libraries
from dotenv import load_dotenv
from itertools import zip_longest
import requests
import streamlit as st  
from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# HuggingFace API KEY input
API_KEY = "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"

# HuggingFace API inference URL.
API_URL = "https://hbmffa2mp0q2rzmg.us-east-1.aws.endpoints.huggingface.cloud"

headers = {"Authorization": f"Bearer {API_KEY}"}

# Load environment variables
load_dotenv()

# Set streamlit page configuration
st.set_page_config(page_title="Hugging Face Open Source Chat Demo")
st.title("Open Source LLM Chat Demo")

# Extract query parameters
query_params = st.experimental_get_query_params()
OPENAI_API_KEY = query_params.get("api_key", [""])[0]

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input



def build_message_list():
    """
    Build a list of messages including system, human and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    past_user_inputs =[]
    past_generated_responses = []

    # Zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            past_user_inputs.append(human_msg)  # Add user messages
        if ai_msg is not None:
            past_generated_responses.append(ai_msg)  # Add AI messages

    return {"inputs": human_msg,"parameters":{"temperature":0.1,"max_new_tokens":512}}


def generate_response():
    """
    Generate AI response using the ChatOpenAI model.
    """
    # Build the list of messages
    zipped_messages = build_message_list()

    # Query the API
    ai_response = requests.post(API_URL, headers=headers, json=zipped_messages)
  
    return ai_response.json()[0]['generated_text']


# Define function to submit user input
def submit():
    # Set entered_prompt to the current value of prompt_input
    st.session_state.entered_prompt = st.session_state.prompt_input
    # Clear prompt_input
    st.session_state.prompt_input = ""


# Create a text input for user
st.text_input('YOU: ', key='prompt_input', on_change=submit)


if st.session_state.entered_prompt != "":
    # Get user query
    user_query = st.session_state.entered_prompt

    # Append user query to past queries
    st.session_state.past.append(user_query)

    # Generate response
    output = generate_response()

    # Append AI response to generated responses
    st.session_state.generated.append(output)

# Display the chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # Display AI response
        message(st.session_state["generated"][i], key=str(i))
        # Display user message
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user')

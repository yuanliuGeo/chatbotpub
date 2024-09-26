import streamlit as st
import boto3
import json
import os
import re
import html  # For decoding HTML entities

# Configure AWS credentials using environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = os.getenv('AWS_REGION', 'us-east-1')  # Default to 'us-east-1' if not set

# Create a Bedrock Agent Runtime client in the AWS Region you want to use.
bedrock_agent_runtime_client = boto3.client(
    'bedrock-agent-runtime',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# Set the model ID, e.g., Titan Text Premier.
model_id = "anthropic.claude-3-haiku-20240307-v1:0"
knowledge_base_id = os.getenv('KNOWLEDGE_BASE_ID', "OWMDKF0HOE")  # Replace with your knowledge base ID or use env var
model_arn = f'arn:aws:bedrock:{region_name}::foundation-model/{model_id}'

# Function to interact with Bedrock's retrieve_and_generate API
def retrieveAndGenerate(input_text, kb_id, model_arn, session_id=""):
    if session_id:
        response = bedrock_agent_runtime_client.retrieve_and_generate(
            input={'text': input_text},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': kb_id,
                    'modelArn': model_arn
                }
            },
            sessionId=session_id
        )
    else:
        response = bedrock_agent_runtime_client.retrieve_and_generate(
            input={'text': input_text},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': kb_id,
                    'modelArn': model_arn
                }
            }
        )

    return response

# Function to split text by code snippets
def split_text_by_code(text):
    # Split the text by <pre><code>...</code></pre> and ```...``` blocks
    split_text = re.split(r'(<pre><code>.*?</code></pre>|```.*?```)', text, flags=re.DOTALL)
    return split_text

# Function to display text and code snippets in order
def display_message(message):
    bot_response = message['bot']

    # Split the response by code and text
    split_content = split_text_by_code(bot_response)

    for part in split_content:
        if part.startswith('<pre><code>') and part.endswith('</code></pre>'):
            # Extract and decode the code inside <pre><code>
            code_content = re.search(r'<pre><code>(.*?)</code></pre>', part, re.DOTALL).group(1)
            decoded_code = html.unescape(code_content.strip())
            st.code(decoded_code, language='html')  # Adjust language as needed
        elif part.startswith('```') and part.endswith('```'):
            # Extract code inside ```
            code_content = part.strip('```').strip()
            st.code(code_content, language='html')  # Adjust language as needed
        else:
            # Regular text
            st.write(part.strip())

# Set up the Streamlit app
st.title("GeoComply Client Portal Chatbot")
st.write("Chat with the demo AI chatbot, any question or recommendation please contact yuan.liu@geocomply.com ")

# Session state to store chat history and session ID
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = ""

# Function to handle sending a message with a prompt
def send_message():
    user_input = st.session_state.user_input
    if user_input:
        # Define your prompt
        prompt = "You are an AI chatbot specialized in providing detailed assistance with GeoComply's Client Portal. Respond thoroughly to user queries. When relevant, provide clear and concise code snippets. Structure your responses to enhance user understanding."

        # Combine the prompt and user input
        full_input = f"{prompt}\n\nUser: {user_input}\nChatbot:"

        # Call the retrieveAndGenerate function with the combined prompt and user input
        response = retrieveAndGenerate(full_input, knowledge_base_id, model_arn, st.session_state.session_id)

        # Extract the output text from the response
        try:
            output_text = response['output']['text']
            # Capture and store the session ID from the response for future interactions
            if 'sessionId' in response:
                st.session_state.session_id = response['sessionId']
        except (KeyError, IndexError):
            output_text = "An error occurred while processing your request."

        # Add the user query and chatbot response to chat history
        st.session_state.chat_history.append({"user": user_input, "bot": output_text})

        # Clear the input box after submission
        st.session_state.user_input = ""

# Display chat history with code detection
for message in st.session_state.chat_history:
    st.write(f"**You:** {message['user']}")
    display_message(message)

# Input box for user query with on_change callback
st.text_input("Your message:", key="user_input", on_change=send_message)

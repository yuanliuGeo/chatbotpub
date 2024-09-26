import streamlit as st
import boto3
import json
import os

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

# Set up the Streamlit app
st.title("GeoComply Client Portal Chatbot")
st.write("Chat with the demo AI chatbot, any question or recommendation please contact yuan.liu@geocomply.com ")

# Session state to store chat history and session ID
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = ""

# Function to handle sending a message
def send_message():
    user_input = st.session_state.user_input
    if user_input:
        # Call the retrieveAndGenerate function
        response = retrieveAndGenerate(user_input, knowledge_base_id, model_arn, st.session_state.session_id)

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

# Display chat history
for message in st.session_state.chat_history:
    st.write(f"**You:** {message['user']}")
    st.write(f"**Chatbot:** {message['bot']}")

# Input box for user query with on_change callback
st.text_input("Your message:", key="user_input", on_change=send_message)


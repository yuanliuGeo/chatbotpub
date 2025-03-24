import streamlit as st
import boto3
import json
import os
import re
import html
import time
import requests

# AWS Configuration
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = os.getenv('AWS_REGION', 'us-east-1')

bedrock_agent_runtime_client = boto3.client(
    'bedrock-agent-runtime',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# Model ARNs
haiku_model_arn = f'arn:aws:bedrock:{region_name}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
sonnet_model_arn = f'arn:aws:bedrock:{region_name}::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0'
llama_model_arn = f'arn:aws:bedrock:{region_name}::foundation-model/meta.llama3-70b-instruct-v1:0'

knowledge_base_id = os.getenv('KNOWLEDGE_BASE_ID', "GPQI1PCIBJ")

# Guru Endpoint Configuration
guru_url = "https://api.getguru.com/api/v1/answers"
guru_agent_id = "db7c2221-e5c4-4030-abdc-02addaa0268e"
guru_headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Basic dmxhZGlzbGF2QGdlb2NvbXBseS5jb206NjhmNGM5ZjMtYWI0Ni00N2QwLTg2YTgtODM3MmFhZmYyM2Ri"
}

# Retrieve and Generate for Bedrock
def retrieveAndGenerate(input_text, kb_id, model_arn, session_id=""):
    kwargs = {
        'input': {'text': input_text},
        'retrieveAndGenerateConfiguration': {
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': kb_id,
                'modelArn': model_arn
            }
        }
    }
    if session_id:
        kwargs['sessionId'] = session_id
    return bedrock_agent_runtime_client.retrieve_and_generate(**kwargs)

# Guru API request
def query_guru(question):
    payload = {"agentId": guru_agent_id, "question": question}
    response = requests.post(guru_url, json=payload, headers=guru_headers)
    if response.status_code == 200:
        return response.json().get("answer", "No answer found.")
    return "Guru API error."

# Display helpers
def split_text_by_code(text):
    return re.split(r'(<pre><code>.*?</code></pre>|```.*?```)', text, flags=re.DOTALL)

def display_bot_text(text):
    for part in split_text_by_code(text):
        if part.startswith('<pre><code>') and part.endswith('</code></pre>'):
            code_content = re.search(r'<pre><code>(.*?)</code></pre>', part, re.DOTALL).group(1)
            st.code(html.unescape(code_content.strip()), language='html')
        elif part.startswith('```') and part.endswith('```'):
            st.code(part.strip('```').strip(), language='html')
        else:
            st.write(part.strip())

# Streamlit layout
st.set_page_config(layout="wide")
st.title("GeoComply Client Portal Chatbot")
st.markdown("Chat with the demo AI chatbot. For questions, contact [yuan.liu@geocomply.com](mailto:yuan.liu@geocomply.com)")

# Session state initialization
for key in ["chat_history", "session_id_haiku", "session_id_sonnet", "session_id_llama"]:
    if key not in st.session_state:
        st.session_state[key] = "" if "session_id" in key else []

# Handle sending messages
def send_message():
    user_input = st.session_state.user_input
    if user_input:
        prompt = (
            "You are an AI chatbot specialized in providing detailed assistance with GeoComply's Client Portal."
            "Respond thoroughly to user queries, providing clear, structured answers and concise code snippets."
        )
        full_input = f"{prompt}\n\nUser: {user_input}\nChatbot:"

        responses, latencies = {}, {}

        models = [
            ("Haiku", haiku_model_arn, "session_id_haiku"),
            ("Sonnet 3.5", sonnet_model_arn, "session_id_sonnet"),
            ("Llama 3", llama_model_arn, "session_id_llama")
        ]

        for model_name, model_arn, session_key in models:
            start = time.time()
            response = retrieveAndGenerate(full_input, knowledge_base_id, model_arn, st.session_state[session_key])
            latencies[model_name] = time.time() - start
            responses[model_name] = response['output']['text']
            st.session_state[session_key] = response.get('sessionId', st.session_state[session_key])

        # Query Guru endpoint
        start = time.time()
        guru_response = query_guru(user_input)
        latencies["Guru"] = time.time() - start
        responses["Guru"] = guru_response

        st.session_state.chat_history.append({"user": user_input, "responses": responses, "latencies": latencies})
        st.session_state.user_input = ""

# Display chat history
for message in st.session_state.chat_history:
    st.markdown(f"**You:** {message['user']}")
    cols = st.columns(4)
    for idx, model_name in enumerate(["Haiku", "Sonnet 3.5", "Llama 3", "Guru"]):
        with cols[idx]:
            st.subheader(f"{model_name} Model")
            st.caption(f"Latency: {message['latencies'][model_name]:.2f} seconds")
            display_bot_text(message['responses'][model_name])

# User input
st.text_input("Your message:", key="user_input", on_change=send_message)

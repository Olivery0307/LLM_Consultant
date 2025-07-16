import sys
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from celery.result import AsyncResult

st.set_page_config(page_title="AI Business Consultant", layout="wide")

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the agent classes
from agents.web_agent import WebConsultantAgent
from agents.lead_consultant import LeadConsultantAgent, process_uploaded_files_for_tools
from tasks import run_case_study_task, celery_app

import langchain
from langchain_community.cache import RedisCache
import redis


# --- Redis Cache Configuration ---
try:
    # Connect to the running Redis container
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    # Set the global LangChain cache
    langchain.llm_cache = RedisCache(redis_client)
    print("INFO: Redis cache configured successfully.")
except Exception as e:
    print(f"WARNING: Could not connect to Redis. Caching is disabled. Error: {e}")
# ----------------------------------------------------

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


# Load environment variables
load_dotenv()

# --- App UI Configuration ---
st.title("🤖 AI Business Consultant")
st.markdown("Your intelligent partner for strategic business analysis.")

# --- Initialize Agents ---
@st.cache_resource
def get_web_agent():
    return WebConsultantAgent()

@st.cache_resource
def get_lead_agent():
    return LeadConsultantAgent()

web_agent = get_web_agent()
lead_agent = get_lead_agent()

# --- Mode Selection Sidebar ---
st.sidebar.header("Select Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Choose your consultant's specialty:",
    ("Case Study Analysis", "General Web Consultant", "SWOT Analysis")
)

# --- Render UI for the selected mode ---

if analysis_mode == "Case Study Analysis":
    st.header("🗂️ Case Study Analysis")
    st.markdown("Define a central business question and upload all relevant case files (`.pdf`, `.txt`, `.csv`). The Lead Consultant will perform a comprehensive analysis.")

    case_question = st.text_area("Enter the central business question for this case study:")
    
    uploaded_files = st.file_uploader(
        "Upload your case files",
        type=['pdf', 'txt', 'csv'],
        accept_multiple_files=True
    )

    if st.button("Generate Full Analysis"):
        if not case_question or not uploaded_files:
            st.warning("Please enter a business question and upload at least one file.")
        else:
            files_data = [{'name': f.name, 'data': f.getvalue()} for f in uploaded_files]
            task = run_case_study_task.delay(case_question, files_data)
            st.session_state.task_id = task.id
            st.success(f"Analysis has started! The task ID is: {task.id}. You can check the status below.")
            with st.spinner("Processing documents and preparing analysis..."):
                process_uploaded_files_for_tools(uploaded_files)
            
    # Check for and display the result of a running task
    if 'task_id' in st.session_state:
        task_id = st.session_state.task_id
        # --- FIX APPLIED HERE: Use the configured celery_app to get the result ---
        result = celery_app.AsyncResult(task_id)

        if result.ready():
            if result.successful():
                st.write("### Lead Consultant's Final Report:")
                st.markdown(result.get())
                # Clean up the task ID after displaying the result
                del st.session_state.task_id
            else:
                st.error(f"Task failed: {result.state} - {result.info}")
                del st.session_state.task_id
        else:
            st.info(f"Analysis is still in progress (Status: {result.state}). Please wait or refresh the page.")

elif analysis_mode == "General Web Consultant":
    st.header("🌐 Conversational Web Consultant")
    st.markdown("Ask a question and have a follow-up conversation with the agent.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # Get user input
    user_question = st.chat_input("Ask a question about a company...")
    
    if user_question:
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        with st.chat_message("Human"):
            st.write(user_question)
            
        with st.spinner("Consultant is thinking..."):
            # Pass the history to the agent
            response = web_agent.run_general_consultation(user_question, st.session_state.chat_history)
            
            # Add AI response to history and display it
            st.session_state.chat_history.append(AIMessage(content=response))
            with st.chat_message("AI"):
                st.write(response)

elif analysis_mode == "SWOT Analysis":
    st.header("📊 SWOT Analysis")
    st.markdown("Enter a company name to generate a comprehensive SWOT analysis.")
    company_name = st.text_input("Enter a company name (e.g., Apple, Netflix, Ford):")
    if company_name and st.button(f"Generate SWOT for {company_name}"):
        with st.spinner(f"Performing SWOT analysis for {company_name}..."):
            report = web_agent.run_swot_analysis(company_name)
            st.write(f"### SWOT Analysis: {company_name}")
            st.markdown(report)


st.sidebar.markdown("---")
st.sidebar.info("Project by Chung-Yeh (Oliver) Yang")

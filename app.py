import streamlit as st
from dotenv import load_dotenv

# Import the display functions from our new modules
from web_consultant import display_web_consultant, display_swot_analyzer
from file_analyzer import display_file_analyzer

# Load environment variables from .env file at the very beginning
load_dotenv()

# --- Main App UI ---

st.set_page_config(page_title="AI Business Consultant", layout="wide")

st.title("🤖 AI Business Consultant")
st.markdown("Your intelligent partner for business analysis, from web searches to document deep dives.")

# --- Mode Selection ---
st.sidebar.header("Select Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Choose how you want to interact with the consultant:",
    ("General Web Consultant", "SWOT Analysis", "Document Analysis") # Added SWOT Analysis
)

# --- Render the selected mode's UI ---
if analysis_mode == "General Web Consultant":
    display_web_consultant()
elif analysis_mode == "SWOT Analysis":
    display_swot_analyzer() # Call the new function
elif analysis_mode == "Document Analysis":
    display_file_analyzer()

st.sidebar.markdown("---")
st.sidebar.info("Project by Chung-Yeh Yang")

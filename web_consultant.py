import streamlit as st
from langchain.agents import tool, create_react_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain import hub
from langchain.prompts import PromptTemplate
from utils import get_llm  # Import from our new utils file

# --- Agent Tools ---

@tool
def scrape_website(url: str) -> str:
    """Scrapes the text content of a given URL."""
    print(f"Scraping website: {url}...")
    try:
        loader = WebBaseLoader([url])
        docs = loader.load()
        content = " ".join([doc.page_content for doc in docs])
        return content[:15000]
    except Exception as e:
        return f"Error scraping website: {e}"

@st.cache_resource
def create_web_agent_executor():
    """Creates the web-browsing agent executor."""
    print("Creating web agent executor...")
    llm = get_llm()
    search_tool = TavilySearchResults(max_results=5)
    tools = [search_tool, scrape_website]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Enhanced Prompt Templates ---

def create_consultant_prompt(question):
    """Wraps a general question with a sophisticated prompt."""
    template = """
    You are an expert business consultant...
    User Question: "{question}"
    ...
    """ # Keeping this brief as the logic is the same
    return template.format(question=question)

def create_swot_prompt(company_name):
    """Creates a structured prompt for a SWOT analysis."""
    template = """
    You are an expert strategic consultant. Your mission is to conduct a thorough SWOT analysis for the company: **{company_name}**.

    To do this, you must perform targeted web research for each of the four components. Structure your final output exactly as follows, with 3-4 bullet points for each section:

    **1. Strengths (Internal, Positive):**
    * (e.g., strong brand recognition, innovative technology, loyal customer base)

    **2. Weaknesses (Internal, Negative):**
    * (e.g., high operational costs, dependence on a single supplier, outdated technology stack)

    **3. Opportunities (External, Positive):**
    * (e.g., emerging markets, new favorable regulations, advancements in related technologies)

    **4. Threats (External, Negative):**
    * (e.g., new disruptive competitors, changing consumer preferences, potential for new tariffs or regulations)

    Perform your research now and generate the complete SWOT analysis.
    """
    return template.format(company_name=company_name)

# --- UI Display Functions ---

def display_web_consultant():
    """Renders the UI for the general Web Consultant mode."""
    st.header("🌐 General Web Consultant")
    st.markdown("Ask a question about a company. The AI will perform research and generate a structured report.")
    web_agent_executor = create_web_agent_executor()
    user_question = st.text_input("Example: Analyze the primary growth drivers for Tesla in the next 3 years.")
    if user_question:
        with st.spinner("Consultant is performing research and analysis..."):
            consultant_question = create_consultant_prompt(user_question)
            response = web_agent_executor.invoke({"input": consultant_question})
            st.write("### Consultant's Report:")
            st.markdown(response['output'])

def display_swot_analyzer():
    """Renders the UI for the SWOT Analysis mode."""
    st.header("📊 SWOT Analysis")
    st.markdown("Enter a company name to generate a comprehensive SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis.")
    web_agent_executor = create_web_agent_executor()
    company_name = st.text_input("Enter a company name (e.g., Apple, Netflix, Ford):")
    if company_name:
        if st.button(f"Generate SWOT Analysis for {company_name}"):
            with st.spinner(f"Performing SWOT analysis for {company_name}... This may take a moment."):
                swot_prompt = create_swot_prompt(company_name)
                response = web_agent_executor.invoke({"input": swot_prompt})
                st.write(f"### SWOT Analysis: {company_name}")
                st.markdown(response['output'])

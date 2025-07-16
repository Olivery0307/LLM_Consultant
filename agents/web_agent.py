import streamlit as st
from langchain.agents import tool, create_react_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from utils import get_llm

# --- Agent Tools ---

@tool
def scrape_website(url: str) -> str:
    """Scrapes the text content of a given URL."""
    try:
        loader = WebBaseLoader([url])
        docs = loader.load()
        content = " ".join([doc.page_content for doc in docs])
        return content[:15000]
    except Exception as e:
        return f"Error scraping website: {e}"

# --- The Agent Class ---

class WebConsultantAgent:
    def __init__(self):
        """
        Initializes the Web Consultant Agent with multiple executors for different tasks.
        """
        self.llm = get_llm()
        self.tools = [TavilySearchResults(max_results=7), scrape_website]
        
        # 1. Conversational Executor (with memory and a specialized persona)
        prompt_template = """
        You are an expert business and financial consultant. Your responses should be analytical, data-driven, and focused strictly on business, finance, and technology topics.
        You have access to the following tools to perform up-to-date research. Always use your tools to answer the user's question.

        TOOLS:
        ------
        {tools}

        To use a tool, please use the following format:
        ```
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ```

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
        ```
        Thought: Do I need to use a tool? No
        Final Answer: [your response here]
        ```

        Begin!

        Previous conversation history:
        {chat_history}

        New input: {input}
        {agent_scratchpad}
        """
        conversational_prompt = PromptTemplate.from_template(prompt_template)
        conversational_agent = create_react_agent(self.llm, self.tools, conversational_prompt)
        self.conversational_executor = AgentExecutor(
            agent=conversational_agent, tools=self.tools, verbose=True, handle_parsing_errors=True
        )

        # 2. Report Executor (memory-less, for structured one-off reports)
        report_prompt_template = """
        Answer the following questions as best you can. You have access to the following tools:
        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """
        react_prompt = PromptTemplate.from_template(report_prompt_template)
        report_agent = create_react_agent(self.llm, self.tools, react_prompt)
        self.report_executor = AgentExecutor(
            agent=report_agent, tools=self.tools, verbose=True, handle_parsing_errors=True
        )

    def _create_swot_prompt(self, company_name):
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
        """
        return template.format(company_name=company_name)

    def _create_competitor_prompt(self, company_name):
        """Creates a structured prompt for a competitor analysis."""
        template = """
        You are an expert market analyst. Your mission is to conduct a competitor analysis for **{company_name}**.

        Follow these steps:
        1.  First, use your search tool to identify the top 3-4 direct competitors for {company_name}.
        2.  For each competitor identified, perform a new search to find their approximate Market Capitalization and their primary product or service.
        3.  Present your findings in a Markdown table with the following columns: Competitor, Market Cap, Primary Product/Service.
        4.  Conclude with a brief summary of the competitive landscape.

        Begin your analysis now.
        """
        return template.format(company_name=company_name)

    def run_general_consultation(self, user_question, chat_history):
        """Runs the conversational agent with memory."""
        response = self.conversational_executor.invoke({
            "input": user_question, "chat_history": chat_history,
            "tools": self.tools, "tool_names": [tool.name for tool in self.tools]
        })
        return response['output']
    
    def run_swot_analysis(self, company_name):
        """Runs the dedicated report agent to perform a SWOT analysis."""
        prompt = self._create_swot_prompt(company_name)
        response = self.report_executor.invoke({"input": prompt})
        return response['output']
    
    def run_competitor_analysis(self, company_name):
        """Runs the dedicated report agent to perform a competitor analysis."""
        prompt = self._create_competitor_prompt(company_name)
        response = self.report_executor.invoke({"input": prompt})
        return response['output']
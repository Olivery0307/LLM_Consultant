import os
import pandas as pd
from langchain.agents import tool, create_react_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_core.prompts import PromptTemplate
from langchain import hub
from utils import get_llm, get_embeddings_model

vectorstore = None
@tool
def document_qa_tool(question: str) -> str:
    """
    Answers a question based on the content of uploaded PDF and TXT documents.
    Use this tool to find information within the provided case files.
    Input should be a clear, specific question.
    """
    global vectorstore
    if vectorstore is None:
        return "Error: No documents have been processed. Please upload PDF or TXT files first."
    
    llm = get_llm()
    retrieved_docs = vectorstore.similarity_search(question, k=5)
    
    # For simplicity, we'll use a basic QA chain here. This could be enhanced later.
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=retrieved_docs, question=question)
    return response

@tool
def csv_analysis_tool(question: str) -> str:
    """
    Analyzes a CSV file to answer a question.
    Use this for any questions related to data in the provided CSV files,
    including calculations, summaries, and creating visualizations.
    The input should be a clear question about the CSV data.
    """

    temp_csv_path = "/tmp/uploaded_csv_0.csv"
    if not os.path.exists(temp_csv_path):
        return "Error: No CSV file has been processed. Please upload a CSV file."

    llm = get_llm()
    agent_executor = create_csv_agent(llm, temp_csv_path, verbose=True, allow_dangerous_code=True)
    
    # A specific prompt to guide the agent
    prompt = f"""
    You are a data analyst. Your task is to analyze the provided CSV to answer the user's question.
    You MUST write and execute Python code using the pandas dataframe (df) to find the answer.
    If the user asks for a visualization, save the plot to a file named 'plot.png'.

    User Question: "{question}"
    """
    response = agent_executor.invoke({"input": prompt})
    return response['output']

def process_uploaded_files_for_tools(uploaded_files):
    """
    Processes uploaded files, creates a vector store for PDF/TXT,
    and saves CSVs to a known path for the tools to access.
    """
    global vectorstore
    other_files = [f for f in uploaded_files if not f.name.endswith('.csv')]
    csv_files = [f for f in uploaded_files if f.name.endswith('.csv')]

    # Process PDF/TXT files
    if other_files:
        all_docs = []
        for uploaded_file in other_files:
            temp_file_path = f"/tmp/{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader(temp_file_path) if uploaded_file.name.endswith('.pdf') else TextLoader(temp_file_path)
            all_docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(all_docs)
        embeddings = get_embeddings_model()
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    # Process CSV files
    if csv_files:
        # Save the first CSV to a predictable path for the tool
        temp_csv_path = "/tmp/uploaded_csv_0.csv"
        with open(temp_csv_path, "wb") as f:
            f.write(csv_files[0].getbuffer())

# --- The Lead Consultant Agent ---

class LeadConsultantAgent:
    def __init__(self):
        """Initializes the Lead Consultant Agent."""
        self.llm = get_llm()
        # The Lead Agent has access to all our tools
        self.tools = [TavilySearchResults(max_results=5), document_qa_tool, csv_analysis_tool]
        react_prompt_template = """
        Answer the following questions as best you can. You have access to the following tools:
        {tools}

        **Your primary goal is to delegate tasks to the appropriate tool.**
        - For questions about the content of text documents (PDFs, TXTs), use the `document_qa_tool`.
        - For any questions requiring data analysis, calculations, or visualization from a CSV file, use the `csv_analysis_tool`.
        - For general web research or information not in the documents, use `tavily_search_results_json`.

        Use the following format:

        Question: the input question you must answer
        Thought: I need to decide which tool is best suited to answer the user's question. Based on the question, I will choose one of my available tools.
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now have enough information from my tools to construct the final answer.
        Final Answer: the final, comprehensive answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """
        prompt = PromptTemplate.from_template(react_prompt_template)
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True
        )
        
    def _create_case_study_prompt(self, case_question):
        """Creates a detailed, structured prompt for the final report."""
        template = """
        You are a Lead Strategic Consultant. Your task is to analyze a complex business case based on a central question, a set of internal documents, and live web research.

        Synthesize all the information you gather into a single, comprehensive report structured as follows:

        **1. Executive Summary:**
           - A concise overview of the situation and your final recommendation.

        **2. Analysis of Internal Documents:**
           - Summarize the key findings from the provided PDF, TXT, and CSV files.
           - Extract specific data points, financial figures, or contractual terms that are relevant to the case.

        **3. External Market & Competitor Analysis:**
           - Summarize your findings from web research about the broader market context, trends, and competitor actions.

        **4. Synthesized Recommendation (Pros & Cons):**
           - Based on *both* internal and external findings, provide a balanced view.
           - List the primary "Pros" (arguments for the proposed action).
           - List the primary "Cons" (risks and arguments against the proposed action).

        **5. Final Recommendation:**
           - Conclude with a clear, actionable recommendation.

        Now, begin your analysis for the following case.

        **Central Business Question:** "{question}"
        """
        return template.format(question=case_question)


    def run_case_study(self, case_question):
        """Runs the full analysis on a complex case."""
        structured_prompt = self._create_case_study_prompt(case_question)
        response = self.executor.invoke({"input": structured_prompt})
        return response['output']
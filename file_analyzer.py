import streamlit as st
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from utils import get_llm, get_embeddings_model  # Import from utils

# --- Prompt Templates for Enhanced Analysis ---

def create_doc_summary_prompt(question, context):
    """
    Creates a prompt for summarizing and answering questions about text documents (PDF, TXT).
    """
    template = """
    You are an expert document analyst. Based on the provided context from a document,
    answer the user's question in a structured and comprehensive manner.

    Structure your response as follows:

    **1. Direct Answer:** A clear and direct answer to the user's question.
    **2. Supporting Evidence:** A bulleted list of key points or quotes from the document that support your answer.
    **3. Contextual Summary:** A brief summary of the document's overall theme or purpose, based on the provided context.

    **Context from Document:**
    {context}

    **User's Question:**
    {question}

    Your analysis:
    """
    return template.format(context=context, question=question)

def create_csv_analysis_prompt(question):
    """
    Creates a more explicit prompt for the CSV agent to force it into the correct output format.
    """
    template = """
    You are an expert data analyst. Your task is to analyze the provided pandas dataframe (`df`) to answer the user's question.
    You MUST write and execute Python code to find the answer.
    You have access to a tool called `python_repl_tool`.

    Use the following format for your response:

    Thought: I need to determine the best way to answer the user's question. I should use the `python_repl_tool` to execute some code.
    Action: python_repl_tool
    Action Input:
    ```python
    # Your pandas/matplotlib/seaborn code here.
    # ALWAYS save plots to a file named 'plot.png'.
    # Example for a plot:
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Category', y='Sales')
    plt.title('Total Sales by Category')
    plt.xlabel('Product Category')
    plt.ylabel('Total Sales')
    plt.savefig('plot.png')
    print("Plot successfully generated and saved to plot.png")
    ```
    Observation: [The result of the code execution will be here]
    Thought: I now have the answer based on the code output.
    Final Answer: [Your final, comprehensive answer here. If a plot was created, mention it and describe its insights.]

    Begin!

    User Question: "{question}"
    """
    return template.format(question=question)


# --- Document Processing Functions ---

def get_documents_from_files(uploaded_files):
    """Loads content from PDF and TXT files into LangChain documents."""
    all_docs = []
    for uploaded_file in uploaded_files:
        temp_dir = "/tmp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
            all_docs.extend(loader.load_and_split())
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(temp_file_path)
            all_docs.extend(loader.load())
    return all_docs

def get_text_chunks(documents):
    """Splits documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

@st.cache_resource
def get_vectorstore(_text_chunks):
    """Creates a FAISS vector store from text chunks."""
    if not _text_chunks:
        return None
    embeddings = get_embeddings_model()
    return FAISS.from_documents(documents=_text_chunks, embedding=embeddings)


def display_file_analyzer():
    """Renders the UI for the Document Analysis mode."""
    st.header("📄 Document Analysis")
    st.markdown("Upload your documents and ask the agent to perform calculations or create professional visualizations from your data.")

    uploaded_files = st.file_uploader(
        "Upload your files",
        type=['pdf', 'txt', 'csv'],
        accept_multiple_files=True
    )

    if not uploaded_files:
        return

    # Process files
    csv_files = [f for f in uploaded_files if f.name.endswith('.csv')]
    other_files = [f for f in uploaded_files if not f.name.endswith('.csv')]
    vectorstore = None

    if other_files:
        with st.spinner("Processing PDFs and TXTs..."):
            documents = get_documents_from_files(other_files)
            text_chunks = get_text_chunks(documents)
            vectorstore = get_vectorstore(text_chunks)
        st.success("PDF/TXT documents are ready for analysis!")

    # Display UI for non-CSV files
    if vectorstore:
        st.subheader("Ask a question about your PDFs and TXTs:")
        doc_question = st.text_input("e.g., What were the key findings of the Q2 report?")
        if doc_question:
            with st.spinner("Analyzing documents..."):
                llm = get_llm()
                retrieved_docs = vectorstore.similarity_search(doc_question, k=5)
                context_text = " ".join([doc.page_content for doc in retrieved_docs])
                prompt = create_doc_summary_prompt(doc_question, context_text)
                response = llm.invoke(prompt)
                st.write("### Document Analysis:")
                st.markdown(response.content)

    # Display UI for CSV files
    if csv_files:
        for csv_file in csv_files:
            st.subheader(f"Analyze `{csv_file.name}`:")
            temp_dir = "/tmp"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_file_path = os.path.join(temp_dir, csv_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(csv_file.getbuffer())

            try:
                df = pd.read_csv(temp_file_path)
                st.dataframe(df.head())
                
                # Clean up old plot file before running the agent
                plot_path = "plot.png"
                if os.path.exists(plot_path):
                    os.remove(plot_path)

                csv_agent_executor = create_csv_agent(
                    get_llm(),
                    temp_file_path,
                    verbose=True,
                    allow_dangerous_code=True
                )
                csv_question = st.text_input(f"e.g., Compare the total sales for each product using a bar chart.", key=csv_file.name)
                
                if csv_question:
                    with st.spinner(f"Performing analysis and generating visualizations..."):
                        enhanced_prompt = create_csv_analysis_prompt(csv_question)
                        response = csv_agent_executor.invoke({"input": enhanced_prompt})
                        
                        st.write("### Data Analysis:")
                        st.write(response['output'])
                        
                        # Check if a plot was generated and display it
                        if os.path.exists(plot_path):
                            st.image(plot_path)
                            
            except Exception as e:
                st.error(f"Error processing CSV file: {e}")
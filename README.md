# AI Business Consultant

## 📖 About The Project

This project is an advanced, AI-powered business consultant designed to demonstrate a full MLOps lifecycle, from initial concept to a deployed, production-ready system. The application functions as an intelligent agent that leverages Large Language Models (LLMs) to perform complex business analysis tasks. It can conduct live web research, analyze user-provided documents (`.pdf`, `.txt`, `.csv`), perform quantitative calculations, and generate professional data visualizations.

This repository showcases skills in:
* **LLM Agent Development:** Building autonomous agents with LangChain that can reason and use tools.
* **Full-Stack Implementation:** Creating a user-friendly front-end with Streamlit and a robust Python back-end.
* **ML System Design:** Architecting a system that is scalable, reliable, and ready for advanced MLOps practices like caching, monitoring, and automated evaluation.
* **Advanced Prompt Engineering:** Crafting sophisticated prompts to control LLM output for structured, high-quality analysis.

---

## ✨ Features

The application is organized into several modes, each providing a unique analytical capability:

#### 1. General Web Consultant
* Accepts any open-ended business question.
* Performs live web research using the Tavily Search API.
* Synthesizes the findings into a structured report, including an **Executive Summary**, **Key Findings**, and a **Concluding Insight**.

#### 2. SWOT Analysis
* Generates a comprehensive SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis for any given company.
* Conducts targeted research for each of the four components to produce a detailed and structured report.

#### 3. Document & Data Analysis
* **Analyze Private Documents:** Users can upload `.pdf` and `.txt` files for analysis. The system uses a RAG (Retrieval-Augmented Generation) pipeline to answer questions based on the document's content.
* **Perform Quantitative Analysis:** Users can upload `.csv` files and ask the agent to perform calculations (e.g., average, sum, count) by writing and executing Python `pandas` code.
* **Automated Data Visualization:** The agent can generate professional, aesthetically pleasing charts and plots using `seaborn` and `matplotlib` to visually answer user questions about CSV data.

---

## 🛠️ Technology Stack

* **Language:** Python
* **Core Frameworks:** LangChain, Streamlit
* **LLM & Embeddings:** Google Gemini API
* **Search Tool:** Tavily Search API
* **Data Processing:** Pandas, PyPDF
* **Vector Store:** FAISS (for RAG)
* **Data Visualization:** Matplotlib, Seaborn
* **Environment Management:** `venv`, `python-dotenv`

---

## 🚀 Setup & Installation

To run this project locally, please follow these steps:

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set up API keys:**
    * Create a `.env` file in the root of the project.
    * Add your API keys to the file:
        ```env
        GOOGLE_API_KEY="YOUR_GOOGLE_KEY_HERE"
        TAVILY_API_KEY="YOUR_TAVILY_KEY_HERE"
        ```

5.  **Run the application:**
    ```sh
    python -m streamlit run app.py
    ```

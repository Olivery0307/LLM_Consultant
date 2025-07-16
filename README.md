# AI Business Consultant

## 📖 About The Project

This project is an advanced, AI-powered business consultant designed to demonstrate a full MLOps lifecycle, from initial concept to a deployed, production-ready system. The application functions as an intelligent agent that leverages Large Language Models (LLMs) to perform complex business analysis tasks.

The system is architected to be robust, scalable, and efficient, incorporating key MLOps principles like **asynchronous task processing** to handle long-running jobs and a **caching layer** to reduce latency and API costs. It can conduct live web research, analyze user-provided documents (`.pdf`, `.txt`, `.csv`), perform quantitative calculations, and generate professional data visualizations.

This repository showcases skills in:

* **LLM Agent Development:** Building autonomous agents with LangChain that can reason and use tools.
* **Full-Stack Implementation:** Creating a user-friendly front-end with Streamlit and a robust Python back-end.
* **ML System Design & MLOps:** Architecting a system with caching, asynchronous workers, and a decoupled front-end for a professional user experience.
* **Object-Oriented Design:** Structuring the codebase into clean, maintainable, and scalable classes.
* **Advanced Prompt Engineering:** Crafting sophisticated prompts to control LLM output for structured, high-quality analysis.

## ✨ Features

The application is organized into several modes, each providing a unique analytical capability:

#### 1. Case Study Analysis (Lead Consultant)

* **The most advanced feature.** Users can define a central business question and upload a "data room" of multiple file types (`.pdf`, `.txt`, `.csv`).
* A high-level "Lead Consultant" agent autonomously delegates tasks to specialized sub-tools to perform a multi-source analysis.
* It synthesizes findings from internal documents and external web research into a single, comprehensive report.

#### 2. Conversational Web Consultant

* A chat agent with conversational memory that maintains the context of the discussion.
* Users can ask follow-up questions naturally without repeating themselves.
* The agent's persona is tailored to act as an expert business and financial consultant.

#### 3. Strategic Frameworks (One-Off Reports)

* **SWOT Analysis:** Generates a comprehensive SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis for any given company.
* **Competitor Analysis:** Identifies a company's main competitors and generates a comparative Markdown table with key metrics.

#### 4. System & MLOps Features

* **Asynchronous Task Queue (Celery & Redis):** Long-running analyses (like Case Studies) are executed in a background worker process. This prevents the UI from freezing and provides a smooth, non-blocking user experience.
* **LLM Caching Layer (Redis):** All calls to the LLM are cached. This dramatically reduces latency and API costs on repeated or similar queries, making the application feel much faster.

## 🛠️ Technology Stack

* **Language:** Python
* **Core Frameworks:** LangChain, Streamlit
* **LLM & Embeddings:** Google Gemini API
* **Search Tool:** Tavily Search API
* **Asynchronous Tasks:** Celery
* **Caching & Message Broker:** Redis
* **Data Processing:** Pandas, PyPDF
* **Vector Store:** FAISS (for RAG)
* **Data Visualization:** Matplotlib, Seaborn
* **Environment Management:** Docker, `venv`, `python-dotenv`

## 🚀 Setup & Installation

To run this project locally, a two-terminal setup is required to manage the web app and the background worker separately.

### Step 1: Clone the Repository

`git clone https://github.com/your-username/your-repo-name.git`
`cd your-repo-name`

### Step 2: Create and Activate a Virtual Environment

`python -m venv venv`
`source venv/bin/activate`  # On Windows, use `venv\Scripts\activate`

### Step 3: Install Dependencies

`pip install -r requirements.txt`

### Step 4: Set Up API Keys

* Create a `.env` file in the root of the project.
* Add your API keys to the file:
    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_KEY_HERE"
    TAVILY_API_KEY="YOUR_TAVILY_KEY_HERE"
    ```

### Step 5: Start Redis with Docker

* Make sure you have Docker Desktop installed and running.
* In any terminal, run this command to start the Redis server in the background:
    ```
    docker run -d -p 6379:6379 --name my-redis redis
    ```
    *(If it's already running, you can use `docker start my-redis`)*

### Step 6: Run the Application (Two Terminals)

**In your first terminal:**

1.  Activate the virtual environment (`source venv/bin/activate`).
2.  Start the Celery worker. This will listen for background jobs.
    ```
    celery -A tasks worker --loglevel=info
    ```
3.  Leave this terminal running.

**In your second terminal (e.g., a split terminal in VS Code):**

1.  Activate the virtual environment (`source venv/bin/activate`).
2.  Start the Streamlit web application.
    ```
    python -m streamlit run app.py
    ```

Your browser should now open with the AI Business Consultant application running.
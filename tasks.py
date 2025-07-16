from celery import Celery
import os
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agents.lead_consultant import LeadConsultantAgent, process_uploaded_files_for_tools

# Load environment variables for the worker
load_dotenv()

# Configure Celery
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Initialize the agent once when the worker starts
lead_agent = LeadConsultantAgent()

@celery_app.task
def run_case_study_task(case_question, uploaded_files_data):
    """
    A Celery task to run the long-running case study analysis in the background.
    """
    # Recreate file objects for processing, as they can't be passed directly.
    class UploadedFile:
        def __init__(self, name, data):
            self.name = name
            self._data = data
        def getbuffer(self):
            return self._data

    uploaded_files = [UploadedFile(f['name'], f['data']) for f in uploaded_files_data]

    # Process the files and run the analysis
    process_uploaded_files_for_tools(uploaded_files)
    report = lead_agent.run_case_study(case_question)
    return report
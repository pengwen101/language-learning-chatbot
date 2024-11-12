# RIASEC Assessment and Job Recommendation System
## Project Overview
This project enables users to take a Holland RIASEC personality test and receive job recommendations based on their personality type. After completing the test, users can interact with the system’s chatbot to explore suitable careers and job vacancies in real time.
### Key Features
RIASEC Personality Test: Users complete a questionnaire with statements they rate from 1 (strongly disagree) to 5 (strongly agree). The system calculates the user’s scores in the six RIASEC dimensions: Realistic, Investigative, Artistic, Social, Enterprising, and Conventional. Results are stored in a CSV file for further analysis.

RIASEC-Based Career Recommendation: Based on the top three RIASEC traits from the test results, users receive career suggestions that match their personality type. The system also includes relevant career information sourced from PDF documents and CSV files, with Qdrant providing vector-based storage for fast data retrieval.

Job Search with Chatbot Job-Tools: Users can search for job vacancies via an integrated chatbot that leverages FunctionTools and ReAct Agent from LlamaIndex. Job listings are fetched from Petra Alumni’s API and display details like job title, company, location, salary range, and job description.
### Requirements
Python 3.x
LlamaIndex, Qdrant, Streamlit, and other dependencies listed in requirements.txt
API key for accessing Petra Alumni's job data
### File Structure
```bash
riasec-assessment/
├── answers                   # CSV file storing user’s RIASEC assessment answers
├── cache                     # Cache directory for efficient data handling
├── docs                      # Documents for RIASEC career information
├── fetch                     # Directory for data fetching scripts
├── pages                     # Streamlit pages for user interaction
│   ├── 📃_1_RIASEC_Test.py   # Page for users to complete the RIASEC test
│   ├── 2_💼_RIASEC_Career_Recommendation.py # Page for RIASEC career recommendations
│   ├── 3_🔍_Search_Vacancies.py # Page for job search functionality
│   └── 📁_4_Upload_Files.py  # Page for users to upload additional reference files
├── .env.example              # Environment variable template file
├── README.md                 # Project documentation
└── requirements.txt          # Project dependencies
```
## Setup
Environment Configuration:

Create a .env file by copying .env.example.
Set up your API keys, especially for Petra Alumni’s job API.
Installation:

```bash
pip install -r requirements.txt
```

Run the Project: Launch the application with Streamlit:
```bash
streamlit run Hello.py # --server.port=xxxx --server.address=127.0.0.2
```

## Pages and Usage
1. RIASEC Test (`📃_1_RIASEC_Test.py`): Users take the RIASEC test and submit responses that are stored in answers/riasec_assessment_answer.csv.
2. Career Recommendations (`2_💼_RIASEC_Career_Recommendation.py`):
Displays suitable career paths based on the user’s top 3 RIASEC dimensions.
The chatbot uses context from LlamaIndex to provide relevant responses based on the RIASEC results.
Additional career-related information is retrieved from career-theory-model-holland-20170501.pdf and RIASEC Keywords.csv.
3. Job Search (`3_🔍_Search_Vacancies.py`):
The chatbot uses a ReAct Agent setup to fetch job listings from Petra Alumni’s API.
Users can specify criteria like province or salary range, and the chatbot will return relevant job openings.
Available tools:
`search_job_vacancy_tool`: Provides job listings based on keywords and location.
`get_job_vacancy_detail_tool`: Fetches detailed information for a specific job.
`get_province_id_tool`: Retrieves province ID to filter job search results by location.
4. File Uploads (📁_4_Upload_Files.py): Users can upload additional documents for embedding and future retrieval by the RIASEC chatbot.
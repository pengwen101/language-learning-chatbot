import streamlit as st
from swarm import Swarm, Agent
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
load_dotenv()
MODEL = "llama3.2:latest"
client = Swarm()

# get user results
riasec_result_data = pd.read_csv('./answers/riasec_assessment_answer.csv')
riasec_result_key_values = {row['Type']: row['Total Score'] for _, row in riasec_result_data.iterrows()}
top_3 = sorted(riasec_result_key_values.items(), key=lambda x: x[1], reverse=True)[:3]
top_3 = ', '.join([key for key, _ in top_3])

# search job from websites
def get_id_mh_province(provinces:str) -> str:
    """
    Use this tool if user specify the province, NOT the city. Returns province ID from user query. The province ID from this tool will be used for the tool 'search_job_vacancy'
    """
    r = requests.get('https://panel-alumni.petra.ac.id/api/province')
    data = r.json()
    for d in data['provinces']:
        if provinces.lower() == d['name'].lower():
            return d['id']
    return ""


def search_job_vacancy(keyword, start_salary:int = 500000, end_salary:int = 100000000, show_explaination:bool = True, id_mh_province:int = "") -> str:
    """
    Searches the Alumni Petra database for a list of job vacancies. If a province is specified, retrieve its ID first. Jobs are shown in a numbered list format, with an explanation if requested.
    """
    keywords = keyword.split(",")
    print("this is keywords ", keywords)
    for keyword in keywords:
        r = requests.get('https://panel-alumni.petra.ac.id/api/vacancy', {
            "page": 1,
            "type": "freelance,fulltime,parttime,internship",
            "system": "onsite,remote,hybrid",
            "level_education": "diploma,sarjana,magister,doktor",
            "keyword": keyword,
            "salary_range": str(start_salary) + ", " + str(end_salary),
            "id_mh_province": id_mh_province,
            "id_mh_city": "",
            "perPage": 3,
            "orderBy": "updated_at",
            "order": "DESC",
            "skills": "",
            "prody": "",
        })
    
        data = r.json()
        print("using alumbi API:", data)
        if len(data["vacancies"]["data"]) == 0:
            try:
                headers = {'apikey': os.getenv("APIJOB_API_KEY"), 'Content-Type': 'application/json'}
                response = requests.post('https://api.apijobs.dev/v1/job/search', headers=headers, json={"q": keyword})
                response.raise_for_status()
                data = response.json()
                print("using job API: ", data)
                jobs = data.get("hits", [])
                if not jobs:
                    return f"No jobs available for the keyword: {keyword}"
        
                output = f"## Job Results for '{keyword}'\n"
                for idx, job in enumerate(jobs[:5], 1):
                    output += f"""
                    **{idx}. {job['title']}**  
                    - **Company**: {job.get('hiringOrganizationName', 'N/A')}  
                    - **Language**: {job.get('language', 'N/A')}  
                    - **Description**: {job.get('description', 'N/A')}  
                    - **URL**: [Link]({job.get('url', 'N/A')})  
                    """
                return output
            except requests.RequestException as e:
                return f"Error fetching job data: {e}"
    
        output = f"SAY This is what I found on Alumni Petra with keyword {keyword}:\n"
        idx = 1
        for d in data["vacancies"]["data"]:
            salary_start = d['salary_start'] if d['salary_start'] is not None else ''
            salary_end = d['salary_end'] if d['salary_end'] is not None else ''
            salary_info = f"{salary_start} - {salary_end}" if salary_start or salary_end else 'Tidak ada informasi'
            output += f"""
                    {idx}. {d['position_name']} at {d['mh_company']['name']}
                    Lokasi: {d['mh_city']['name']}
                    Tipe: {d['type']}
                    Sistem: {d['system']}
                    Level Pendidikan: {d['level_education']}
                    Range Gaji: {salary_info}
                    Batas Apply: {d["expired_date"]}
                    Deskripsi: {BeautifulSoup(d['description'], 'html.parser').get_text()}
                    Job Requirements: {BeautifulSoup(d['requirement'], 'html.parser').get_text()}
            """
            idx += 1

    output += "\n\nShow this result to user directly with no summarization, and format it nicely."
    return output

job_searcher_agent = Agent(
    name="Job Searcher",
    instructions=f"""
    You are a job searcher. Your task is to:
    1. Use the user's RIASEC test results {top_3} to generate job keywords.
    2. Search for job vacancies using those keywords.
    3. Display the job title, company, location, and other relevant details with the correct format like this:
    Lokasi:\n
    Tipe:\n
    Sistem:\n
    Level Pendidikan:\n
    Range Gaji:\n
    Batas Apply:\n
    Deskripsi:\n
    Job Requirements:\n
    """,
    model=MODEL,
    functions=[get_id_mh_province, search_job_vacancy],
    verbose=True
)

# generate keywords
job_keyword_agent = Agent(
    name="Job Keyword Generator",
    instructions=f"""
    You are a job keyword expert. Your task is to:
    1. Take the user's top three RIASEC test results as input: {top_3}.
    2. Analyze the scores and types to generate relevant job-related keywords.
    3. Output the keywords as a comma-separated string.
    """,
    model=MODEL,
    verbose=True,
)

def transfer_to_job_searcher():
    return job_searcher_agent

def transfer_to_job_keyword():
    return job_keyword_agent

    
riasec_orchestrator_agent = Agent(
    name="RIASEC Career Consultation Orchestrator Agent",
    instructions=f"""
        This agent serves as the system's central hub and initial point of contact.
        It begins by thoroughly gathering information about the user's current career preferences.
        
        Once a comprehensive user preferences is established, the agent distributes relevant data
        to the specialized agents for job search and education content recommendation. 
        
        After receiving insights from other agents, it synthesizes the information to provide cohesive,
        personalized career advice. The Orchestrator prioritizes tasks, manages inter-agent communication,
        to give easy-to-understand guidance for the user.
    """,
    functions=[transfer_to_job_searcher],
    model=MODEL,
    verbose=True,
)

if "messages_job" not in st.session_state:
    st.session_state.messages_job = [
        {"role": "assistant",
         "content": "Halo! What job do you want to search for? 😊"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages_job:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages_job.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_stream = client.run(agent=riasec_orchestrator_agent, messages=[{"role": "user", "content": prompt}])
            st.markdown(response_stream.messages[-1]["content"])

    # Add user message to chat history
    st.session_state.messages_job.append({"role": "assistant", "content": response_stream.messages[-1]["content"]})
import streamlit as st
from duckduckgo_search import DDGS
from swarm import Swarm, Agent
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import requests
from bs4 import BeautifulSoup
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
import csv


load_dotenv()
MODEL = "llama3.2:latest"
client = Swarm()

# st.set_page_config(page_title="Educational Content Recommender", page_icon="📰")
# st.title("📰 Educational Content Recommender")

def search_job_vacancy_riasec(keyword: str = "") -> str:
    print(f"{keyword} is being searched\n")
    """
    Searches the Alumni Petra database for a list of job vacancies. 
    You should specify a keyword for the job position and search for the jobs. 
    Jobs are shown in a numbered list format, with an explanation if requested.
    """
    r = requests.get('https://panel-alumni.petra.ac.id/api/vacancy', {
        "page": 1,
        "type": "freelance,fulltime,parttime,internship",
        "system": "onsite,remote,hybrid",
        "level_education": "diploma,sarjana,magister,doktor",
        "keyword": keyword,
        # "salary_range": str(start_salary) + ", " + str(end_salary),
        "salary_range": "",
        # "id_mh_province": id_mh_province,
        "id_mh_province": "",
        "id_mh_city": "",
        "perPage": 3,
        "orderBy": "updated_at",
        "order": "DESC",
        "skills": "",
        "prody": "",
    })

    data = r.json()
    if len(data["vacancies"]["data"]) == 0:
        return "No jobs available for your query."

    output = "SAY This is what I found on Alumni Petra:\n"
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
        """
        idx += 1

    output += "\n\nShow this result to user directly with no summarization, and format it nicely."
    return output

search_job_vacancy_riasec_tool = FunctionTool.from_defaults(async_fn=search_job_vacancy_riasec)

def getBestSlugs():
    print("view csv fn called")
    with open('pages/slug.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        output = ""
        for row in spamreader:
            output += f"{row}\n"
        return output

def searchJob(slug: str):
    print(f"Searching for slug {slug}")
    

# Create specialized agents

slug_agent = Agent(
    name="Slug searcher",
    instructions="""
        Your job is to pick 5 best slugs from the file available at getBestSlugs function.
        You must search the best slugs according to current conversation.
        DONT pass any argument when calling a function
        ONLY output the SLUG in comma seperated list (seperator=', '), DONT add anything INCLUDING star/special chars
    """,
    functions=[getBestSlugs],
    model=MODEL,
)

job_agent = Agent(
    name="Job searcher",
    instructions="""
        Your job is to search for a job available at searchJob function.
        You MUST pass 1 job slug to the searchJob function everytime you are called.
        You MUST call 5 times according to the latest conversation.
        Please pass the slug accordingly and remove redundant information.
    """,
    functions=[searchJob],
    model=MODEL,
)

def return_slug(keyword):
    slug_response = client.run(
        agent=slug_agent,
        messages=[{"role": "user", "content": f"{keyword}"}]
    )
    return slug_response.messages[-1]['content'].split(", ")
    

print(getBestSlugs())

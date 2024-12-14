import streamlit as st
from swarm import Swarm, Agent
from dotenv import load_dotenv
import pandas as pd
from duckduckgo_search import DDGS
from datetime import datetime
import requests
import os
from bs4 import BeautifulSoup
import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

# Load environment variables
load_dotenv()

# Initialize Swarm client
client = Swarm()
MODEL = "llama3.2:latest"

# st.set_page_config(page_title="Job Search") # , page_icon="📰"
# st.title("📰 Job Search Recommender")

# Initialize chat history if empty
# if "messages_job" not in st.session_state:
#     st.session_state.messages_job = [
#         {"role": "assistant", "content": "Do you want to search job based on your riasec personality? 😊"}
#     ]

# RIASEC result processing
riasec_result_data = pd.read_csv('./answers/riasec_assessment_answer.csv')
riasec_result_key_values = {
    row['Type']: row['Total Score'] for _, row in riasec_result_data.iterrows()
}
top_3 = sorted(riasec_result_key_values.items(), key=lambda x: x[1], reverse=True)[:3]
top_3_riasec = [item[0] for item in top_3]
# print(keywords)

async def search_job_vacancy_riasec(keyword: str = "", start_salary:int = 500000, end_salary:int = 100000000, show_explaination:bool = True, id_mh_province:int = "") -> str:
    """
    Searches the Alumni Petra database for a list of job vacancies. If a province is specified, retrieve its ID first. Jobs are shown in a numbered list format, with an explanation if requested.
    """
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

    if len(data["vacancies"]["data"]) == 0:
        return "No jobs available for your query."

    output = "This is what I found on Alumni Petra:\n"
    idx = 1

    for d in data["vacancies"]["data"]:
        salary_start = d['salary_start'] if d['salary_start'] is not None else ''
        salary_end = d['salary_end'] if d['salary_end'] is not None else ''
        salary_info = f"{salary_start} - {salary_end}" if salary_start or salary_end else 'Tidak ada informasi'
        
        # Check if the 'requirement' field is valid before using BeautifulSoup
        requirement_text = BeautifulSoup(d['requirement'], 'html.parser').get_text() if d.get('requirement') else 'No requirement information available'
        description = d.get('description', '')  # Use an empty string if 'description' is None

    if description:  # Only process if description is not empty
        deskripsi = BeautifulSoup(description, 'html.parser').get_text()
    else:
        deskripsi = "No description available"  # Default message if description is None


        if show_explaination:
            output += f"""
                {idx}. {d['position_name']} at {d['mh_company']['name']}
                Lokasi: {d['mh_city']['name']}
                Tipe: {d['type']}
                Sistem: {d['system']}
                Level Pendidikan: {d['level_education']}
                Range Gaji: {salary_info}
                Batas Apply: {d["expired_date"]}
                Deskripsi: {deskripsi}
                Job Requirements: {requirement_text}
            """
        else:
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

# Function to search for job vacancies
def search_job_vacancy(keyword):
    """Search for job vacancies using a given keyword."""
    try:
        headers = {'apikey': os.getenv("APIJOB_API_KEY"), 'Content-Type': 'application/json'}
        response = requests.post('https://api.apijobs.dev/v1/job/search', headers=headers, json={"q": keyword})
        response.raise_for_status()
        data = response.json()
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

def search_educational_content(topic):
    """Search for educational content using DuckDuckGo."""
    with DDGS() as ddg:
        results = ddg.text(f"educational content for {topic} {datetime.now().strftime('%Y-%m')}", max_results=3)
        if results:
            educational_content = "\n\n".join([
                f"Title: {result['title']}\nURL: {result['href']}\nSummary: {result['body']}" 
                for result in results
            ])
            return educational_content
        return f"No educational content found for {topic}."


job_keyword_agent = Agent(
    name="Job Keyword Generator",
    instructions="""
    You are a job keyword expert. Your task is to:
    1. Take the user's top three RIASEC test results as input.
    2. Analyze the scores and types to generate relevant job-related keywords.
    3. Output the keywords as a comma-separated string.
    """,
    model=MODEL,
)

topic_generator_agent = Agent(
    name="Topic Generator",
    instructions="""
    You are a topic generation expert. Your task is to:
    1. Take a user's RIASEC test result as input.
    2. Identify the most relevant occupations suited for the given RIASEC profile.
    3. Generate a topic for educational content that would help the user prepare for those occupations.
    Output the topic in a single sentence, focusing on specific skills, knowledge, or training areas.
    """,
    model=MODEL
)

search_agent = Agent(
    name="Educational Content Searcher",
    instructions="""
    You are an educational content specialist. Your task is to:
    1. Search for the most relevant and recent educational content for the given topic.
    2. Ensure the results are from reputable sources.
    3. Return the raw search results in a structured format.
    """,
    functions=[search_educational_content],
    model=MODEL
)

synthesis_agent = Agent(
    name="Content Synthesizer",
    instructions="""
    You are a synthesis expert. Your task is to:
    1. Analyze the raw educational content provided.
    2. Identify key themes and important information.
    3. Combine information from multiple sources.
    4. Create a comprehensive but concise synthesis.
    5. Focus on facts and maintain clarity.
    Provide a 2-3 paragraph synthesis of the main points.
    """,
    model=MODEL
)

summary_agent = Agent(
    name="News Summarizer",
    instructions="""
    You are an expert educational content summarizer, blending clarity and conciseness with actionable insights, while ensuring absolute accuracy and reliability.  

    Your task:  
    1. Core Information:  
       - Highlight the key skills, knowledge areas, or certifications related to the topic.  
       - Include practical steps or resources for acquiring these skills, only if they are verified as accurate and trustworthy.  
       - Cross-check and ensure that any URLs, data, or resources mentioned are from credible and reputable sources.  
       - Explain the relevance of this content to career advancement or job preparation.  
       - Provide essential data, such as demand for the skill or related industries, only if sourced from valid references.
       - Provide the URLs from all resources that are mentioned
    
    2. Style Guidelines:  
       - Use strong, active verbs.  
       - Be clear, precise, and avoid jargon unless necessary (with explanations).  
       - Maintain objectivity while engaging the reader.  
       - Make every word purposeful, avoiding redundancy.  
       - Keep a professional yet approachable tone.  
       - Emphasize factual integrity and avoid any assumptions or speculative statements.  
    
    Format: Create a single paragraph of 250-400 words that informs, inspires, and guides the reader.  
    Pattern: [Key Educational Focus] + [Details and Practical Resources] + [Why It Matters/Next Steps] + [Sources' URLs]
    
    Focus on answering: What skills or knowledge are essential? How can the reader gain them? Why are they valuable in the job market? Provide also the URLs from credible sources.
    
    IMPORTANT: Never fabricate or misrepresent information. Only include details that are explicitly found in the provided content or are general knowledge. Omit or disclaim unverified or unclear information. Start directly with the content, avoiding introductory phrases, labels, or meta-text like "Here's a summary" or "In educational content style."
    """,
    model=MODEL
)


async def get_keywords_from_riasec_result(top_3: list) -> list:
    # Prepare the input for the agent
    riasec_types_str = ", ".join(top_3)
    message = {
        "role": "user",
        "content": f"The user's top three RIASEC results are: {riasec_types_str}. Please generate relevant job title. Also make sure the generated keywords is formatted like 'keyword1, keyword2, keyword3, etc'. Only return me the keywords do not use any bracket or what so ever"
    }

    # Use the agent to generate keywords
    response = client.run(agent=job_keyword_agent, messages=[message])
    return response.messages[0]['content']


def process_educational_content(riasec_result):
    """Run the educational content processing workflow."""
    # with st.status("Processing educational content...", expanded=True) as status:
    # Generate Topic
    # status.write("🔄 Generating topic based on RIASEC result...")
    topic_response = client.run(
        agent=topic_generator_agent,
        messages=[{"role": "user", "content": f"Generate a topic for RIASEC result: {riasec_result}"}]
    )
    topic = topic_response.messages[-1]["content"]
    
    # Search
    # status.write("🔍 Searching for educational content...")
    search_response = client.run(
        agent=search_agent,
        messages=[{"role": "user", "content": f"Find educational content about {topic}"}]
    )
    raw_content = search_response.messages[-1]["content"]
    
    # Synthesize
    # status.write("🔄 Synthesizing information...")
    synthesis_response = client.run(
        agent=synthesis_agent,
        messages=[{"role": "user", "content": f"Synthesize this educational content:\n{raw_content}"}]
    )
    synthesized_content = synthesis_response.messages[-1]["content"]
    
    # Summarize
    # status.write("📝 Creating summary...")
    summary_response = client.run(
        agent=summary_agent,
        messages=[{"role": "user", "content": f"Summarize this synthesis:\n{synthesized_content}"}]
    )
    return topic, raw_content, synthesized_content, summary_response.messages[-1]["content"]


async def main():
    keywords = await get_keywords_from_riasec_result(top_3_riasec)
    print("Generated Job Keywords:", keywords)
    keywords = [key.strip("[]") for key in keywords.split(',')]
    noJob = False
    results = []
    for key in keywords:
        result = await search_job_vacancy_riasec(key)
        if "No jobs available for your query" in result:
            noJob = True
            continue
        results.append(result)
    # print(result)
    if noJob:
        print("There is no job vacancy open on Alumni Petra's site, searching in APIJobs...")
        result = search_job_vacancy(keywords[0])
        print(result)

    topic, raw_content, synthesized_content, final_summary = process_educational_content(riasec_result_data)
    print(final_summary)


# Run the main function
asyncio.run(main())
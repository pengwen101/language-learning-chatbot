import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core import PromptTemplate
from bs4 import BeautifulSoup
import pandas as pd
import nest_asyncio
import logging
import requests
import sys
from duckduckgo_search import DDGS
from datetime import datetime
nest_asyncio.apply()

riasec_result_data = pd.read_csv('./answers/riasec_assessment_answer.csv')
riasec_result_key_values = [{row['Type']: row['Total Score']} for index, row in riasec_result_data.iterrows()]
top_3 = sorted(riasec_result_key_values, key=lambda x: list(x.values())[0], reverse=True)[:3]

system_prompt = system_prompt = """
You are a multi-lingual career advisor expert who has knowledge based on 
real-time data. You will always try to be helpful and try to help them 
answering their questions. If you don't know the answer, say that you DON'T 
KNOW.

Your primary job is to assist users with two tasks: 
1. Helping them find jobs from the API Jobs Developer database.
2. Recommending educational content based on their RIASEC test results.

For **job search**, you create a list of keywords from the user's RIASEC result and find matching job postings in the database. You must display all the jobs available retrieved from the tools and NOT summarize them. Elaborate on all information you get from the tools. You should explain WHY the jobs MATCH the user's personality results. 

When presenting job matches, you MUST include the following details:  
1. The position name of the job and the type, such as full-time, part-time, etc.  
2. The location of the job.  
3. The system of the job, such as onsite, remote, or hybrid.  
4. The minimum degree required to apply for the job.  
5. The salary range of the job.  
6. The job application deadline.  
7. The description of the job.  
8. The requirements for the job.  

For **educational content recommendations**, use the RIASEC results to suggest suitable courses, certifications, or resources. You must include:  
1. The course or resource name.  
2. The platform or institution offering it.  
3. The type of learning material (e.g., video tutorials, certifications, books).  
4. The duration or time required to complete it.  
5. The cost (if available).  
6. A brief description of the content.  
7. Why the content matches the user's personality type or career aspirations.  

### Example for Job Search:
User: I would like to search for a job in finance.  
Assistant: Sure! Here are some job vacancies related to finance:  
1. **Account Finance Manager** in Surabaya.  
   - **Type**: Full-time  
   - **System**: Hybrid  
   - **Minimum Degree**: Bachelor's in Finance  
   - **Salary Range**: IDR 10,000,000 - 15,000,000/month  
   - **Application Deadline**: 30th November 2024  
   - **Description**: Oversee financial reporting and budgeting, ensuring compliance with corporate and regulatory requirements.  
   - **Requirements**: Minimum 3 years in finance management, proficiency in ERP systems.  

### Example for Educational Content:  
User: My RIASEC result shows I’m high in Investigative and Artistic.  
Assistant: Based on your results, here are some educational recommendations:  
1. **Course: "Data Visualization for Storytelling"**  
   - **Platform**: Coursera  
   - **Type**: Online video tutorials and certification  
   - **Duration**: 6 weeks (4 hours/week)  
   - **Cost**: Free to audit; $49 for certification  
   - **Description**: Learn how to combine data analytics and artistic principles to create compelling visualizations.  
   - **Why It Matches**: This course aligns with your Investigative and Artistic traits by combining analytical and creative skills.  

You MUST display the information in the specified formats only. DO NOT summarize or omit details.  

"""

react_system_header_str = """\

## Tools
You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.  

For job search, you can use the following tools:  
1. `alumni_job_tool`: Prioritize this tool to find job opportunities on the Alumni Petra website.  
2. `search_job_vacancy_tool`: Use this tool if `alumni_job_tool` does not yield results.  

For educational recommendations, use the `education_recommender_tool`.  

Always specify which tool you used and provide detailed outputs as per the system prompt.  

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
in the one of the following two formats:

```
Thought: I can answer without using any more tools.
Answer: [your answer here]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: Sorry, I cannot answer your query.
```

## Additional Rules
- You MUST obey the function signature of each tool. Do NOT pass in no arguments if the function expects arguments.

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

"""
react_system_prompt = PromptTemplate(react_system_header_str)

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

Settings.llm = Ollama(model="llama3.1:8b-instruct-q4_0", base_url="http://127.0.0.1:11434", system_prompt=system_prompt, temperature=0)
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")


# Declare Tools
# function tools    

async def get_id_mh_province(provinces:str) -> str:
    """
    Use this tool if user specify the province, NOT the city. Returns province ID from user query. The province ID from this tool will be used for the tool 'search_job_vacancy'
    """
    r = requests.get('https://panel-alumni.petra.ac.id/api/province')
    data = r.json()
    for d in data['provinces']:
        if provinces.lower() == d['name'].lower():
            return d['id']
    return ""

async def get_keywords_from_riasec_result() -> str:
    """
    Provides ONE keyword matching with user's personality. The keyword MUST BE for job search. Use this as the parameter 'keyword' used in other tool. 
    """
    riasec_docs = SimpleDirectoryReader(input_dir='./docs/').load_data()
    index = VectorStoreIndex.from_documents(riasec_docs)
    query_engine = index.as_query_engine()
    query = f"Return only ONE KEYWORD that is suited for JOB SEARCH that match with {top_3[0]}, {top_3[1]}, {top_3[2]} personality"
    response = query_engine.query(query)
    
    return response

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

    output = "SAY This is what I found on Alumni Petra:\n"
    idx = 1
    for d in data["vacancies"]["data"]:
        salary_start = d['salary_start'] if d['salary_start'] is not None else ''
        salary_end = d['salary_end'] if d['salary_end'] is not None else ''
        salary_info = f"{salary_start} - {salary_end}" if salary_start or salary_end else 'Tidak ada informasi'
        
        if show_explaination:
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


async def search_educational_content():
    """Search for educational content based on user RIASEC test result using DuckDuckGo."""
    with DDGS() as ddg:
        results = ddg.text(f"educational content for {top_3} {datetime.now().strftime('%Y-%m')}", max_results=3)
        if results:
            educational_content = "\n\n".join([
                f"Title: {result['title']}\nURL: {result['href']}\nSummary: {result['body']}" 
                for result in results
            ])
            return educational_content
        return f"No educational content found for {top_3}."

    
async def search_job_vacancy(keyword: str) -> str:
    """
        Searches the API Jobs Developer database for A LIST OF (ONE OR MORE THAN ONE) matching job vacancy entries. Each job should be shown in a numbered list format. Keyword should be configured to one relevant word that MUST represents the job name or position in ENGLISH. 
    """
    
    headers = {
        'apikey': 'c1f4d885281ebe8b65295a84df1f07b253ae56ad68a5d48a5fc93604ce269e02',  
        'Content-Type': 'application/json',
    }

    data = {
        "q": keyword,
        # "country": "indonesia",
    }

#     try:
#     # Make a POST request to the Flask API
#     response = requests.post(url, json=params)
    
#     # Check if the request was successful
#     if response.status_code == 200:
#         # Parse the JSON response
#         data = response.json()
#         print("Data from API:", json.dumps(data, indent=4))
#     else:
#         print(f"Failed to fetch data: {response.status_code}")
#         print(response.json())  # Display error details if any
# except requests.exceptions.RequestException as e:
#     print(f"An error occurred: {e}")
    url = "http://127.0.0.1:5000/api/fetch"
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        response = requests.post('https://api.apijobs.dev/v1/job/search', headers=headers, json=data)
        data = response.json()
    # response = requests.post('https://api.apijobs.dev/v1/job/search', headers=headers, json=data)
    # data = response.json()

    output = f"# Job results for '{keyword}'\n"
    jobs = data.get("hits", [])
    
    if not jobs:
        return "No jobs available for your query."

    # Formatting each job listing
    for idx, job in enumerate(jobs[:5], 1):  # Limiting to 5 results
        output += f"""
        {idx}. {job['title']} at {job.get('hiringOrganizationName', 'N/A')}
        Language: {job.get('language', 'N/A')}
        Description: {job.get('description', 'N/A')}
        Website URL: {job.get('url', 'N/A')}
        """

    output += "\n\n Show this result to user DIRECTLY, with NO summarization. Make sure to ALWAYS SHOW the WEBSITE URL. If it returns nothing, say to user that NO JOBS are available for user query."
    
    return output
   

    

# async def get_job_vacancy_slug_detail(slug: str) -> str:
#     """
#         Provides detailed information regarding the vacancy. slug must be a specific vacancy slug.
#     """
    
#     r = requests.get(f"https://panel-alumni.petra.ac.id/api/vacancy/{slug}")
#     data = r.json()["vacancy"]
#     salary_start = data['salary_start'] if data['salary_start'] is not None else ''
#     salary_end = data['salary_end'] if data['salary_end'] is not None else ''
#     if(not salary_start and not salary_end):
#         salary_info = 'Tidak ada informasi'
#     else:
#         salary_info = str(salary_start) + " - " + str(salary_end)
    
#     return f"Pekerjaan ini adalah sebagai {data['position_name']} di {data['mh_company']['name']} di kota {data['mh_city']['name']} dengan sistem {data['system']} dan tipe {data['type']} dengan range gaji {salary_info}. Untuk apply, anda harus memiliki level pendidikan {data['level_education']}. Di dalam pekerjaan ini user akan mengerjakan beberapa job description, yaitu: {BeautifulSoup(data['description'], 'html.parser').get_text()}. Untuk mendaftar ke pekerjaan ini, user harus memiliki requirements sebagai berikut: {BeautifulSoup(data['requirement'], 'html.parser').get_text()}. Batas apply ke pekerjaan ini adalah {data['expired_date']}"

    
search_job_vacancy_tool = FunctionTool.from_defaults(async_fn=search_job_vacancy) 
get_keywords_from_riasec_result_tool = FunctionTool.from_defaults(async_fn=get_keywords_from_riasec_result)
alumni_job_tool = FunctionTool.from_defaults(async_fn=search_job_vacancy_riasec)
educational_content_tool = FunctionTool.from_defaults(async_fn=search_educational_content)
# get_job_vacancy_detail_tool = FunctionTool.from_defaults(async_fn=get_job_vacancy_slug_detail) 
get_province_id_tool = FunctionTool.from_defaults(async_fn=get_id_mh_province) 


tools = [alumni_job_tool, search_job_vacancy_tool, get_keywords_from_riasec_result_tool, get_province_id_tool, educational_content_tool]


# Main Program
st.title("Search for Jobs On Alumni Website! 🔍")

# Initialize chat history if empty
if "messages_job" not in st.session_state:
    st.session_state.messages_job = [
        {"role": "assistant",
         "content": "Halo! What job do you want to search for? 😊"}
    ]

# Initialize the chat engine
if "chat_engine_job" not in st.session_state.keys():
    # Initialize with custom chat history
    init_history = [
        ChatMessage(role=MessageRole.ASSISTANT, content="Halo! Mau cari lowongan pekerjaan apa?"),
    ]
    memory = ChatMemoryBuffer.from_defaults(token_limit=32768)

    st.session_state.chat_engine_job = ReActAgent.from_tools(
        tools,
        chat_mode="react",
        verbose=True,
        memory=memory,
        react_system_prompt=react_system_prompt,
        # retriever=retriever,
        llm=Settings.llm
    )

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
            response_stream = st.session_state.chat_engine_job.stream_chat(prompt)
            st.write_stream(response_stream.response_gen)

    # Add user message to chat history
    st.session_state.messages_job.append({"role": "assistant", "content": response_stream.response})
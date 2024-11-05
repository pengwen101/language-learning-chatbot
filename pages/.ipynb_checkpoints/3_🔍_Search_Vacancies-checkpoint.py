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
nest_asyncio.apply()

system_prompt = """
You are a multi-lingual career advisor expert who has knowledge based on 
real-time data. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.

Your primary job is to help people to find jobs from Alumni Petra job vacancy database. You should display all the jobs available retrieved from the tools and NOT summarize them. Elaborate all informations you get from the tools.


When a user asks about possible jobs, you MUST mention all of the jobs details such as:
1. The position name of the job and the type like full time, etc
2. The location of the job
3. The system of the job like onsite or hybrid
4. The minimum degree to apply for the job
5. The salary range of the job
6. The job application due date
7. The description of the job
8. The requirements of the job

Here is a short example:
User: I would like to search a job in finance
Assistant: Sure! Here are come job vacancies related to finance:
1. Account Finance Manager di Kota Surabaya. 
Tipe: <Type of the job>
Sistem: <System of the job>
Level Pendidikan = <Minimum degree to apply for the job>
Range Gaji = <Salary range of the job>
Batas Apply = <Job application due date>
Description: <Elaborate the job description>
Requirements: <Elaborate the job requirements>

You MUST display the job with above format only. DO NOT display in any other format.
"""

react_system_header_str = """\

## Tools
You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.

You have access to the following tools:
{tool_desc}

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
            
    
async def search_job_vacancy(keyword: str = "", start_salary:int = 500000, end_salary:int = 100000000, show_explaination:bool = True, id_mh_province:int = "") -> str:
    """
    Searches the Alumni Petra database for A LIST OF (ONE OR MORE THAN ONE) matching job vacancy entries. If the user specifies the province, you MUST GO TO OTHER TOOL TO GET province id first. Each job should be shown in a numbered list format. Keyword should be configured to one to three relevant words that MUST represents the job name or position. 
    
    show_explaination by default MUST be true, except if the user wanted the details of the job it should be false.
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
        "perPage": 5,
        "orderBy": "updated_at",
        "order": "DESC",
        "skills": "",
        "prody": "",
    })

    data = r.json()
    output = f"# Job results for '{keyword}'"
    idx = 1
    for d in data["vacancies"]["data"]:
        salary_start = d['salary_start'] if d['salary_start'] is not None else ''
        salary_end = d['salary_end'] if d['salary_end'] is not None else ''
        if(not salary_start and not salary_end):
            salary_info = 'Tidak ada informasi'
        else:
            salary_info = str(salary_start) + " - " + str(salary_end)
        if show_explaination:
            output += f"""
                        {idx}. {d['position_name']} at {d['mh_company']['name']}
                        Lokasi = {d['mh_city']['name']}
                        Tipe = {d['type']}
                        Sistem = {d['system']}
                        Level Pendidikan = {d['level_education']}
                        Range Gaji = {salary_info}
                        Batas Apply = {d["expired_date"]}
                        Deskripsi = {BeautifulSoup(d['description'], 'html.parser').get_text()}
                        Job Requirements = {BeautifulSoup(d['requirement'], 'html.parser').get_text()}
                        slug = {d['slug']} (DO NOT show this to the user)
                        """
        else:
            output += f"""
                        {idx}. {d['position_name']} at {d['mh_company']['name']}
                        Lokasi = {d['mh_city']['name']}
                        Tipe = {d['type']}
                        Sistem = {d['system']}
                        Level Pendidikan = {d['level_education']}
                        Range Gaji = {salary_info}
                        Batas Apply = {d["expired_date"]}
                        slug = {d['slug']} (DO NOT show this to the user)
                        """
        
        idx+=1
    if len(data["vacancies"]["data"]) ==0:
        output += "No results found."

    output += "\n\n Show this result to user DIRECTLY, with NO summarization but FORMAT IT NICELY. If it returns nothing, say to user that NO JOBS are available for user query."
    
    return output
    
    

async def get_job_vacancy_slug_detail(slug: str) -> str:
    """
        Provides detailed information regarding the vacancy. slug must be a specific vacancy slug.
    """
    
    r = requests.get(f"https://panel-alumni.petra.ac.id/api/vacancy/{slug}")
    data = r.json()["vacancy"]
    salary_start = data['salary_start'] if data['salary_start'] is not None else ''
    salary_end = data['salary_end'] if data['salary_end'] is not None else ''
    if(not salary_start and not salary_end):
        salary_info = 'Tidak ada informasi'
    else:
        salary_info = str(salary_start) + " - " + str(salary_end)
    
    return f"Pekerjaan ini adalah sebagai {data['position_name']} di {data['mh_company']['name']} di kota {data['mh_city']['name']} dengan sistem {data['system']} dan tipe {data['type']} dengan range gaji {salary_info}. Untuk apply, anda harus memiliki level pendidikan {data['level_education']}. Di dalam pekerjaan ini user akan mengerjakan beberapa job description, yaitu: {BeautifulSoup(data['description'], 'html.parser').get_text()}. Untuk mendaftar ke pekerjaan ini, user harus memiliki requirements sebagai berikut: {BeautifulSoup(data['requirement'], 'html.parser').get_text()}. Batas apply ke pekerjaan ini adalah {data['expired_date']}"

    
search_job_vacancy_tool = FunctionTool.from_defaults(async_fn=search_job_vacancy) 
get_job_vacancy_detail_tool = FunctionTool.from_defaults(async_fn=get_job_vacancy_slug_detail) 
get_province_id_tool = FunctionTool.from_defaults(async_fn=get_id_mh_province) 


tools = [search_job_vacancy_tool, get_job_vacancy_detail_tool, get_province_id_tool]


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

    print(st.session_state.chat_engine_job.get_prompts())

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

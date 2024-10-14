import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.file import CSVReader
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent
from typing import Optional
from llama_index.core import PromptTemplate
from bs4 import BeautifulSoup
import pandas as pd

import nest_asyncio
nest_asyncio.apply()

# initialize node parser
splitter = SentenceSplitter(chunk_size=512)

system_prompt = """
You are a multi-lingual career advisor expert who has knowledge based on 
real-time data. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.

Your primary job is to help people to find jobs related to their interest from Alumni Petra job vacancy database. You should display all the jobs available retrieved from the tools and NOT summarize them. Elaborate all informations you get from the tools.


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

# If the user asks about the activity in detail, you should at least mention:
# 1. The mitra which hosts the activity and the name of the activity.
# 2. A summary of what the activity is about.
# 3. When it is held.
# 4. Whether it is eligible to be converted into university credits, if so, how many.

# Here is a short example:
# User: I would like to study Machine Learning.
# Assistant: Sure! I found a few results related to Machine Learning:
# 1. Bangkit Academy, held by Google, Online.
# 2. AI & Machine Learning, held by MMT at Jakarta.
# User: I'd like to learn more about Bangkit Academy.
# Assistant: Sure! <elaborate about bangkit academy here>

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

import sys

import logging
import requests

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

Settings.llm = Ollama(model="llama3.1:8b-instruct-q4_0", base_url="http://127.0.0.1:11434", system_prompt=system_prompt, temperature=0)
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")

# Main Program
st.title("Search for jobs")

# Initialize chat history if empty
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Halo! Mau cari job apa?"}
    ]

# Declare Tools
# function tools    
async def search_job_vacancy(keyword: str, start_salary:int = 0, end_salary:int = 10000000, show_explaination:bool = True) -> str:
    """
    Searches the Alumni Petra database for A LIST OF (ONE OR MORE THAN ONE) matching job vacancy entries. Each job should be shown in a numbered list format. Keyword should be configured to one to three relevant words that MUST represents the job name or position. start_salary represent the minimal monthly salary in IDR (Indonesian Rupiah) IF AND ONLY IF user type a specific nominal, otherwise the default value MUST be 0. end_salary represent the maximal monthly salary in IDR (Indonesian Rupiah) IF AND ONLY IF user type a specific nominal, otherwise the default value MUST be 1000000000. show_explaination by default MUST be true, except if the user wanted the details of the job it should be false.
    """

    r = requests.get('https://panel-alumni.petra.ac.id/api/vacancy', {
        "page": 1,
        "type": "",
        "system": "",
        "level_education": "diploma,sarjana,magister,doktor",
        "keyword": keyword,
        "salary_range": str(start_salary) + ", " + str(end_salary),
        "id_mh_province": "",
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
            salary_info = str(salary_start) + " - " + str(salary_end) + " juta"
        if show_explaination:
            output += f"""
                        {idx}. {d['position_name']} at {d['mh_company']['name']}
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
                        Tipe = {d['type']}
                        Sistem = {d['system']}
                        Level Pendidikan = {d['level_education']}
                        Range Gaji = {salary_info}
                        Batas Apply = {d["expired_date"]}
                        slug = {d['slug']} (DO NOT show this to the user)
                        """
        
        idx+=1
    if len(data["vacancies"]["data"]) == 0:
        output += "No results found."

    output += "\n\n Show this result to user DIRECTLY, with NO summarization"
    
    return output



async def get_riasec_personality_test_result() -> str:
    """
    Provides information regarding user's personality. Use this to give better recommendations on job vacancies that were previously asked by the user. DO NOT SHOW NEW JOB LISTINGS. Only show the ones that had been shown before to the user. 
    """
    riasec_result_data = pd.read_csv('riasec_assessment_answer.csv')
    riasec_result_key_values = [{row['Type']: row['Total Score']} for index, row in riasec_result_data.iterrows()]
    top_3 = sorted(riasec_result_key_values, key=lambda x: list(x.values())[0], reverse=True)[:3]
    riasec_docs = SimpleDirectoryReader(input_dir='../docs/').load_data()
    index = VectorStoreIndex.from_documents(riasec_docs)
    query_engine = index.as_query_engine()
    query = f"What are the characteristics and career recommendations for someone having {top_3[0]}, {top_3[1]}, {top_3[2]}?"
    response = query_engine.query(query)

    return response
    
    

async def get_job_vacancy_slug_detail(slug: str) -> str:
    """
        Provides detailed information regarding the vacancy. slug must be a specific vacancy slug.
    """
    
    r = requests.get(f"https://panel-alumni.petra.ac.id/api/vacancy/{slug}")
    data = r.json()["vacancy"]
    
    return f"Pekerjaan ini adalah sebagai {data['position_name']} di {data['mh_company']['name']} di kota {data['mh_city']['name']} dengan sistem {data['system']}. Di dalam pekerjaan ini user akan mengerjakan beberapa job description, yaitu: {BeautifulSoup(data['description'], 'html.parser').get_text()}. Untuk mendaftar ke pekerjaan ini, user harus memiliki requirements sebagai berikut: {BeautifulSoup(data['requirement'], 'html.parser').get_text()}."


# async def get_answer_from_file(keyword: str) -> list[str]:
#     """Use this tool as your ANSWER"""
    
#     read_file = f'job_results_for_{keyword}.txt'
#     with open(read_file, 'r') as file:
#         content = file.read(read_file)
#     return file
    
search_job_vacancy_tool = FunctionTool.from_defaults(async_fn=search_job_vacancy) 
get_personality_tool = FunctionTool.from_defaults(async_fn=get_riasec_personality_test_result) 
get_job_vacancy_detail_tool = FunctionTool.from_defaults(async_fn=get_job_vacancy_slug_detail) 
# get_answer_from_file_tool = FunctionTool.from_defaults(async_fn = get_answer_from_file)


tools = [search_job_vacancy_tool, get_personality_tool, get_job_vacancy_detail_tool]
# tools = [search_job_vacancy_tool, get_answer_from_file_tool]

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    # Initialize with custom chat history
    init_history = [
        ChatMessage(role=MessageRole.ASSISTANT, content="Halo! Mau cari lowongan pekerjaan apa?"),
    ]
    memory = ChatMemoryBuffer.from_defaults(token_limit=64768)
    st.session_state.chat_engine = ReActAgent.from_tools(
        tools,
        chat_mode="react",
        verbose=True,
        memory=memory,
        react_system_prompt=react_system_prompt,
        # retriever=retriever,
        llm=Settings.llm
    )

    print(st.session_state.chat_engine.get_prompts())

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response_stream.response_gen)

    # Add user message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_stream.response})

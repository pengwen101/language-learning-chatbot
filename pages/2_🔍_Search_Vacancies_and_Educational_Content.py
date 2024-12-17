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
from agents.rating_agent import get_relevant_jobs

nest_asyncio.apply()

riasec_result_data = pd.read_csv('./answers/riasec_assessment_answer.csv')
riasec_result_key_values = [{row['Type']: row['Total Score']} for index, row in riasec_result_data.iterrows()]
top_3 = sorted(riasec_result_key_values, key=lambda x: list(x.values())[0], reverse=True)[:3]
print(top_3)

system_prompt = system_prompt = """
You are a multi-lingual career advisor expert who has knowledge based on 
real-time data. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.

Your responsibilities are as follows:

1. Job Preference Detection:
Detect and record users' job preferences whenever they mention keywords or phrases related to their career interests (e.g., "remote work," "full-stack development," "data analysis"). Preferences are dynamically stored and used to refine job search queries.

2. Job Matching:
Use user preferences (keywords and RIASEC results) to search the Petra Alumni database for relevant job vacancies.

3. Detailed Job Information Display:
When displaying jobs, include all details retrieved from the database without summarizing them. Each job must include:
    Position name and type (e.g., full-time, part-time).
    Job location.
    Job system (e.g., onsite, hybrid, remote).
    Minimum degree required to apply.
    Salary range.
    Application deadline.
    Job description.
    Job requirements.

4. Educational Resources:
Based on user RIASEC results, search and recommend educational content to help users enhance their skills or explore career options.

5. Output Rules:
Always use a numbered list format for job listings.
Include all details explicitly as retrieved from the database.
Explain how the listed jobs align with the user's RIASEC personality results.

Here is a short example:
User: I would like to search a job according to my riasec result
Assistant: Sure! Here are come job vacancies related your riasec result:
1. Account Finance Manager di Kota Surabaya. 
Tipe: <Type of the job>
Sistem: <System of the job>
Level Pendidikan = <Minimum degree to apply for the job>
Range Gaji = <Salary range of the job>
Batas Apply = <Job application due date>
Description: <Elaborate the job description>
Requirements: <Elaborate the job requirements>
<A reason why this job match user's RIASEC result>
Link: <Link to job vacancy>

You MUST display the job with above format only. DO NOT display in any other format.
"""

react_system_header_str =  """\

You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.

## Tools
You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.
Prioritize using the record preference tool every time for storing the user preference everytime user inputs, then use the other tool.
ONLY use search vacan if the user interested in exploring other options, if the user only wants to talk, noo need to use this tool.

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

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

"""
react_system_prompt = PromptTemplate(react_system_header_str)

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

Settings.llm = Ollama(model="llama3.1:8b-instruct-q4_0", base_url="http://127.0.0.1:11434", system_prompt=system_prompt, temperature=0)
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")


# Declare Tools

async def record_new_preference(preference: str):
    """
    Detect user's job preferences everytime user mention anything that can be interpreted as a preference towards a job. The job preference should be in the form of keywords such as "marketing", "remote work," "full-stack development," "data analysis", "machine learning", "creative writing", and others. If the preference was "career assessment" or anything that asked the previously tested riasec assessment, skip this tool.
    """

    if "riasec" not in preference.lower():
        if 'preference' not in riasec_result_data.columns:
            riasec_result_data['preference'] = ""
        preferences = riasec_result_data['preference'].iloc[0].split(", ") if not riasec_result_data['preference'].empty else []
        if preference not in preferences:
            preferences.append(preference)
        riasec_result_data['preference'] = ", ".join(set(preferences))
        print(riasec_result_data['preference'])
    return "User preference is stored"


async def search_job_vacancy_riasec() -> str:
    """
    Searches the Alumni Petra database for a list of job vacancies. Jobs are shown in a numbered list format. For each job you must explain why that job matches user's RIASEC result.
    If the user seems to be interested in certain jobs or want to explore other options related to their RIASEC test result and preferences, use this tool as well.
    """

    try:

        keyword = list(top_3[0].keys())[0] + ", " + list(top_3[1].keys())[0] + ", " + list(top_3[2].keys())[0]
        print("Keyword:", keyword)
        if 'preference' in riasec_result_data.columns and not riasec_result_data['preference'].empty:
            preferences = set(riasec_result_data['preference'].iloc[0].split(", "))
            keyword += ", " + ", ".join(preferences)
        print("Keywords:", keyword)
    
        relevant_jobs = get_relevant_jobs(keyword)
        relevant_slugs = relevant_jobs['Link'].map(lambda x: x[35:]).to_numpy()
        idx = 1
        output = ""
        for slug in relevant_slugs:
            if idx > 5:
                break
            r = requests.get(f'https://panel-alumni.petra.ac.id/api/vacancy/{slug}')
            print(r, f'for link: https://panel-alumni.petra.ac.id/api/vacancy/{slug}')
            if r.status_code != 200:
                continue
            data = r.json()
            print(data["vacancy"]["salary_start"])
            salary_start = data['vacancy']['salary_start'] if data['vacancy']['salary_start'] is not None else ''
            print("salary start:", salary_start)
            print(data["vacancy"]["salary_end"])
            salary_end = data['vacancy']['salary_end'] if data['vacancy']['salary_end'] is not None else ''
            print("salary end:", salary_end)
            salary_info = f"{salary_start} - {salary_end}" if salary_start or salary_end else 'Tidak ada informasi'
            print("salary info:", salary_info)
            output += f"""
                {idx}. {data['vacancy']['position_name']} at {data['vacancy']['mh_company']['name']}
                Lokasi: {data['vacancy']['mh_city']['name']}
                Tipe: {data['vacancy']['type']}
                Sistem: {data['vacancy']['system']}
                Level Pendidikan: {data['vacancy']['level_education']}
                Range Gaji: {salary_info}
                Batas Apply: {data['vacancy']["expired_date"]}
                Deskripsi: {BeautifulSoup(data['vacancy']['description'], 'html.parser').get_text() if data['vacancy']['description'] is not None else ''}
                Job Requirements: {BeautifulSoup(data['vacancy']['requirement'], 'html.parser').get_text() if data['vacancy']['requirement'] is not None else ''}
                <A reason why this job match user's RIASEC result (Why {data['vacancy']['position_name']} match {keyword})>
                Link: https://alumni.petra.ac.id/vacancy/{slug} [ALWAYS SHOW THIS TO USER]
            """
            idx += 1
        output += "\n\nShow it directly to user with location, type, system, educational level, salary range, application deadline, description, job requirements, and reason why that job match user RIASEC result. DON'T call other tools again."
        return output
    except Exception as e:
        return f"Error fetching jobs: {str(e)}"

async def provide_topic_for_educational_content():
    """
    You are a topic generation expert. Your task is to:
    1. Take a user's RIASEC test result as input.
    2. Identify the most relevant occupations suited for the given RIASEC profile.
    3. Generate a topic for educational content that would help the user prepare for those occupations.
    Output the topic in a single sentence, focusing on specific skills, knowledge, or training areas.

    """

    this_system_prompt = """
    You are a topic generation expert. Your task is to:
    1. Take a user's RIASEC test result as input.
    2. Identify the most relevant occupations suited for the given RIASEC profile.
    3. Generate a topic for educational content that would help the user prepare for those occupations.
    Output the topic in a single sentence, focusing on specific skills, knowledge, or training areas.

    """

    holland_docs = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(holland_docs)
    query_engine = index.as_query_engine(
        chat_mode="context",
        memory=memory,
        system_prompt = this_system_prompt,
        verbose=True
    )

    response = query_engine.query(f"Generate a topic for {top_3}")

    return response
    
    

async def search_educational_content(topic: str):
    """Search for educational content based on user job recommendation using DuckDuckGo. Call other tool to provide a topic."""

    try:
        with DDGS() as ddg:
            results = ddg.text(f"educational content for {topic} {datetime.now().strftime('%Y-%m')}", max_results=1)
            if results:
                print(results)
                educational_content = "\n\n".join([
                    f"Title: {result['title']}\nURL: {result['href']}\nSummary: {result['body']}" 
                    for result in results
                ])
                educational_content += f"Show to user the summary and reason why the content match user RIASEC result ({top_3}). ALWAYS show user the URL. DON'T call other tools again."
                return educational_content
            return f"No educational content found for {top_3}."
    except Exception as e:
        return f"Error fetching educational content. {str(e)}"

async def get_job_details(job_slug: str):
    """Fetch detailed information about a specific job based on its slug."""
    try:
        r = requests.get(f'https://panel-alumni.petra.ac.id/api/vacancy/{job_slug}')
        if r.status_code == 200:
            data = r.json()
            salary_start = data['vacancy']['salary_start'] or ''
            salary_end = data['vacancy']['salary_end'] or ''
            salary_info = f"{salary_start} - {salary_end}" if salary_start or salary_end else 'Tidak ada informasi'

            return f"""
            {data['vacancy']['position_name']} at {data['vacancy']['mh_company']['name']}
            Lokasi: {data['vacancy']['mh_city']['name']}
            Tipe: {data['vacancy']['type']}
            Sistem: {data['vacancy']['system']}
            Level Pendidikan: {data['vacancy']['level_education']}
            Range Gaji: {salary_info}
            Batas Apply: {data['vacancy']['expired_date']}
            Deskripsi: {BeautifulSoup(data['vacancy']['description'], 'html.parser').get_text()}
            Job Requirements: {BeautifulSoup(data['vacancy']['requirement'], 'html.parser').get_text()}
            <Provide a reason why this job match user's RIASEC result and preference ({keyword})
            Link: https://alumni.petra.ac.id/vacancy/{job_slug} [ALWAYS SHOW THIS TO USER]
            """
        else:
            return f"Failed to fetch details for job ID: {job_slug}"
    except Exception as e:
        return f"Error fetching job details: {str(e)}"
        
alumni_job_tool = FunctionTool.from_defaults(async_fn=search_job_vacancy_riasec)
educational_content_tool = FunctionTool.from_defaults(async_fn=search_educational_content)
record_preference_tool = FunctionTool.from_defaults(async_fn=record_new_preference)
job_detail_tool = FunctionTool.from_defaults(async_fn=get_job_details)
provide_topic_tool = FunctionTool.from_defaults(async_fn=provide_topic_for_educational_content)

tools = [record_preference_tool, alumni_job_tool, job_detail_tool, provide_topic_tool, educational_content_tool]


# Main Program
st.title("Search for Jobs On Alumni Website or Educational Content! 🔍")

# Initialize chat history if empty
if "messages_job" not in st.session_state:
    st.session_state.messages_job = [
        {"role": "assistant",
         "content": "Hello! I can provide you with jobs or educational content based on your RIASEC result! 😊"}
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
            try:
                response_stream = st.session_state.chat_engine_job.stream_chat(prompt)
                st.write_stream(response_stream.response_gen)
            except Exception as e:
                response_stream = "Unable to process your request. Please try again."
                st.write(response_stream)

    # Add user message to chat history
    st.session_state.messages_job.append({"role": "assistant", "content": response_stream.response})
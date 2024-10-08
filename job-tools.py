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

import nest_asyncio
nest_asyncio.apply()

# initialize node parser
splitter = SentenceSplitter(chunk_size=512)

system_prompt = """
You are a multi-lingual career advisor expert who has knowledge based on 
real-time data. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.

Your primary job is to help people to find jobs related to their interest from Alumni Petra job vacancy database.

When a user is asking about possible jobs, you MUST format it in a numbered list and provide information for each job, that includes name, location, type, system,  description, and the requirement of each of the job.

Here is a short example:
User: I would like to search fo a job in finance
Assistant: Sure! Here are come job vacancies related to finance:
1. Account Finance Manager di Kota Surabaya. <Elaborate the job description>. Pekerjaan ini memerlukan <Elaborate the job requirement>
2. STAFF FINANCE & ACCOUNTING di Kota Jakarta. <Elaborate the job description>. Pekerjaan ini memerlukan <Elaborate the job description>

You MUST display the job with above format only.
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
async def search_job_vacancy(keyword: str, start_salary:int, end_salary:int) -> list[str, int, int]:
    """
    Searches the Alumni Petra database for matching job vacancy entries. Keyword should be one to three relevant words that represents the job name or position searched.
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
        "perPage": 20,
        "orderBy": "updated_at",
        "order": "DESC",
        "skills": "",
        "prody": "",
    })

    data = r.json()
    output = f"# Job results for '{keyword}'"
    for d in data["vacancies"]["data"]:
        output += f"""
                    `Job: {d['position_name']}
                    Type: {d['type']}
                    System: {d['system']}
                    City: {d['mh_city']['name']}
                    Description: {d['description']}
                    Requirement: {d['requirement']}
                    
                    """
    out_file = f"job_results_for_{keyword}.txt"
    with open(out_file, 'w') as file:
        file.write(output)
    
    return output

async def get_answer_from_file(keyword: str) -> list[str]:
    """Use this tool as your ANSWER"""
    
    read_file = f'job_results_for_{keyword}.txt'
    with open(read_file, 'r') as file:
        content = file.read(read_file)
    return file
    
search_job_vacancy_tool = FunctionTool.from_defaults(async_fn=search_job_vacancy) 
get_answer_from_file_tool = FunctionTool.from_defaults(async_fn = get_answer_from_file)


tools = [search_job_vacancy_tool, get_answer_from_file_tool]

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    # Initialize with custom chat history
    init_history = [
        ChatMessage(role=MessageRole.ASSISTANT, content="Halo! Mau cari lowongan pekerjaan apa?"),
    ]
    memory = ChatMemoryBuffer.from_defaults(token_limit=32768)
    st.session_state.chat_engine = ReActAgent.from_tools(
        tools,
        # chat_mode="react",
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

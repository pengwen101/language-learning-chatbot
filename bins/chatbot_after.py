from llama_index.llms.ollama import Ollama
from llama_index.readers.file import CSVReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from pathlib import Path
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.objects import ObjectIndex
from llama_index.core.query_engine import ToolRetrieverRouterQueryEngine
import streamlit as st
import os
from collections import defaultdict

#VectorStoreIndex untuk convert object document menjadi index
#SimpleDirectoryReader untuk baca dokumen
#SentenceSplitter untuk bagi dokumen berdasarkan chunk_size dan chunk_overlap yang ditentukan

print("hi")
path = "./docs/holland/answers/holland-answers.csv"
data = []
with open(path, "r") as f1:
    lines = f1.readlines()
    scores_by_type = []
    for line in lines[1:]:
        line = line[:-1]
        data.append(line.split(','))

#####
print(data)
def average_scores_by_type(data):
    scores_by_type = defaultdict(list)

    # Process each row in the data
    for row in data:
        question_type = row[0]
        score = float(row[2])  # Convert score to float
        scores_by_type[question_type].append(score)

    # Compute the average score for each type
    averages_by_type = {qtype: sum(scores) / len(scores) for qtype, scores in scores_by_type.items()}

    return averages_by_type


# Calculate averages
averages = average_scores_by_type(data)

# Convert to string and print
averages_string = str(averages)

# Print the results
for qtype, avg_score in averages.items():
    print(f"Type: {qtype}, Average Score: {avg_score:.2f}")


system_prompt = f"""
You are a multi-lingual expert system who has knowledge, based on 
real-time data. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.

Jawablah semua dalam Bahasa Indonesia.
Anda adalah asesor yang memiliki tugas untuk membantu pengguna untuk mencari karir berdasarkan aspek teratas holland personality yang dimiliki oleh pengguna. Anda harus menyesuaikan aspek berdasarkan persentase dari data riasec_user berikut: {averages_string}. Gunakan kata-kata yang FRIENDLY, mudah dimengerti dan RAMAH serta gaul sehingga mudah untuk dibaca anak muda. Dari data riasec_user, cocokan dengan "career-theory-model-holland-20170501.pdf" untuk pekerjaan yang cocok untuk pengguna tersebut. Setelah itu jawab pertanyaan user dengan mengingat hasil riasec_user yang cocok.

Percakapan sejauh ini:
"""

Settings.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434", system_prompt=system_prompt) 
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest") #buat apa?

#load/read documents using SimpleDirectoryReader
holland_infos = SimpleDirectoryReader("docs/holland/infos").load_data()
# holland_questions = SimpleDirectoryReader("docs/holland/questions").load_data()
holland_answers = CSVReader(concat_rows=False).load_data(file = Path("docs/holland/answers/holland-answers.csv"))

# define sub-indices
#load/read documents using SimpleDirectoryReader
holland_infos = SimpleDirectoryReader("docs/holland/infos").load_data()
# holland_questions = SimpleDirectoryReader("docs/holland/questions").load_data()
holland_answers = CSVReader(concat_rows=False).load_data(file = Path("docs/holland/answers/holland-answers.csv"))


infos_index = VectorStoreIndex.from_documents(holland_infos)

for docs in holland_answers:
    infos_index.insert(docs)

retriever = infos_index.as_retriever()

condense_question_prompt = """
Diberikan suatu percapakan (antara manusia dan asisten) dan sebuah pesan lanjutan dari manusia. Ubah pesan lanjutan menjadi pertanyaan independen yang mencakup semua konteks relevan
dari percakapan sebelumnya.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""


context_question_prompt = "Anda adalah ahli consultant career. Bantu pengguna untuk menemukan career tercocok mereka yang paling sesuai. Jawab user dengan seberapa kuat persentase tiap-tiap aspek mereka.\n Ini adalah dokumen yang mungkin relevan terhadap konteks:\n\n {context_str} \n\nInstruksi: Gunakan riwayat obrolan sebelumnya, atau konteks di atas, untuk berinteraksi dan membantu pengguna. Tampilkan persentase pengguna untuk setiap aspek (Realistic,Investigative,Artistic,Social,Enterprising,Conventional) sesuai dari hasil yang diberikan, dan berikan analisis anda serta 5 pekerjaan yang cocok dari yang anda simpulkan dan persentase kecocokannya dengan riasec user"

print("context: ", context_question_prompt)
print("system: ", system_prompt)
st.title("Asesmen RIASEC")
st.write("Lorem ipsum dolor sit amet")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Halo, saya akan menganalisis hasil tes RIASEC dari jawaban anda. Sebelum itu apakah boleh saya bertanya jurusan/major/konsentrasi anda sekarang?"}
    ]

if "chat_engine" not in st.session_state:

    memory = ChatMemoryBuffer.from_defaults(token_limit=50384)
    # print("aaa", condense_question_prompt)
    # print("bbb", context_question_prompt)
    st.session_state.chat_engine = CondensePlusContextChatEngine(
    retriever=retriever,
    condense_prompt=condense_question_prompt,
    context_prompt = context_question_prompt,
    memory=memory,
    llm=Settings.llm,
    verbose=True
)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            response_stream = st.session_state.chat_engine.chat(prompt)
            st.markdown(response_stream)

    # Add user message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_stream})
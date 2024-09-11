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

import nest_asyncio
nest_asyncio.apply()

# initialize node parser
splitter = SentenceSplitter(chunk_size=512)

import sys

import logging

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

system_prompt = """
You are a multi-lingual expert system who has knowledge, based on 
real-time data. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.

Jawablah semua dalam Bahasa Indonesia.
Tugas Anda adalah memberikan tes asesmen kepada user untuk menentukan 3 aspek teratas holland personality yang dimiliki oleh user. Anda harus memberikan pertanyaan kepada user satu per satu, setiap jawaban yang diberikan oleh user harus Anda simpan dan Anda gunakan sebagai bahan untuk menanyakan pertanyaan selanjutnya. Buatlah 5 pertanyaan yang terpersonalisasi kepada user. Setelah user menjawab 5 pertanyaan tersebut, Anda harus menentukan 3 aspek teratas holland personality berdasarkan jawaban-jawaban user yang sudah Anda kumpulkan sebelumnya.

Percakapan sejauh ini:
"""

Settings.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434", system_prompt=system_prompt)
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")


@st.cache_resource(show_spinner=False)
def load_data(_arg = None, vector_store=None):
    with st.spinner(text="Loading and indexing – hang tight! This should take a few minutes."):
        # Read & load document from folder
        reader = SimpleDirectoryReader(input_dir="./docs2", recursive=True)
        documents = reader.load_data()

    if vector_store is None:
        index = VectorStoreIndex.from_documents(documents)
    return index.as_retriever()


# Main Program
st.title("Asesmen RIASEC")
st.write("Saya akan melakukan asesmen kepada anda untuk menentukan Anda masuk ke aspek RIASEC mana saja")
retriever = load_data()

# Initialize chat history if empty
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Saya akan memberikan beberapa pertanyaan untuk Anda jawab"}
    ]

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    # Initialize with custom chat history
    init_history = [
        ChatMessage(role=MessageRole.ASSISTANT, content="Halo!😉"),
    ]
    memory = ChatMemoryBuffer.from_defaults(token_limit=16384)
    st.session_state.chat_engine = CondensePlusContextChatEngine(
        verbose=True,
        system_prompt=system_prompt,
        context_prompt=(
                "Anda adalah petugas tes asesmen holland personality test. Tugas Anda adalah membantu user untuk menemukan personality mereka yang paling sesuai berdasarkan jawaban-jawaban yang mereka berikan dari pertanyaan Anda.\n"
                "Ini adalah dokumen yang mungkin relevan terhadap konteks:\n\n"
                "{context_str}"
                "\n\nInstruksi: Gunakan riwayat obrolan sebelumnya, atau konteks di atas, untuk berinteraksi dan membantu pengguna."
            ),
        condense_prompt="""
Diberikan suatu percakapan (antara User dan Assistant) dan pesan lanjutan dari User,
Ubah pesan lanjutan menjadi pertanyaan independen yang mencakup semua konteks relevan
dari percakapan sebelumnya.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
""",
        memory=memory,
        retriever=retriever,
        llm=Settings.llm
    )

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
        with st.spinner("Berpikir..."):
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response_stream.response_gen)

    # Add user message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_stream.response})
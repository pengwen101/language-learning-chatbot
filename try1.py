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
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.objects import ObjectIndex
from llama_index.core.query_engine import ToolRetrieverRouterQueryEngine
import streamlit as st

#VectorStoreIndex untuk convert object document menjadi index
#SimpleDirectoryReader untuk baca dokumen
#SentenceSplitter untuk bagi dokumen berdasarkan chunk_size dan chunk_overlap yang ditentukan

system_prompt = """
You are a multi-lingual expert system who has knowledge, based on 
real-time data. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.

Jawablah semua dalam Bahasa Indonesia.
Tugas Anda adalah memberikan tes asesmen kepada user untuk menentukan 3 aspek teratas holland personality yang dimiliki oleh user. Anda harus menggunaan dokumen yang sesuai untuk memberikan pertanyaan kepada pengguna. Anda harus memberikan pertanyaan kepada user satu per satu, setiap jawaban yang diberikan oleh user harus Anda simpan dan Anda gunakan sebagai bahan untuk menanyakan pertanyaan selanjutnya. Jika pengguna mengalami kebingungan ketika menjawab, lakukanlah penggalian lebih dalam dengan cara menanyakan kembali pertanyaan tersebut kepada pengguna. Setelah pengguna menjawab 5 pertanyaan, Anda harus menentukan 3 aspek teratas holland personality berdasarkan jawaban-jawaban user yang sudah Anda kumpulkan sebelumnya.

Percakapan sejauh ini:
"""

Settings.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434", system_prompt=system_prompt) 
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest") #buat apa?




#load/read documents using SimpleDirectoryReader
holland_infos = SimpleDirectoryReader("docs/holland/infos").load_data()
# holland_questions = SimpleDirectoryReader("docs/holland/questions").load_data()
holland_questions = CSVReader(concat_rows=False).load_data(file = Path("docs/holland/questions/holland-questions.csv"))

#SentenceWindowNodeParser digunakan untuk scope yang sangat spesifik, seperti holland_questions yang memiliki kolom "tipe" dan "pertanyaan" (asosiasi tipe dan pertanyaan sangat spesifik)

node_parser = SentenceWindowNodeParser.from_defaults(
    # how many sentences on either side to capture
    window_size=1,
    # the metadata key that holds the window of surrounding sentences
    window_metadata_key="window",
    # the metadata key that holds the original sentence
    original_text_metadata_key="original_sentence",
)

nodes = node_parser.get_nodes_from_documents(holland_questions)

# define sub-indices
index_holland_infos = VectorStoreIndex.from_documents(holland_infos)
index_holland_questions = VectorStoreIndex(nodes)

# define query engines and tools
tool_holland_infos = QueryEngineTool.from_defaults(
    query_engine=index_holland_infos.as_query_engine(),
    description="Gunakan query ini untuk mencari informasi mengenai ciri-ciri, karakteristik, kepribadian, kecenderungan, dan karir yang cocok untuk setiap tipe holland personality.",
)
tool_holland_questions = QueryEngineTool.from_defaults(
    query_engine=index_holland_questions.as_query_engine(),
    description="Dokumen ini bukan untuk menjawab pertanyaan, namun untuk memberikan pertanyaan kepada pengguna. Gunakan query ini untuk memberikan pertanyaan parafrase kepada pengguna untuk menentukan tipe holland personality pengguna. Jangan berikan pertanyaan yang tidak diparafrase, karena parafrase digunakan agar pengguna dapat merasa nyaman untuk menjawab pertanyaan yang Anda lontarkan. Jika jawaban pengguna belum cukup untuk menjawab pertanyaan Anda, lakukanlah penggalian lebih dalam dengan cara menanyakan kembali pertanyaan yang sama namun diparafrase",
)

obj_index = ObjectIndex.from_objects([tool_holland_infos, tool_holland_questions], index_cls = VectorStoreIndex)

# query_engine = ToolRetrieverRouterQueryEngine(obj_index.as_retriever())

query_engine = ToolRetrieverRouterQueryEngine(obj_index.as_retriever())

condense_question_prompt = PromptTemplate(
    """\
Diberikan suatu percapakan (antara manusia dan asisten) dan sebuah pesan lanjutan dari manusia. Ubah pesan lanjutan menjadi pertanyaan independen yang mencakup semua konteks relevan
dari percakapan sebelumnya.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""
)

context_prompt = PromptTemplate(
                "Anda adalah petugas tes asesmen holland personality test. Tugas Anda adalah memberikan pertanyaan asesmen berdasarkan pertanyaan yang tersedia namun dengan pembahasaan yang sesuai dengan pengguna. Bantu pengguna untuk menemukan personality mereka yang paling sesuai berdasarkan jawaban-jawaban yang mereka berikan dari pertanyaan Anda.\n"
                "Ini adalah dokumen yang mungkin relevan terhadap konteks:\n\n"
                "{context_str}"
                "\n\nInstruksi: Gunakan riwayat obrolan sebelumnya, atau konteks di atas, untuk berinteraksi dan membantu pengguna."
            )
st.title("Asesmen RIASEC")
st.write("Saya akan melakukan asesmen kepada anda untuk menentukan Anda masuk ke aspek RIASEC mana saja")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Saya akan memberikan beberapa pertanyaan untuk Anda jawab"}
    ]

if "chat_engine" not in st.session_state:

    memory = ChatMemoryBuffer.from_defaults(token_limit=50384)

    st.session_state.chat_engine = CondenseQuestionChatEngine(
    query_engine=query_engine,
    condense_question_prompt=condense_question_prompt,
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
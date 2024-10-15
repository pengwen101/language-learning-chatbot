from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
import streamlit as st
import pandas as pd

riasec_result_data = pd.read_csv('../answers/riasec_assessment_answer.csv')
riasec_result_key_values = [{row['Type']: row['Total Score']} for index, row in riasec_result_data.iterrows()]
top_3 = sorted(riasec_result_key_values, key=lambda x: list(x.values())[0], reverse=True)[:3]
print(top_3)

system_prompt = f"""
You are a multi-lingual expert system who has knowledge, based on 
real-time data. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.

Jawablah semua dalam Bahasa Indonesia.
Anda adalah asesor yang memiliki tugas untuk membantu pengguna untuk mencari karir berdasarkan aspek teratas holland personality yang dimiliki oleh pengguna. Aspek teratas yang dimiliki user adalah {top_3[0]}, {top_3[1]}, {top_3[2]}. JANGAN SEBUTKAN ASPEK LAIN SELAIN TIGA ASPEK TERSEBUT. Berikan analisis serta 5 pekerjaan yang cocok untuk user.

Percakapan sejauh ini:
"""

Settings.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434", system_prompt=system_prompt) 
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest") 

holland_docs = SimpleDirectoryReader("../docs").load_data()
index = VectorStoreIndex.from_documents(holland_docs)

st.title("Rekomendasi Karir Berdasarkan Hasil Tes RIASEC 💼")
# st.write("Lorem ipsum dolor sit amet")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Halo, apakah anda siap untuk mendapatkan rekomendasi karir?"}
    ]

if "chat_engine" not in st.session_state:

    memory = ChatMemoryBuffer.from_defaults(token_limit=50384)
    st.session_state.chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt= system_prompt,
    verbose=True
)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt:= st.chat_input("What is up?"):
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
import streamlit as st
import pandas as pd
import os

# Path ke file CSV
csv_file_path = os.path.join(".." , "docs", "holland", "questions", "holland-questions.csv")

# Membaca file CSV
questions_df = pd.read_csv(csv_file_path)

st.title("RIASEC Assessment")

with st.form("riasec_form"):
    st.write("Jawablah pertanyaan berikut dengan skala 1-5: \n")
    st.write("**(1 = Sangat Tidak Yakin, 2 = Tidak Yakin, 3 = Netral, 4 = Yakin, 5 = Sangat Yakin)**")

    answers = {}
    
    st.subheader("Questions:")

    for index, row in questions_df.iterrows():
        question = row['Question']
        
        answers[f"answer_{index}"] = st.slider(f"{index+1}. {question}", min_value=1, max_value=5, value=1, 
                                               format="%d", help="1 = Sangat Tidak Yakin, 2 = Tidak Yakin, 3 = Netral, 4 = Yakin, 5 = Sangat Yakin")

    st.markdown("""
    <style>
    div.stButton > button {
        display: block;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    submitted = st.form_submit_button("Submit")

if submitted:
    total_scores = questions_df.groupby('Type')['Question'].count().to_dict()
    for index in range(len(questions_df)):
        type_ = questions_df.loc[index, 'Type']
        score = answers[f'answer_{index}']
        total_scores[type_] += score

    # Menyimpan hasil ke file CSV baru
    total_scores_df = pd.DataFrame(total_scores.items(), columns=['Type', 'Total Score'])
    total_scores_df.to_csv('riasec_assessment_answer.csv', index=False)
    st.success("Jawaban anda telah disimpan!")
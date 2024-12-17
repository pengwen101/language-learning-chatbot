import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to RIASEC Carrer Recommendation! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    The Holland Codes or the Holland Occupational Themes refers to a taxonomy 
    of interests based on a theory of careers and vocational choice that was 
    initially developed by American psychologist John L. Holland.
    ### The 6 types of personality
    - Realistic
    - Investigative
    - Artistic
    - Social
    - Enterprising
    - Conventional
    
    To learn more about the 6 types, check out [O*NET Interest Profiler Manual](https://www.onetcenter.org/dl_files/IP_Manual.pdf)

    This test is useful for you who wants to search for careers that best fit your personality and likings.
    
    ### Navigate the app by following these steps:
    1. Go to **📃 RIASEC Test** to do the full test
    2. Search for vacancies on the [Alumni Petra Website](https://alumni.petra.ac.id/blog/) or educational content that are related to you RIASEC result with **🔍 Search Vacancies and Educational Content**

    You can also upload documents to feed the chatbot through **📁 Upload Files**

    ### Happy Searching!
"""
)

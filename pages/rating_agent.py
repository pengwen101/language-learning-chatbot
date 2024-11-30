import streamlit as st
from swarm import Swarm, Agent
from dotenv import load_dotenv
import pandas as pd
import csv
import time
import pandas as pd

load_dotenv()
MODEL = "llama3.2:latest"
client = Swarm()

rating_agent = Agent(
    name="Rating Agent",
    instructions="""
        Instructions:
        Your task is to evaluate the compatibility between a keyword and a job slug. 
        The compatibility score must be an integer between 0 and 100, where:
        
        - 100: The keyword is highly relevant to the job slug, meaning it is essential or frequently used for the job described by the job slug.
        - 0: The keyword has no relevance to the job slug and is not typically associated with the job.
        
        Requirements:
        Focus on whether the keyword is relevant for the job described in the job slug, such as being a required skill, tool, or qualification.
        Output only the compatibility score as an integer (e.g., 85). DO NOT include any additional text, symbols, or characters.
    """,
    functions=[],
    model=MODEL,
)

def get_compability_rating(keyword, job):
    prompt = f"From the scale 0 - 100, what is the compatibility for {keyword} to have a job as a {job}"
    print(prompt)
    compatibility_response = client.run(
        agent=rating_agent,
        messages=[{"role": "user", "content": f"{prompt}"}]
    )
    rating = compatibility_response.messages[-1]['content']
    return rating

def load_csv_slugs():
    with open('pages/slug.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        slugs = []
        for row in spamreader:
            slugs.append(row[0])
        return slugs

def load_xlsx_jobs():
    file_path = 'pages/jobs.xlsx' 
    data = pd.read_excel(file_path)
    data["rating"] = 0
    return data     
    
def get_relevant_jobs(keyword):
    jobs = load_xlsx_jobs()
    
    for index, row in jobs.iterrows():
        job_position = jobs.loc[index, "Position"]
        rating = get_compability_rating(keyword, job_position)
        print(rating)
        jobs.loc[index, "rating"] = rating
    
    jobs_sorted = jobs.sort_values(by="rating", ascending=False)
    print(jobs_sorted.head())
    return jobs_sorted


# Comment mulai sini kalo mau diimport
while True:
    keyword = input("Keyword: ")
    get_relevant_jobs(keyword)
    print()

# print(jobs)
# slugs = load_csv_slugs()
# print(f"SLUGS = {slugs}\n\n\n")
# print(len(slugs))

# while True:
#     keyword = input("Keyword: ")
#     start_time = time.time()
#     for slug in slugs:
#         print(f"{keyword} and {slug} = {get_compability_rating(keyword, slug)}")
#     end_time = time.time()
#     runtime = end_time - start_time
#     print(f"Runtime: {runtime:.4f} seconds\n")
        
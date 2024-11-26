import requests
from bs4 import BeautifulSoup
import csv

def search_job_vacancy_riasec(keyword: str = "", start_salary:int = 5000, end_salary:int = 100000000000, show_explaination:bool = True, id_mh_province:int = "") -> str:
    """
    Searches the Alumni Petra database for a list of job vacancies. If a province is specified, retrieve its ID first. Jobs are shown in a numbered list format, with an explanation if requested.
    """
    print("Requesting to alumni petra website...")
    r = requests.get('https://panel-alumni.petra.ac.id/api/vacancy', {
        "page": 1,
        "type": "freelance,fulltime,parttime,internship",
        "system": "onsite,remote,hybrid",
        "level_education": "diploma,sarjana,magister,doktor",
        "keyword": keyword,
        "salary_range": str(start_salary) + ", " + str(end_salary),
        "id_mh_province": id_mh_province,
        "id_mh_city": "",
        "perPage": 1000,
        "orderBy": "updated_at",
        "order": "DESC",
        "skills": "",
        "prody": "",
    })

    data = r.json()
    if len(data["vacancies"]["data"]) == 0:
        return "No jobs available for your query."

    output = []
    idx = 1
    for d in data["vacancies"]["data"]:
       
        output.append({'slug': d['slug']})
        idx += 1

    return output

slugs = search_job_vacancy_riasec('')
with open('alumni_job_slugs.csv', 'w', newline='') as csvfile:
    fieldnames = ['slug']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(slugs)
    
with open('alumni_job_slugs.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        print(', '.join(row))
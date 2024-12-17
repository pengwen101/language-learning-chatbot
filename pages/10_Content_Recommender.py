import streamlit as st
from duckduckgo_search import DDGS
from swarm import Swarm, Agent
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
MODEL = "llama3.2:latest"
client = Swarm()

st.set_page_config(page_title="Educational Content Recommender", page_icon="📰")
st.title("📰 Educational Content Recommender")

def search_educational_content(topic):
    """Search for educational content using DuckDuckGo."""
    with DDGS() as ddg:
        results = ddg.text(f"educational content for {topic} {datetime.now().strftime('%Y-%m')}", max_results=3)
        if results:
            educational_content = "\n\n".join([
                f"Title: {result['title']}\nURL: {result['href']}\nSummary: {result['body']}" 
                for result in results
            ])
            return educational_content
        return f"No educational content found for {topic}."

# Create specialized agents

topic_generator_agent = Agent(
    name="Topic Generator",
    instructions="""
    You are a topic generation expert. Your task is to:
    1. Take a user's RIASEC test result as input.
    2. Identify the most relevant occupations suited for the given RIASEC profile.
    3. Generate a topic for educational content that would help the user prepare for those occupations.
    Output the topic in a single sentence, focusing on specific skills, knowledge, or training areas.
    """,
    model=MODEL
)


search_agent = Agent(
    name="Educational Content Searcher",
    instructions="""
    You are an educational content specialist. Your task is to:
    1. Search for the most relevant and recent educational content for the given topic.
    2. Ensure the results are from reputable sources.
    3. Return the raw search results in a structured format.
    """,
    functions=[search_educational_content],
    model=MODEL
)

synthesis_agent = Agent(
    name="Content Synthesizer",
    instructions="""
    You are a synthesis expert. Your task is to:
    1. Analyze the raw educational content provided.
    2. Identify key themes and important information.
    3. Combine information from multiple sources.
    4. Create a comprehensive but concise synthesis.
    5. Focus on facts and maintain clarity.
    Provide a 2-3 paragraph synthesis of the main points.
    """,
    model=MODEL
)

summary_agent = Agent(
    name="News Summarizer",
    instructions="""
    You are an expert educational content summarizer, blending clarity and conciseness with actionable insights, while ensuring absolute accuracy and reliability.  

    Your task:  
    1. Core Information:  
       - Highlight the key skills, knowledge areas, or certifications related to the topic.  
       - Include practical steps or resources for acquiring these skills, only if they are verified as accurate and trustworthy.  
       - Cross-check and ensure that any URLs, data, or resources mentioned are from credible and reputable sources.  
       - Explain the relevance of this content to career advancement or job preparation.  
       - Provide essential data, such as demand for the skill or related industries, only if sourced from valid references.
       - Provide the URLs from all resources that are mentioned
    
    2. Style Guidelines:  
       - Use strong, active verbs.  
       - Be clear, precise, and avoid jargon unless necessary (with explanations).  
       - Maintain objectivity while engaging the reader.  
       - Make every word purposeful, avoiding redundancy.  
       - Keep a professional yet approachable tone.  
       - Emphasize factual integrity and avoid any assumptions or speculative statements.  
    
    Format: Create a single paragraph of 250-400 words that informs, inspires, and guides the reader.  
    Pattern: [Key Educational Focus] + [Details and Practical Resources] + [Why It Matters/Next Steps] + [Sources' URLs]
    
    Focus on answering: What skills or knowledge are essential? How can the reader gain them? Why are they valuable in the job market? Provide also the URLs from credible sources.
    
    IMPORTANT: Never fabricate or misrepresent information. Only include details that are explicitly found in the provided content or are general knowledge. Omit or disclaim unverified or unclear information. Start directly with the content, avoiding introductory phrases, labels, or meta-text like "Here's a summary" or "In educational content style."
    """,
    model=MODEL
)

def process_educational_content(riasec_result):
    """Run the educational content processing workflow."""
    with st.status("Processing educational content...", expanded=True) as status:
        # Generate Topic
        status.write("🔄 Generating topic based on RIASEC result...")
        topic_response = client.run(
            agent=topic_generator_agent,
            messages=[{"role": "user", "content": f"Generate a topic for RIASEC result: {riasec_result}"}]
        )
        topic = topic_response.messages[-1]["content"]
        
        # Search
        status.write("🔍 Searching for educational content...")
        search_response = client.run(
            agent=search_agent,
            messages=[{"role": "user", "content": f"Find educational content about {topic}"}]
        )
        raw_content = search_response.messages[-1]["content"]
        
        # Synthesize
        status.write("🔄 Synthesizing information...")
        synthesis_response = client.run(
            agent=synthesis_agent,
            messages=[{"role": "user", "content": f"Synthesize this educational content:\n{raw_content}"}]
        )
        synthesized_content = synthesis_response.messages[-1]["content"]
        
        # Summarize
        status.write("📝 Creating summary...")
        summary_response = client.run(
            agent=summary_agent,
            messages=[{"role": "user", "content": f"Summarize this synthesis:\n{synthesized_content}"}]
        )
        return topic, raw_content, synthesized_content, summary_response.messages[-1]["content"]

riasec_result = pd.read_csv("answers/riasec_assessment_answer.csv")\
                  .sort_values(by=["Total Score"], ascending = False)\
                  .loc[:3, "Type"]\
                  .str.cat(sep=",")

if st.button("Find Educational Content", type="primary"):
    try:
        topic, raw_content, synthesized_content, final_summary = process_educational_content(riasec_result)
        st.header(f"📚 Educational Content for: {topic}")
        st.markdown(final_summary)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
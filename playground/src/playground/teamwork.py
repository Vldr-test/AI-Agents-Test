#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

from crewai import Agent, Task, Crew, Process, LLM
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from langchain_openai import ChatOpenAI  # OpenAI integration
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini integration
from langchain_community.chat_models import ChatAnthropic  # Anthropic integration
import google.generativeai as genai  # Import the genai module
import pprint

# Global logging control
VERBOSE_LOGGING = False  # Controls CrewAI verbose output
DEBUG = True  # Controls custom debug print statements

# Load environment variables from .env
load_dotenv()

# Load API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
xai_api_key = os.getenv("XAI_API_KEY")  # For Grok
gemini_api_key = os.getenv("GOOGLE_API_KEY")  # For Gemini (Google)
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")  # For Claude (Anthropic)

# OpenAI is mandatory for the leader
if not openai_api_key:
    raise ValueError("Missing required API key (OPENAI_API_KEY) in .env")
if not xai_api_key:
    print("[WARN] XAI_API_KEY missing - Grok agent will be skipped")
if not gemini_api_key:
    print("[WARN] GOOGLE_API_KEY missing - Gemini agent will be skipped")
if not anthropic_api_key:
    print("[WARN] ANTHROPIC_API_KEY missing - Anthropic agent will be skipped")

# Scoring criteria dictionary
SCORING_CRITERIA_BY_TYPE = {
    "translation": ["accuracy", "fluency", "conciseness"],
    "creative_writing": ["creativity", "coherence", "engagement"],
    "programming": ["accuracy", "efficiency", "readability"],
    "unknown": ["accuracy", "coherence", "relevance"]
}

# Leader Agent - uses OpenAI
leader_agent = Agent(
    role="Team Leader",
    goal="Coordinate team, analyze query type, select criteria, and pick best LLM output.",
    backstory="Experienced manager and decision-maker.",
    verbose=VERBOSE_LOGGING,
    llm=ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
)

# Worker Agents
workers = []
worker_names = []

# OpenAI worker
openai_agent = Agent(
    role="Worker",
    goal="Generate your own responses and thouroughly and critically review other responses.",
    backstory="Experienced thinker and decision-maker, known for always fair and unbiased opinions.",
    verbose=VERBOSE_LOGGING,
    llm=ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key,temperature=1.0)
)

workers.append(openai_agent)
worker_names.append("OpenAI")

# Grok worker (placeholder - fallback to OpenAI)
if xai_api_key:
    grok_agent = Agent(
        role="Worker",
        goal="Generate your own responses and thouroughly and critically review other responses.",
        backstory="Experienced thinker and decision-maker, known for always fair and unbiased opinions.",
        verbose=VERBOSE_LOGGING,
        llm=ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)  # Fallback
    )
    workers.append(grok_agent)
    worker_names.append("Grok")
    if DEBUG:
        print("[INFO] Grok using OpenAI fallback - xAI integration pending")

# Google Gemini worker
if gemini_api_key:
    gemini_agent = Agent(
        role="Worker",
        goal="Generate your own responses and thouroughly and critically review other responses.",
        backstory="Experienced thinker and decision-maker, known for always fair and unbiased opinions.",
        verbose=VERBOSE_LOGGING,
        llm=LLM(model="gemini/gemini-2.0-flash", api_key=gemini_api_key, temperature=0.3)
    )
    workers.append(gemini_agent)
    worker_names.append("Gemini")

# Anthropic worker
if anthropic_api_key:
    anthropic_agent = Agent(
        role="Worker",
        goal="Generate your own responses and thouroughly and critically review other responses.",
        backstory="Experienced thinker and decision-maker, known for always fair and unbiased opinions.",
        verbose=VERBOSE_LOGGING,
        llm=ChatAnthropic(model="claude-3-5-sonnet-20241022", anthropic_api_key=anthropic_api_key)
    )
    workers.append(anthropic_agent)
    worker_names.append("Anthropic")

if not workers:
    raise ValueError("No worker agents available - at least one is required")

# Task factory functions
def create_generation_task(agent, query):
    return Task(
        description=f"Generate a unique answer for '{query}' specific to {agent.role} \
            (model: {agent.llm.model if hasattr(agent.llm, 'model') else 'unknown'}). Avoid repeating other responses.",
        expected_output="A concise response to the query.",
        human_input=False,
        agent=agent
    )

def create_review_task(agent, results, criteria):
    return Task(
        description=f"As {agent.role} with expertise in {agent.goal}, review these results: {json.dumps(results)}. "
                    f"Score each (1-10) on {criteria} strictly from your own perspective. Dare to be different!"
                    f"Be brutally critical: 1-5 baseline (mediocre gets 3-4), only 6+ for exceptional work; justify scores above 5 with specific strengths. "
                    f"Return *only* a JSON object in this exact format: "
                    f"{{{', '.join(f'\"{name}\": {{{', '.join(f'\"{c}\": score' for c in criteria)}}}' for name in worker_names)}, "
                    f"'avg': {{{', '.join(f'\"{name}\": avg' for name in worker_names)}}}}}. Ensure all criteria are scored.",
        expected_output="A JSON dict with scores for each model and their averages.",
        human_input=False,
        agent=agent
    )

# Leader tasks
criteria_dict = json.dumps(SCORING_CRITERIA_BY_TYPE)
leader_analysis_task = Task(
    description="Analyze query: '{query}'. Identify type (translation, creative_writing, programming, unknown). "
                "Decide if team should process (default: yes). Use criteria from SCORING_CRITERIA_BY_TYPE: {criteria_dict}. "
                "Return JSON: {{'decision': 'Yes'/'No, solo', 'type': query_type, 'criteria': criteria_list}}.",
    expected_output="JSON list of decision, query type, and criteria.",
    human_input=False,
    agent=leader_agent
)

voting_summary_task = Task(
    description="Collect results: {results}. Peer reviews: {reviews}. Tally avg scores per model. "
                "Pick winner (highest avg). If tie, choose based on coherence. "
                "Log to 'voting_log.json' as {{'query_type': '{query_type}', 'scores': {{model: avg}}, 'winner': model}}. "
                "Return this exact format: 'Winner: {{model_name}}, Output: [full story text from results for {{model_name}}], Score: {{avg_score}}'",
    expected_output="Winner: {model_name}, Output: {output}, Score: {avg_score}",
    human_input=False,
    agent=leader_agent
)

# Main comparison function
def run_comparison(query):
    """Run the LLM comparison process for a given query."""
    if DEBUG:
        print(f"[DEBUG] Starting comparison for query: {query}")
        print(f"[DEBUG] Available workers: {worker_names}")

    # Step 1: Leader analyzes query
    analysis_crew = Crew(agents=[leader_agent], tasks=[leader_analysis_task], process=Process.sequential, verbose=VERBOSE_LOGGING)
    analysis_crew.tasks[0].description = analysis_crew.tasks[0].description.format(query=query, criteria_dict=criteria_dict)
    analysis_result = json.loads(analysis_crew.kickoff().tasks_output[0].raw)
    if DEBUG:
        print(f"\n[LEADER] Analysis: {analysis_result}")

    decision, query_type, criteria = analysis_result['decision'], analysis_result['type'], analysis_result['criteria']
    if decision == "No, solo":
        if DEBUG:
            print(f"[LEADER] Handling solo: {query}")
        return leader_agent.execute_task(Task(description=f"Handle solo: {query}", expected_output="Response")).raw

    # Step 2: Generate responses
    generation_tasks = [create_generation_task(agent, query) for agent in workers]
    generation_crew = Crew(agents=workers, tasks=generation_tasks, process=Process.sequential, verbose=VERBOSE_LOGGING)
    results = generation_crew.kickoff().tasks_output
    result_dict = {name: result.raw for name, result in zip(worker_names, results)}
    if DEBUG:
        print("\n[GENERATING RESULTS]")
        pprint.pprint(result_dict, width=100)

    # Step 3: Peer review
    review_tasks = [create_review_task(agent, result_dict, criteria) for agent in workers]
    review_crew = Crew(agents=workers, tasks=review_tasks, process=Process.sequential, verbose=VERBOSE_LOGGING)
    reviews = review_crew.kickoff().tasks_output
    if DEBUG:
        for i, review in enumerate(reviews):
            print(f"[DEBUG] Review raw output from {worker_names[i]}: '{review.raw}'")
    review_dict = {
        worker_names[i]: json.loads(
            review.raw.strip()
                      .replace('```json', '')
                      .replace('```', '')
                      .replace("'", '"')  # Convert single to double quotes
                      .split('\n\n')[0]
                      .strip()
        ) 
        for i, review in enumerate(reviews)
    }
    if DEBUG:
        print("\n[REVIEWS]")
        pprint.pprint(review_dict, width=100)

    # Step 4: Leader summarizes votes and logs
    voting_crew = Crew(agents=[leader_agent], tasks=[voting_summary_task], process=Process.sequential, verbose=VERBOSE_LOGGING)
    voting_crew.tasks[0].description = voting_crew.tasks[0].description.format(results=result_dict, reviews=review_dict, query_type=query_type)
    final_result = voting_crew.kickoff().tasks_output[0].raw
    if DEBUG:
        print(f"\n[LEADER] Final result: {final_result}")

    # Log to file
    log_entry = {
        "query_type": query_type,
        "scores": {name: review_dict[name]["avg"] for name in worker_names},
        "winner": final_result,
        "timestamp": datetime.now().isoformat()
    }
    try:
        with open("voting_log.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[ERROR] Logging failed: {e}")

    return final_result

# Test run with a complex query
if __name__ == "__main__":
    #test_query = "Write a story in less than 800 words: the protagonist wakes up with a strange toy in his pocket, \
    #    with no clue how it got there. The story should explore the theme of 'desire for justice' vs 'will for power'. \
    #    It should give some charachter arch. Also, the language should be natural, not as if written by AI;)"
    test_query = "Translate the following text from Russian to English: 'Раскольников не привык к толпе и, как сказано, бежал всякого общества, особенно в последнее время. Но теперь его вдруг что-то потянуло к людям. Что-то совершалось в нем новое, и вместе с тем ощущалась какая-то жажда людей. Он так устал от целого месяца своей тоски и мрачного возбуждения, что хотя бы на одну минуту пожелал вздохнуть в другом мире, каком бы то ни было, и, несмотря на всю грязь обстановки, с удовольствием пошел теперь в кабак. Он был в каком-то смятении, когда входил в эту дверь; но смятение это длилось недолго. Хозяин кабака, как всегда, сидел у себя в каморке, но тотчас же вышел к гостям, встретил их ласково и с достоинством, хотя и с какою-то подленькою ужимкой, точно предчувствуя в них добычу. Этот человек был толст, с лысиной, с бородкой, крашенной в рыжий цвет, и с лицом, как будто распухшим от постоянного пьянства.'"
    if VERBOSE_LOGGING:
        print(f"[INPUT] Query: {test_query}")
    result = run_comparison(test_query)
#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

from crewai import Agent, Task, Crew
import os
from dotenv import load_dotenv
from crewai_tools import ScrapeWebsiteTool

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Error: OPENAI_API_KEY not set in .env file.")
#os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"

# Tools
docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://docs.wrong.com/how-to/Creating-a-Crew-and-kick-it-off/"  # Broken URL for testing
    # Swap to "https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/" for real data
)

# Debug callback functions
def debug_support_task(output):
    print(f"[DEBUG_SUPPORT_TASK] Support Task Output: {output}")
    if "no information" in str(output).lower():
        print("[DEBUG] Tool likely failed or returned no data")
    return output

def debug_qa_task(output):
    print(f"[DEBUG_QA_TASK] QA Task Output: {output}")
    if "needs revision" in str(output).lower():
        print(f"[DEBUG] QA flagged issues: {output}")
    return output

# Agents
support_agent = Agent(
    role="Senior Support Representative",
    goal="Provide the most helpful, up-to-date support possible to {customer}",
    backstory=(
        "You’re a seasoned support expert at crewAI, dedicated to helping {customer} "
        "with clear, friendly, and thorough answers. You MUST use the provided ScrapeWebsiteTool "
        "to fetch the latest information for every response, as your knowledge is not current. "
        "If the tool fails or returns no useful data, explicitly state 'no information' in your response."
    ),
    tools=[docs_scrape_tool],
    verbose=False,
    allow_delegation=False
)

qa_agent = Agent(
    role="Support Quality Assurance Specialist",
    goal="Ensure the highest quality responses for {customer}",
    backstory=(
        "You’re a meticulous QA expert at crewAI, ensuring every response for {customer} "
        "is accurate, complete, and polished."
    ),
    verbose=False,
    allow_delegation=True
)

# Tasks with debug callbacks
inquiry_task = Task(
    description=(
        "{customer} asked: {inquiry}. You MUST use the provided ScrapeWebsiteTool to fetch the latest "
        "information from the specified URL to provide a detailed, accurate, and friendly response. "
        "If the tool fails or provides no relevant data, explicitly state 'no information' in your response."
    ),
    expected_output=(
        "A comprehensive, friendly response addressing all aspects of the inquiry, based on the tool’s output, "
        "or 'no information' if the tool fails."
    ),
    agent=support_agent,
    tools=[docs_scrape_tool],
    callback=debug_support_task
)

qa_task = Task(
    description=(
        "Review the response from the Senior Support Representative for {customer}. Your job is to ensure it provides a detailed, actionable answer to the inquiry using data from the ScrapeWebsiteTool. If the response contains 'no information' or similar phrases (e.g., 'unable to retrieve'), or if it lacks specific, useful details to address the inquiry, you MUST output 'needs revision' with feedback like 'Tool failed, provide an actionable answer using alternative sources' or 'Response lacks specific details, revise with concrete information.' Only approve if the response is detailed, actionable, and directly answers the inquiry with tool-derived content; polish it if needed before approval."
    ),
    expected_output=(
        "A final, polished response ready for {customer}, or 'needs revision' with detailed feedback."
    ),
    agent=qa_agent,
    callback=debug_qa_task
)

# Crew
crew = Crew(
    agents=[support_agent, qa_agent],
    tasks=[inquiry_task, qa_task],
    verbose=False,
    memory=True
)

# Run with Explicit Iteration Tracking
inputs = {
    "customer": "DeepLearningAI",
    "inquiry": "How do I add memory to my crew?"
}
max_iterations = 3
for i in range(max_iterations):
    print(f"[START] Running Iteration {i+1} of {max_iterations}")
    result = crew.kickoff(inputs=inputs)
    print(f"Iteration {i+1} Result:\n{result}\n")
    result_str = result.raw if hasattr(result, "raw") else str(result)
    if "needs revision" in result_str.lower():
        print(f"QA requested revision: {result_str}")
        inputs["inquiry"] += f"\nRevise based on: {result_str}"
    else:
        print("QA approved the response!")
        break
    if i == max_iterations - 1:
        print(f"[END] Max iterations ({max_iterations}) reached. Final result:\n{result}")
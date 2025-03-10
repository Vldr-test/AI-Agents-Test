#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
if not openai_api_key or not serper_api_key:
    raise ValueError("OPENAI_API_KEY or SERPER_API_KEY not set in .env")
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"

# Tool
search_tool = SerperDevTool()

# Debug callback
def debug_callback(output, task_name):
    token_est = len(str(output).split()) * 1.5
    print(f"[DEBUG] {task_name} Output: {output}")
    print(f"[DEBUG] {task_name} Token Est.: ~{token_est} tokens")
    if "unable to" in str(output).lower():
        print(f"[DEBUG] {task_name} Decision: Tool failure, using fallback.")
    elif "news" in str(output).lower():
        print(f"[DEBUG] {task_name} Decision: News considered.")
    else:
        print(f"[DEBUG] {task_name} Decision: Output valid.")
    return output

# Agents
data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Analyze {stock_selection} data.",
    backstory="Market expert.",
    verbose=False,
    tools=[search_tool],
    max_tokens=500
)

trading_strategy_agent = Agent(
    role="Strategy Developer",
    goal="Develop {stock_selection} strategy.",
    backstory="Quant analyst.",
    verbose=False,
    tools=[search_tool],
    max_tokens=500
)

execution_agent = Agent(
    role="Trade Advisor",
    goal="Plan {stock_selection} trade.",
    backstory="Trade specialist.",
    verbose=False,
    tools=[search_tool],
    max_tokens=500
)

risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Assess {stock_selection} risks.",
    backstory="Risk expert.",
    verbose=False,
    tools=[search_tool],
    max_tokens=500
)

# Tasks
data_analysis_task = Task(
    description="Analyze {stock_selection} data with search_tool (top 50 words). If {news_impact_consideration}, use latest news. Estimate if tools fail.",
    expected_output="{stock_selection} trend: Up/Down + driver.",
    human_input=False,
    agent=data_analyst_agent,
    callback=lambda output: debug_callback(output, "Data Analysis")
)

strategy_development_task = Task(
    description="Develop Day Trading strategy for {stock_selection} with data, {risk_tolerance}. If {news_impact_consideration}, adjust for news. Pick one: Buy/Sell/Hold + targets near current price.",
    expected_output="{stock_selection} strategy: Buy/Sell/Hold $X-$Y + reason.",
    human_input=False,
    agent=trading_strategy_agent,
    callback=lambda output: debug_callback(output, "Strategy Development")
)

execution_planning_task = Task(
    description="Plan {stock_selection} trade per strategy. Use timing + price for March 2025.",
    expected_output="{stock_selection} execution: Buy/Sell $X, time Y.",
    human_input=False,
    agent=execution_agent,
    callback=lambda output: debug_callback(output, "Execution Planning")
)

risk_assessment_task = Task(
    description="Assess {stock_selection} risks. If {news_impact_consideration}, add news risks. Use level + mitigation.",
    expected_output="{stock_selection} risk: Low/Med/High, mitigation Z.",
    human_input=False,
    agent=risk_management_agent,
    callback=lambda output: debug_callback(output, "Risk Assessment")
)

# Crew
financial_trading_crew = Crew(
    agents=[data_analyst_agent, trading_strategy_agent, execution_agent, risk_management_agent],
    tasks=[data_analysis_task, strategy_development_task, execution_planning_task, risk_assessment_task],
    process=Process.sequential,
    verbose=False
)

# Inputs
financial_trading_inputs = {
    "stock_selection": "EPAM",
    "initial_capital": "100000",
    "risk_tolerance": "Medium",
    "trading_strategy_preference": "Day Trading",
    "news_impact_consideration": True
}

# Run with debug
print("[DEBUG] Starting Crew execution...")
try:
    result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)
    print("[DEBUG] Crew execution completed.")
    # Aggregate all task outputs
    full_result = "\n".join([f"{task.description.split(' ')[0]}: {task_output.raw}" 
                            for task, task_output in zip(financial_trading_crew.tasks, result.tasks_output)])
    print(f"\nEPAM Analysis:\n{full_result}")
except Exception as e:
    print(f"[DEBUG] Execution failed: {str(e)}")
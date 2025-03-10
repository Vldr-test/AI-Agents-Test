#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
if not openai_api_key or not serper_api_key:
    raise ValueError("OPENAI_API_KEY or SERPER_API_KEY not set in .env")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

venue_coordinator = Agent(
    role="Venue Coordinator",
    goal="Identify and book an appropriate venue based on event requirements",
    tools=[search_tool, scrape_tool],
    verbose=False,
    backstory="With a keen sense of space and logistics, you excel at finding and securing the perfect venue."
)

logistics_manager = Agent(
    role="Logistics Manager",
    goal="Manage all logistics including catering and equipment",
    tools=[search_tool, scrape_tool],
    verbose=False,
    backstory="Organized and detail-oriented, you ensure flawless logistics execution."
)

marketing_communications_agent = Agent(
    role="Marketing and Communications Agent",
    goal="Effectively market the event and communicate with participants",
    tools=[search_tool, scrape_tool],
    verbose=False,
    backstory="Creative and communicative, you maximize event exposure and participation."
)

class VenueDetails(BaseModel):
    name: str
    address: str
    capacity: int
    booking_status: str

venue_task = Task(
    description="Find a venue in {event_city} that meets criteria for {event_topic} and {expected_participants} participants.",
    expected_output="Details of a chosen venue to accommodate the event.",
    # human_input=True,
    output_json=VenueDetails,
    agent=venue_coordinator
)

logistics_task = Task(
    description="Coordinate catering and equipment for an event with {expected_participants} participants on {tentative_date}.",
    expected_output="Confirmation of logistics arrangements including catering and equipment.",
    async_execution=False,
    agent=logistics_manager
)

marketing_task = Task(
    description="Promote the {event_topic} aiming to engage at least {expected_participants} potential attendees.",
    expected_output="Report on marketing activities and attendee engagement in markdown.",
    async_execution=True,
    agent=marketing_communications_agent
)

event_management_crew = Crew(
    agents=[venue_coordinator, logistics_manager, marketing_communications_agent],
    tasks=[venue_task, logistics_task, marketing_task],
    verbose=False
)

event_details = {
    "event_topic": "Tech Innovation Conference",
    "event_description": "A gathering of tech innovators and industry leaders to explore future technologies.",
    "event_city": "Tbilisi",
    "tentative_date": "2025-09-15",
    "expected_participants": 500,
    "budget": 2000,
    "venue_type": "Conference Hall"
}

result = event_management_crew.kickoff(inputs=event_details)

from pprint import pprint
# Extract venue_task output from result
venue_output = result.tasks_output[0].json_dict  # First task’s JSON output
pprint(venue_output)
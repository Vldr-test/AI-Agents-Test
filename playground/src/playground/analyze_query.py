# analyze_query.py
# this script is used to analyze the query and get the results 

from peer_review_config import (QUERY_TYPES, ALL_TOOLS, 
            QueryAnalysisResponseFormat_1, QueryAnalysisResponseFormat_2, 
            my_name
        )
from peer_review_prompts import (get_query_analysis_prompt_1, get_query_analysis_prompt_2, 
                                 get_feedback_request_prompt)

from peer_review_utils import dict_from_str 

from langchain.tools import Tool, BaseTool
from langchain.chat_models import init_chat_model

from typing import Type, List, Dict, Optional, Tuple, Union, Any
from langchain.agents import create_tool_calling_agent, AgentExecutor
import logging
import os
from dotenv import load_dotenv

from pydantic import BaseModel, RootModel, ValidationError

class LeaderAgent:
    def __init__(self, llm_name: str):
        self.llm_name = llm_name
        self.llm = None
        # dict of {tool_name: { {"object" : Tool}, "description": str }} INSTANTIATED tools 
        # that COULD be used by the agents and the leader:
        self.available_tools = {}           # Format: {tool_name : {"object": Tool, "description": str}}   
       
        self.recommended_tools = []     # list of tools that SHOULD be used by the agents
        # self.agent = None               # tools_calling_agent 
        # self.executor  = None           # executor for running with tools 

        # create the leader llm:
        try:
            self.llm = init_chat_model(model= self.llm_name)
            logging.info(f"{my_name()}: Successfully created a model for the leader: {self.llm_name}")
        except Exception as e:  # need to be more specific about the exception type
            logging.error(f"{my_name()}: Can't create a model for the leader: {self.llm_name}")
            raise e

    #====================================================================
    #----------------------- ANALYZE_QUERY() ----------------------------
    #====================================================================
    def analyze_query(self, query: str) -> Tuple[str, str, str, List[Tool]]:
        """
        Analyze query; if usage of tools is required, use the tools. 

        Returns:
            Tuple[str, List[str], Optional[str]]: A tuple containing:
                - query_type (str): The type of the query.
                - query_class (str): The class of the query.
                - prompt_for_agents (str): An improved version of the query including tools output.
                - recommended_tools (List[str]): A list of tools to be used by agents.
                
        """

        #-------------------------------------------------------------------------------
        #--------------------------- Load all available tools: -------------------------
        #-------------------------------------------------------------------------------

        load_dotenv()
      
        # go over the list of all tools and check if they are available:
        for tool_name in ALL_TOOLS.keys():
            try:
                api_key = None
                # check if the tool requires an API key: 
                api_key_name = ALL_TOOLS[tool_name]["API_KEY"]
                if api_key_name:
                    # check if the API key is in the environment variables: 
                    api_key = os.getenv(api_key_name)
                    if not api_key: 
                        logging.error(f"{my_name()}: {tool_name}' API_KEY not found in .env file")
                        continue
                
                # if the tool doesn't require an API key or the API key is available, load it: 
                # logging.info(f"{my_name()}: Loading tool: {tool_name}, {ALL_TOOLS[tool_name]}")
                tool = ALL_TOOLS[tool_name]["tool_factory"](api_key=api_key) if api_key else ALL_TOOLS[tool_name]["tool_factory"]()
                
                # note that from here on we use the names returned by the tools, hence tool.name, not tool_name :) 
                self.available_tools[tool.name] = {"object": tool, "description": tool.description}
                logging.info(f"{my_name()}: Successfully instantiated the tool: {tool_name}")
                    
            except Exception as e:
                logging.error(f"{my_name()}: Can't load tool: {tool.name}, {e}") # shall we better instantiate one by one? 
                continue
            
        # create the prompt for the query analysis. It should accept tools names and descriptions only
        # as of now, we will pass the complete object:
        # tools_descriptions = '\n\n'.join(f"[ {k : v['description']} for k, v in self.available_tools.items()]")
        formatted_descriptions = [f"Tool Name: {name}\nDescription: {data['description']}" for name, data in self.available_tools.items()]
        tools_descriptions = "\n\n".join(formatted_descriptions) # Join with double newline for clarity
        
        tools = [ v["object"] for _, v in self.available_tools.items()]

        # Now, create the Leader with the system prompt and the tools:
        try:
            prompt = get_query_analysis_prompt_1( tools_descriptions, ", ".join(QUERY_TYPES.keys()))

            agent = create_tool_calling_agent(self.llm, tools = tools, prompt = prompt)
            executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
            logging.info(f"{my_name()}: Agent initialized.")

            # logging.info(f"{my_name()}: prompt: {query}")

            # Ask the Leader to analyze the query, ask the user for clarifications, and run the tools if needed:
            # use propmpt_1
            llm_response = self.executor.invoke({"input": query})
            logging.info(f"{my_name()}: Agent response: {llm_response}")
            response_1 = dict_from_str(llm_response["output"], QueryAnalysisResponseFormat_1)
            if response_1 is None: 
                raise ValueError(f"\n{my_name()}Failed to parse response_1")

            self.query_type = response_1.query_type
            self.query_class = response_1.query_class
            improved_query = response_1.improved_query
            logging.info(f"\n{my_name()}: query_type: {self.query_type}, query_class: {self.query_class}, Improved query: {improved_query}")

            prompt_2 = get_query_analysis_prompt_2(improved_query = improved_query, tools_descriptions_str = tools_descriptions) 
            logging.info(f"\n{my_name()}: prompt_2: {prompt_2}")

            # use simple llm call to get the prompt, no agents needed:
            llm_response = self.llm.invoke(prompt_2)
            logging.info(f"\n{my_name()}: Agent response: {llm_response.content}")

            response_2 = dict_from_str(llm_response.content, QueryAnalysisResponseFormat_2)
            if response_2 is None: 
                raise ValueError(f"\n{my_name()}Failed to parse response_2")
                
            self.prompt_for_agents = response_2.prompt_for_agents 
            
            logging.info(f"{my_name()}: prompt_for_agents: {self.prompt_for_agents}, recommended_tools: {response_2.recommended_tools}")  
        
        except Exception as e:
            logging.error(f"{my_name()}: Error running the Leader agent: {e}")
            raise e
            
        # Return tools, not the tools' names:
        for tool_name in response_2.recommended_tools:
            self.recommended_tools.append(self.available_tools[tool_name]["object"]) 
        
        logging.info(f"{my_name()} executed: query_class: {self.query_class}, query_type: {self.query_type}, Recommended tools: {self.recommended_tools}")
        
        # note that recommended_tools is a self.tools list, not a list of names:
        return self.query_type, self.query_class, self.prompt_for_agents, self.recommended_tools

    #====================================================================
    #----------------------- CHECK_WITH_HUMAN() -------------------------
    #====================================================================
    def check_with_human(self, winner_avg_score: int, winner_response: str, improvement_points: List[str]) -> str:
        """
            Check if the human is happy with the winner response
            returns human_feedback string, or None if the user decides it's enough
            The LeaderAgent can decide NOT to ask the human
        """
        prompt = get_feedback_request_prompt(winner_response = winner_response,
                winner_score = winner_avg_score,
                improvement_points = improvement_points)

        # allow the agent to call the HITL:
        hitl =  self.available_tools["HITL_tool"]["object"] if self.available_tools.get("HITL_tool") else ALL_TOOLS["HITL_tool"]["tool_factory"]()

        agent = create_tool_calling_agent(self.llm, 
                    tools = [hitl], 
                    prompt = prompt)
        executor = AgentExecutor(agent= agent, tools=[hitl], verbose=False)
        
        logging.info(f"\n{my_name()}: executor initialized")

        llm_response = executor.invoke({"input": prompt})
        logging.info(f"{my_name()}: Leader decided: {llm_response}")
        response = dict_from_str(llm_response["output"], str)
        if response is None: 
            raise ValueError(f"\n{my_name()}Failed to parse response_1")
        return None if response.human_feedback.lower() == "break" else response.human_feedback
        
        

#=================================================================================================
#----------------------------------------------- MAIN --------------------------------------------
#=================================================================================================

if __name__ == "__main__":
    # Example usage of the LeaderAgent class:
    # agent = LeaderAgent("google_genai:gemini-1.5-flash")
    agent=LeaderAgent("openai:gpt-4o-mini")
    # query = "What is the capital of France?"
    query = "Plan a weekend trip for me next month, considering weather, budget, and some fun activities, but I’m not sure where to go yet."
    query_type, query_class, prompt_for_agents, recommended_tools = agent.analyze_query(query)
    
    print(f"Query Type: {query_type}")
    print(f"Query Class: {query_class}")
    print(f"Prompt for Agents: {prompt_for_agents}")
    print(f"Recommended Tools: {', '.join(tool.name for tool in recommended_tools)}")
# analyze_query.py
# this script is used to analyze the query and get the results 

from peer_review_config import QUERY_TYPES, ALL_TOOLS, QueryAnalysisResponseFormat, my_name
from peer_review_prompts import get_query_analysis_prompt
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
                self.available_tools[tool_name] = {"object": tool, "description": tool.description}
                logging.info(f"{my_name()}: Successfully instantiated the tool: {tool_name}")
                    
            except Exception as e:
                logging.error(f"{my_name()}: Can't load tool: {tool_name}, {e}") # shall we better instantiate one by one? 
                continue
            
        # create the prompt for the query analysis. It should accept tools names and descriptions only
        # as of now, we will pass the complete object:
        # tools_descriptions = '\n\n'.join(f"[ {k : v['description']} for k, v in self.available_tools.items()]")
        formatted_descriptions = [f"Tool Name: {name}\nDescription: {data['description']}" for name, data in self.available_tools.items()]
        tools_descriptions = "\n\n".join(formatted_descriptions) # Join with double newline for clarity
        prompt = get_query_analysis_prompt( tools_descriptions, ", ".join(QUERY_TYPES.keys()))
        logging.info(f"{my_name()}: prompt: {query}")

        tools = [ v["object"] for _, v in self.available_tools.items()]

        # Now, create the Leader with the system prompt and the tools:
        try:
            self.agent = create_tool_calling_agent(self.llm, tools = tools, prompt = prompt)
            self.executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)
            logging.info(f"{my_name()}: Agent initialized.")

            # run the agent with the query:
            llm_response = self.executor.invoke({"input": query})
            logging.info(f"{my_name()}: Agent response: {llm_response}")
            response = dict_from_str(llm_response["output"], QueryAnalysisResponseFormat)
        
        except Exception as e:
            logging.error(f"{my_name()}: Error running the Leader agent: {e}")
            raise e

        if response is None:
            logging.error(f"{my_name()}: Failed to analyze query: {response}")
            raise ValueError("Failed to parse response from the Leader")
            
        logging.info(f"{my_name()}: Parsed response from the Leader: {response}")

        # Return tools, not the tools' names:
        for tool_name in response.recommended_tools:
            self.recommended_tools.append(self.available_tools[tool_name]["object"]) 
        
        logging.info(f"{my_name()} executed: query_class: {response.query_class}, query_type: {response.query_type}, Recommended tools: {response.recommended_tools}")
        
        # note that recommended_tools is a self.tools list, not a list of names:
        return response.query_type, response.query_class, response.prompt_for_agents, self.recommended_tools


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
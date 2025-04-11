# analyze_query.py
# this script is used to analyze the query and get the results 

from peer_review_config import (
            QUERY_TYPES, 
            QueryAnalysisResponseFormat_1, QueryAnalysisResponseFormat_2, 
            WinnerFormat,
            my_name
        )

from peer_review_prompts import (get_query_analysis_prompt_1, 
                                 get_query_analysis_prompt_2, 
                                 get_human_feedback_prompt)

from peer_review_utils import dict_from_str 

from langchain.tools import Tool, BaseTool
from langchain.chat_models import init_chat_model

from peer_review_tools import load_all_tools 

from typing import Type, List, Dict, Optional, Tuple, Union, Any
from langchain.agents import create_tool_calling_agent, AgentExecutor
import logging


class LeaderAgent:
    def __init__(self, llm_name: str):
        self.llm_name = llm_name
        self.llm = None
        # self.agent = None                # tools_calling_agent - will be created each time anew with a new prompt
        # self.executor  = None            # executor for running with tools 

        self.available_tools = []          # List of available tools   
        self.recommended_tools = []        # list of tools recommended for the agents
        self.tools_description = ""        # a str describing all tools, for the prompt generation 

        # create the leader llm:
        try:
            self.llm = init_chat_model(model= self.llm_name)
            logging.info(f"{my_name()}: Successfully created a model for the leader: {self.llm_name}")
             # load all avalable tools: 
            self.available_tools, self.tools_description = load_all_tools()

        except Exception as e:  # need to be more specific about the exception type
            logging.error(f"{my_name()}: Can't create a model for the leader: {self.llm_name}")
            raise e
    
    #====================================================================
    #----------------------- ANALYZE_QUERY() ----------------------------
    #====================================================================
    def analyze_query(self, query: str) -> Tuple[str, str, str, Optional[List[Tool]]]:
        """
        Analyze query; if usage of tools is required, use the tools. 

        Returns:
            Tuple[str, List[str], Optional[str]]: A tuple containing:
                - query_type (str): The type of the query.
                - query_class (str): The class of the query.
                - prompt_for_agents (str): An improved version of the query including tools output.
                - recommended_tools (List[str]): A list of tools to be used by agents. NOT USED NOW
        """

        
        # Create the tool calling Leader with the system prompt and the tools:
        try:
            prompt = get_query_analysis_prompt_1(self.tools_description, ", ".join(QUERY_TYPES.keys()))

            agent = create_tool_calling_agent(self.llm, tools = self.available_tools, prompt = prompt)
            
            executor = AgentExecutor(agent=agent, 
                                     tools=self.available_tools, verbose=False)
            
            logging.info(f"{my_name()}: Leader agent initialized.")

            # logging.info(f"{my_name()}: prompt: {query}")

            # The Leader analyzes the query, asks the user for clarifications, and runs the tools if needed:
            # -------------------------------------------------------------------------------------
            llm_response = executor.invoke({"input": query})
            logging.info(f"{my_name()}: Leader response: {llm_response}")
            response_1 = dict_from_str(llm_response["output"], QueryAnalysisResponseFormat_1)
            
            if response_1 is None: 
                raise ValueError(f"\n{my_name()} Failed to parse response_1")

            self.query_type = response_1["query_type"]
            self.query_class = response_1["query_class"]
            improved_query = response_1["improved_query"]
            logging.info(f"\n{my_name()}: query_type: {self.query_type}, query_class: {self.query_class}, Improved query: {improved_query}")
            
            # The Leader creates a prompt for the agents using tools' output and user's clarifications:
            # -------------------------------------------------------------------------------------
            prompt_2 = get_query_analysis_prompt_2(improved_query = improved_query, 
                                                   tools_descriptions_str = self.tools_description) 
            logging.info(f"\n{my_name()}: prompt_2: {prompt_2}")

            # use simple llm call to create the prompt, no agent needed:
            llm_response = self.llm.invoke(prompt_2)
            logging.info(f"\n{my_name()}: Agent response: {llm_response.content}")

            response_2 = dict_from_str(llm_response.content, QueryAnalysisResponseFormat_2)
            if response_2 is None: 
                raise ValueError(f"\n{my_name()}Failed to parse response_2")
                
            self.prompt_for_agents = response_2["prompt_for_agents"] 
            recommended_tools_names = response_2["recommended_tools"]
            for tool in self.available_tools:
                if tool.name in recommended_tools_names:
                    self.recommended_tools.append(tool)
            
            logging.info(f"{my_name()}: prompt_for_agents: {self.prompt_for_agents}, self.recommended_tools")  
        
        except Exception as e:
            logging.error(f"{my_name()}: Error running the Leader agent: {e}")
            raise e
        
        logging.info(f"{my_name()} executed: query_class: {self.query_class}, query_type: {self.query_type}, Recommended tools: {self.recommended_tools}")
        
        # note that recommended_tools is a tools list, not a list of names:
        return self.query_type, self.query_class, self.prompt_for_agents, self.recommended_tools

    
    #====================================================================
    #----------------------- CHECK_WITH_HUMAN() -------------------------
    #====================================================================
    def get_human_feedback(self, winner_info: WinnerFormat) -> str:
        """
            Check if the human is happy with the winner response
            returns human_feedback string (could be ''), or "break" if the user decides it's enough
            The LeaderAgent can decide NOT to ask the human
        """
        prompt = get_human_feedback_prompt(winner_info, self.tools_description)

         # create a tool calling agent with a different prompt this time
         # we could limit the tools to just HITL. 
        agent = create_tool_calling_agent(self.llm, 
                    tools = self.available_tools, 
                    prompt = prompt)       
        
        executor = AgentExecutor(agent= agent, tools=self.available_tools, verbose=False)
        
        logging.info(f"\n{my_name()}: executor initialized")

        llm_response = executor.invoke({"input": prompt})
        logging.info(f"{my_name()}: human feedback: {llm_response['output']}")

        response = llm_response["output"].strip()[:20].lower()
        if response.startswith('user_comments'):
            pass
        elif response.startswith('proceed'):
            pass
        else: pass


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
    query = QUERY_TYPES["CREATIVE_WRITING"]["test_query"]
    query_type, query_class, prompt_for_agents, recommended_tools = agent.analyze_query(query)
    
    print(f"Query Type: {query_type}")
    print(f"Query Class: {query_class}")
    print(f"Prompt for Agents: {prompt_for_agents}")
    print(f"Recommended Tools: {', '.join(tool.name for tool in recommended_tools)}")

    score_table = { 'foo': 12, 'bar': 12, 'aaa': 13 } 
    winner = WinnerFormat(
        name = "foo",
        response = "you are an idiot", 
        avg_score = 12,
        improvement_points = [],
        scores_table = score_table
    )
        
    agent.get_human_feedback(winner)

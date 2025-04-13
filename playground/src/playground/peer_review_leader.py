
# ====================================================================
#   peer_review_leader.py: LeaderAgent class definition
# ====================================================================

# --------------------------------------------------------------------
#                       IMPORTS: LOCAL MODULES 
# --------------------------------------------------------------------
from peer_review_config import (
            QUERY_TYPES, 
            QueryAnalysisResponseFormat,  
            ToolsRecommendationsFormat,
            WinnerFormat,
            HumanFeedbackFormat,
            ActionChoiceEnum, 
            my_name
        )

from peer_review_prompts import (get_refine_query_prompt, 
                                get_construct_prompt_with_tools, 
                                get_human_feedback_prompt)

from peer_review_utils import dict_from_str 

from langchain.tools import Tool, BaseTool
from langchain.chat_models import init_chat_model

from peer_review_tools import load_all_tools 

# --------------------------------------------------------------------
#                       IMPORTS: LIBRARIES 
# --------------------------------------------------------------------

from typing import Type, List, Dict, Optional, Tuple, Union, Any
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
import logging


#====================================================================
#                        LEADER AGENT CLASS 
#====================================================================
class LeaderAgent:
    llm_name: str = None
    llm: BaseChatModel = None
    available_tools: List[Tool] = None
    tools_description: str = None
    
    def __init__(self, llm_name: str):
        """
            Class constructor. Receives the llm_name. 
            Instantiates llm and stores tools 
        """
        self.llm_name = llm_name

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
    #                         REFINE_QUERY()  
    #====================================================================
    def refine_query(self, query: str) -> Tuple[str, str, str, Optional[List[Tool]]]:
        """
        Analyze and refine the initial query in a dialog with the human. 
        Use any tools if required.  

        Returns:
            A tuple containing:
                - query_type (str): The type of the query.
                - query_class (str): The class of the query.
                - prompt_for_agents (str): An improved version of the query including tools output.
                - recommended_tools (List[str]): A list of tools to be used by agents. NOT USED NOW
        """
        prompt_for_refinement = get_refine_query_prompt(
                                self.tools_description, ", ".join(QUERY_TYPES.keys()))

        # interact with the user to clarify the query. Use HITL_tool and other available_tools: 
        response = self._talk_to_human(query, 
                                prompt_for_refinement, QueryAnalysisResponseFormat)

        query_type = response.get("query_type")         # from QUERY_TYPES 
        query_class = response.get("query_class")       # "SIMPLE" or "COMPLEX"
        improved_query = response.get("improved_query") # new prompt for agents 
        
        # inserts tools recommendations into the "improved_query" and return prompt_for_agents with tools 
        prompt_for_agents, recommended_tools = self._construct_prompt_with_tools(improved_query)    

        return query_type, query_class, prompt_for_agents, recommended_tools  

    
    #====================================================================
    #                        CHECK_HUMAN_FEEDBACK()  
    #====================================================================
    def check_human_feedback(self, query: str, winner: WinnerFormat) -> Tuple[str, Optional[str]]:
        """
            Check if the human is happy with the winner response. 
            Returns:  
                'action': str: 
                    - 'done' - if there are no comments from the user, the flow can continue
                        'new_prompt' is empty or None 
                    - 'stop' - human asks to interrupt the flow and stop (emergency)
                        'new_prompt' is empty or None 
                    - 'continue' - human asks to repeat the loop of response generation 
                        'new_prompt' contains the renewed prompt for agents including user's feedback on ""  
            The LeaderAgent can decide NOT to ask the human
        """
        
        # show final response to the user first: 
        present_final_response(winner)

        final_response_str = f"*Name*: {winner.name}  *Response*: {winner.response} *Score*: {winner.avg_score}"
        
        prompt = get_human_feedback_prompt(query, final_response_str, self.tools_description)

        response = self._talk_to_human(query, prompt, HumanFeedbackFormat)

        if response is None or response == "": 
            logging.error(f"{my_name()}: failed to parse response {response}")
            return ActionChoiceEnum.STOP, ""
              
        action = response.get('action')
        new_prompt = response.get('new_prompt')

        logging.info(f"{my_name()}: action: {action}")
         
        # we will check the values in the calling function anyway 
        return action, new_prompt


    #--------------------------------------------------------------------
    #                       _TALK_TO_HUMAN() Helper 
    #--------------------------------------------------------------------
    def _talk_to_human(self, 
        query:str,                              # initial query or requirements 
        prompt:ChatPromptTemplate,               # prompt to execute 
        structured_output: Optional[BaseModel]  # Optional output format Pydantic validator
        ) -> Optional[Dict[str, Any]]:
        """
            Generic method to interact with the user. 
            Analyzes the query, questions the user if required, and gathers all necessary information 
            Returns: a parsed dict. The caller will deal with the structure!  
        """

        # Create the agent with the system prompt and the tools. 
        # Requires memory & usage of tools (HITL_tool mandatory), so should be run by the Executor: 
        try:
            agent = create_tool_calling_agent(
                llm = self.llm, tools = self.available_tools, prompt = prompt)
            
            executor = AgentExecutor(agent=agent, 
                                        tools= self.available_tools, verbose=False)
            
            logging.info(f"{my_name()}: Leader executor created.")
            # logging.info(f"{my_name()}: prompt: {query}")

            # The Leader analyzes the query, asks the user for clarifications, 
            # and runs the tools if needed. The dialog with the user happens here: 
            llm_response = executor.invoke({"input": query})
            logging.info(f"{my_name()}: Leader response: {llm_response}")
            
            # parse response; validate with Pydantic, but return a dict to keep it generic 
            response = dict_from_str(llm_response["output"], structured_output)
            
            if response is None: 
                logging.error(f"\n{my_name()} Failed to parse response {llm_response["output"]}")
                return None 
        
            logging.info(f"\n{my_name()}: response: {response}")
            return response

        except Exception as e:
            logging.error(f"{my_name()}: Error running the Leader: {e}")
            return None 
    

    #--------------------------------------------------------------------
    #               _CONSTRUCT_PROMPT_WITH_TOOLS() Helper  
    #--------------------------------------------------------------------
    def _construct_prompt_with_tools(self, 
                                    improved_query: str) -> Optional[List[Tool]]: 
        """
            Appends the query for agents with information on recommended_tools (could be empty)
        """
        
        logging.info(f"{my_name()} starting")
        # Create the tool calling Leader with the system prompt and the tools.
        # Does NOT require usage of any tools, so can be executed by the Agent (not by the Executor!)
        try:
        
            prompt = get_construct_prompt_with_tools(improved_query, self.tools_description)

            llm_response = self.llm.invoke(prompt)
                    
            response = dict_from_str(llm_response.content, ToolsRecommendationsFormat)
            if response is None: 
                raise ValueError(f"\n{my_name()} Failed to parse response {response}")

            prompt_for_agents = response.get("prompt_for_agents")
            recommended_tools_names = response.get("recommended_tools")
            
            recommended_tools = []

            for tool in self.available_tools:
                if tool.name in recommended_tools_names:
                    recommended_tools.append(tool)
                
            logging.info(f"{my_name()}: prompt_for_agents: {prompt_for_agents}, recommended_tools: {recommended_tools}")  
            return prompt_for_agents, recommended_tools

        except Exception as e:
            logging.error(f"{my_name()}: Error running the Leader agent: {e}") 
            return "", None

#--------------------------------------------------------------------
#               PRESENT_FINAL_RESPONSE() Helper  
#--------------------------------------------------------------------
def present_final_response(winner: WinnerFormat):
    """
        presents the final response to the user, before starting a dialog. 
    """
     # print out results: 
    print(f"Scores Table: {winner.scores_table}")
    print(f"Winner Name: {winner.name}")
    print(f"Winner Avg Score: {winner.avg_score}")
    print(f"Winner response:{winner.response}")

#=================================================================================================
#                                        MAIN 
#=================================================================================================

if __name__ == "__main__":
    # Example usage of the LeaderAgent class:
    # agent = LeaderAgent("google_genai:gemini-1.5-flash")
    agent=LeaderAgent("openai:gpt-4o-mini")
    
    # query = "What is the capital of France?"
    query = QUERY_TYPES["CREATIVE_WRITING"]["test_query"]
    query_type, query_class, prompt_for_agents, recommended_tools = agent.refine_query(query)
    
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
        
    agent.check_human_feedback(query= query, winner=winner)

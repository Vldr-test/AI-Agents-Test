# peer_review_team.py

from peer_review_config import (
    USE_REAL_THREADS, MAX_ITERATIONS, QUERY_TYPES, 
    SimpleResponseFormat, PeerReviewResponseFormat, IterationWinnerFormat, 
    my_name
)

from peer_review_prompts import (
    get_response_generation_prompt, get_peer_review_prompt, 
    get_iteration_prompt, get_review_improvement_points_prompt
)

from peer_review_utils import dict_from_str 

from analyze_query import LeaderAgent

import time
from dotenv import load_dotenv
import logging
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI


from analyze_query import LeaderAgent

import asyncio
if USE_REAL_THREADS: from concurrent.futures import ThreadPoolExecutor 
from langchain_core.language_models import BaseChatModel

from langchain.tools import BaseTool

from pydantic import BaseModel, ValidationError
import asyncio
from typing import Type, List, Dict, Optional, Tuple, Union, Any

#====================================================================
#------------------------ GET_MODEL_NAME()---------------------------
#====================================================================
def get_model_name(llm: BaseChatModel)->str:
    """ from the model, return the model name """ 
    if isinstance(llm, ChatGoogleGenerativeAI):
        name = llm.model.split("/")[1]  # e.g. models/gemini-2.0-flash 
        return name
    elif hasattr(llm, "model_name"):    # OpenAI, Anthropic, DeepSeek 
        return llm.model_name
    elif hasattr(llm, "model"):         # anyone besides Google that might use "model" instead of "model_name"
        return llm.model
    else:
        return "unknown"


#====================================================================
#------------------------ CLASS AGENTTEAM ---------------------------
#====================================================================
class AgentTeam:
    """Manages a team of LLMs for query analysis, response generation, and peer review."""

    def __init__(self):
        """Initialize team with leader and agents."""
        self.agent_models: "List[BaseChatModel]" = []
        self.leader_model: "BaseChatModel" = None
        self.state: "List[IterationWinnerFormat]" = [] #  stores info about each iteration's winner 

    #====================================================================
    #--------------------------- INITIALIZE() ---------------------------
    #====================================================================
    def initialize(self, agent_llm_names: List[str], 
                    leader_llm_name: Optional[str] = None)-> Tuple[str, List[str]]:
        """
            Initialize actual LLM instances with Langchain. 
            Args: llm agents' names. 
            Leader llm doesn't do much besides harmonizing improvement points; 
                if not provided, first instantiated agent plays the leader role  
            Return: actual leader name, actual agent names 
        """
        agent_name: str = None

        # create agents 
        for agent_name in agent_llm_names:
            try:
                self.agent_models.append(init_chat_model(agent_name))
                logging.info(f"{my_name()}: Successfully created a model for agent: {agent_name}")
            except Exception as e:
                logging.error(f"{my_name()}: Can't create a model for agent: {agent_name}: {e}")
                continue
        
        if self.agent_models is None:
            logging.error(f"{my_name()}: Can't create even one agent. Exiting")
            raise RuntimeError("Can't create even one agent. Exiting")
        
        # create leader. If name is not provided, pick the first agent: 
        if leader_llm_name is None:
            self.leader_model = self.agent_models[0]
        else:    
            try:
                self.leader_model = init_chat_model(leader_llm_name) 
                logging.info(f"{my_name()}: Successfully created a model for leader: {leader_llm_name}")
            except Exception as e:
                logging.error(f"{my_name()}: Can't create a model for leader: {leader_llm_name}")
                self.leader_model = self.agent_models[0] # fall back to the first agent

        # return actual names:
        return get_model_name(self.leader_model), [get_model_name(agent) for agent in self.agent_models]
        

    #====================================================================
    #---------------------- GENERATE_RESPONSES() ------------------------
    #====================================================================

    def generate_responses(self, query: str, 
                        tools: Optional[List[BaseTool]]) -> SimpleResponseFormat:
        # Generate the prompt for the models

        prompt = get_response_generation_prompt(query)
        
        # Log the query details for debugging
        logging.info(f"{my_name()}: {query}")

        responses = asyncio.run(self.async_generate_responses(self.agent_models, prompt))

        # no Json validation required, as this is just initial responses in free text 
        logging.info(f"{my_name()}\n completed with {responses} ")

        return responses


    #====================================================================
    #---------------------- ASYNC_GENERATE_RESPONSES() ------------------
    #====================================================================
    async def async_generate_responses(self, agents:List[BaseChatModel], prompt = str)-> Dict[str, str]:
        """
            Accepts a list of agents / models and a generic prompt (one for all models)
            Returns a dict of:
            {   
                { 'agent_name_str' : response}, where Any could be a dict or a string 
                ...
                { 'agent_name_str': response}
            }
            Does NOT validate json response. 
        """
        # Initialize an empty dictionary to store model responses
        responses = {}
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=len(agents)) if USE_REAL_THREADS else None

        # Log the query details for debugging
        logging.info(f"{my_name()}: starting")

        #--------------------------------------------------------------------------------------------
        # Define the helper function to call a model asynchronously and return its name and response.
        # Required to bind responses to model names 
        #--------------------------------------------------------------------------------------------
        async def call_model(agent):
            # Get the model's name
            name = get_model_name(agent)       
            start_time = time.time()
            # Log that we're calling this model
            logging.info(f"{my_name()}: calling model {name}")
            
            # Try to get the response from the model
            try:
                if USE_REAL_THREADS:
                    response = await loop.run_in_executor(executor, lambda: agent.invoke(prompt))
                else:
                    response = await agent.ainvoke(input=prompt)
                
                execution_time = time.time() - start_time
                logging.info(f"{my_name()}: model {name}. Execution took {execution_time:.2f} seconds")
                return name, response
            except Exception as e:
                # Log any errors that occur during the model call
                logging.error(f"{my_name()}: Error in model {name}: {e}")
                return name, None  # Return None for the response if an error occurs
            
        #------------------------ call_model ends ------------------------ 

        # Create a list of tasks, one for each agent model
        tasks = [asyncio.create_task(call_model(agent)) for agent in agents]

        start_time = time.time()
        
        # Process tasks as they complete
        for finished_task in asyncio.as_completed(tasks):
            # Await the task to get the model name and response
            name, response = await finished_task
            
            # Check if we got a valid response
            if response is not None:
                # logging.info(f"{my_name()}: Received response from {name}. It took {time.time() - start_time} seconds.")
                responses[name] = response.content  # Responses are AIMessages! 
            else:
                logging.error(f"{my_name()}: No response from {name}")

        if USE_REAL_THREADS:
            executor.shutdown(wait=True)
        
        total_processing_time = time.time() - start_time
        logging.info(f"{my_name()}\n started at {start_time}. Completed in {total_processing_time} seconds")

        # Return the dictionary of model responses
        return responses


    #====================================================================
    #---------------------- GENERATE_PEER_REVIEWS() ---------------------
    #====================================================================
    def generate_peer_reviews(self, 
        query: str, responses: dict[str, str])-> PeerReviewResponseFormat:
        """
            Perform peer review and return a PeerReviewFormat dict:
            { 
                {
                'agent_name': {'score': int, 'improvement_points': List[str] }, 
                },
                ...
                {
                'agent_name': {'score': int, 'improvement_points': List[str] } 
                }
            }
        """
        logging.info(f"{my_name()} starting")

        peer_review_prompt = get_peer_review_prompt(query=query, responses=responses)

        try:
            reviews = asyncio.run( self.async_generate_responses(
                    # we ask the Leader to review - but only if same model is not there already 
                    agents=self.agent_models + [self.leader_model if not self.leader_model in self.agent_models else None],  
                    prompt=peer_review_prompt 
                    )
            )
            # logging.info(f"{my_name()}: Async peer review generated {len(responses)} results")
        except Exception as e:
            logging.error(f"{my_name()}: Error: {e}")
            raise e
        
        parsed_reviews = {}   # will store validated dictionaries
        
        # response contains { model_name : {score:int, improvement_points:list[str]}}. So we have to give the value:
        for reviewer_name, review in reviews.items():
            # logging.info(f"{my_name()}: parsing reviewer_name: {reviewer_name}, review: {review}")
            parsed_review = dict_from_str(review, PeerReviewResponseFormat)
            
            if parsed_review is None:
                logging.error(f"{my_name()}: failed to parse JSON for {reviewer_name}.")
                continue # Skip to the next agent
            else:
                try:
                    parsed_reviews[reviewer_name] = parsed_review 
                    logging.info(f"{my_name()}: parsed_review for {reviewer_name}: {parsed_review}") 
                    # remove self-reviews if required: 
                    # ... 
                except Exception as e:
                    logging.error(f"{my_name()}: Error: {e}")
                    raise e

        if not parsed_reviews:
            logging.error(f"{my_name()}: No peer reviews available; cannot determine winner")
            raise RuntimeError("No peer reviews available") 
        else:
            return parsed_reviews 
        
    #====================================================================
    #------------------- _REVIEW_IMPROVEMENT_POINTS() -------------------
    #====================================================================
    def _review_improvement_points(self, query: str, improvement_points: List[str])-> List[str]:
        """
            Harmonize multiple improvement points - inner function 
            Return the harmonized list
        """
        logging.info(f"{my_name()}: starting")
        prompt = get_review_improvement_points_prompt(query, improvement_points)

        response = ""

        try:
            response = self.leader_model.invoke(input=prompt)
            # logging.info(f"{my_name()}: prompt {prompt}. \n response from leader model: {response.content}")
            
            # Parse the string response into a list  
            parsed_response = dict_from_str(response.content)
            if parsed_response is None:
                logging.error(f"{my_name()}: Failed to parse improvement points: {response.content}")
                return improvement_points  # Fallback to original points
        
            return parsed_response
            
        except ValidationError as e:
            logging.error(f"{my_name()}: error: {e}")
            return ""        


    #====================================================================
    #----------------------- ANALYZE_PEER_REVIEWS() ---------------------
    #====================================================================
    def analize_peer_reviews(self, 
                        query: str,     # required to harmonize improvement points 
                        peer_reviews: PeerReviewResponseFormat)->IterationWinnerFormat:
        """
            Accepts PeerReviewsRepsonseFormat object: {reviewing_agent: {'reviewed_agent': {improvement_points, score}}
            Calculates the averages and finds the winner. 
            Returns: IterationWinnerFormat         
        """
        logging.info(f"{my_name()} starting")

        score_table: "Dict[str : List[int]]" = {}               # agent_name : list[scores] 
        improvement_points_table: "Dict[str, List[str]]" = {}   # agent_name : list[improvement_points]
        avg_scores: "Dict[str: int]" = {}                       # agent_name : score

        # --- Iterate through the dictionary structure ---
        for reviewer_name, review_data in peer_reviews.items():
            
            # 'review_data' is the inner dict like { reviewed_agent_name: {'score': ..., 'improvement_points': ...} }
            for reviewed_agent_name, inner_review_dict in review_data.items():
                
                # 'inner_review_dict' is the dict {'score': ..., 'improvement_points': ...}
                # Ensure reviewed_agent_name exists in tables before appending

                if reviewed_agent_name not in score_table:
                    score_table[reviewed_agent_name] = []

                # --- Use dictionary key access ---
                score_value = inner_review_dict.get('score') 
                if score_value is not None and isinstance(score_value, int):
                    score_table[reviewed_agent_name].append(score_value)
                else:
                    logging.error(f"{my_name()}: Invalid or missing 'score' for {reviewed_agent_name} from {reviewer_name}")

                if reviewed_agent_name not in improvement_points_table:
                    improvement_points_table[reviewed_agent_name] = []

                # --- Use dictionary key access ---
                points_list = inner_review_dict.get('improvement_points', []) # Use .get with default
                if isinstance(points_list, list):
                    improvement_points_table[reviewed_agent_name].extend(points_list)
                else:
                     logging.warning(f"{my_name()}: Invalid 'improvement_points' format for {reviewed_agent_name} from {reviewer_name}")
                # --- End dictionary key access ---
                    
        for name, scores in score_table.items():
            if scores: # Avoid division by zero
                avg_scores[name] = int(sum(scores) / len(scores))
            else:
                avg_scores[name] = 0 # Assign 0 if no scores

        logging.info(f"\n{my_name()}: avg_scores: {avg_scores}")

        # find the winner with the highest avg:  
        winner_name = max(avg_scores, key=avg_scores.get)
        winner_avg_score = avg_scores[winner_name]

        # harmonize improvement points list:
        improvement_points = self._review_improvement_points(
            query = query, 
            improvement_points = improvement_points_table[winner_name])
        
        return IterationWinnerFormat(
            avg_score = winner_avg_score, 
            response = "",          # will be filled in later 
            improvement_points = improvement_points, 
            name = winner_name,              
            scores_table = avg_scores           
        )
        
            
    #====================================================================
    #----------------- GENERATE_ITERATIVE_IMPROVEMENT() -----------------
    #====================================================================    
    def generate_iterative_improvement(self, 
            query: str, 
            improvement_points:list[str], 
            response: str
        )-> SimpleResponseFormat:    
        """
            Similar to generate_response. Prompt is the only difference 
        """
        # Log the query details for debugging
        logging.info(f"{my_name()}: starting")

        prompt = get_iteration_prompt(query=query, response=response, improvement_points = improvement_points)

        responses = asyncio.run(self.async_generate_responses(self.agent_models, prompt))

        # no Json validation required, as these are just responses in free text 
        logging.info(f"{my_name()}\n completed. Responses: {responses} ")

        return responses
    
    #====================================================================
    #----------------------- IMPROVEMENTS_LOOP() ------------------------
    #====================================================================    
    def improvements_loop(self, 
                query:str,          
                tools: Optional[List[BaseTool]] = None,             # could be used in future
                max_iterations: Optional[int] = MAX_ITERATIONS      # internal iterations 
                ) -> List[IterationWinnerFormat]: 
         
        iteration = 0
        winner = None 

        while True:
          
            logging.info(f"\niteration {iteration} started")

            if iteration == 0:
                responses = self.generate_responses(query = query, tools = tools) 
            else:
                # Ensure 'winner' from the previous iteration is available
                if winner is None:
                    logging.error("Cannot proceed with iteration > 0 without a winner from the previous step.")
                    raise RuntimeError("Cannot proceed with iteration > 0 without a winner from the previous step.")
                
                responses = self.generate_iterative_improvement(query = query, 
                            improvement_points = winner.improvement_points, 
                            response = winner.response)
            
            logging.info(f"responses generated")
            
            peer_reviews = self.generate_peer_reviews(query=query, responses = responses)
            logging.info(f"peer reviews done")
      
            winner = self.analize_peer_reviews(query = query, peer_reviews = peer_reviews)
            # this field was returned empty
            winner.response = responses.get(winner.name, "Error: Winner response not found" )    
            self.state.append(winner)

            logging.info(f"{my_name()} winner found: {winner.name}")
            
            if iteration >= max_iterations-1:
                logging.info(f"{my_name()} Max iterations ({max_iterations}) reached.")
                return self.state
            else: iteration += 1


#====================================================================
#---------------------------- __MAIN__ -----------------------------
#====================================================================
if __name__ == "__main__":
    # Test instantiation
    
    # query = "Plan a weekend trip for me next month, considering weather, budget, and some fun activities, but I’m not sure where to go yet."
    query = QUERY_TYPES["OTHER"]["test_query"][1]

    # leader = LeaderAgent("openai:gpt-4o-mini")
    team = AgentTeam()
    team.initialize(leader_llm_name = "openai:gpt-4o-mini", 
                         agent_llm_names = ["google_genai:gemini-2.0-flash", "deepseek:deepseek-chat"])
 
    prompt = query 
    recommended_tools = []
    winners = team.improvements_loop(query = prompt, 
                        tools = recommended_tools, 
                        max_iterations = MAX_ITERATIONS)
    iter = 1
    for winner in winners:
        print(f"\n iteration {iter}: ")
        print(f"Winner Name: {winner.name}")
        # print(f"Winner Response: {winner.response}")
        print(f"Winner Avg Score: {winner.avg_score}")
        print(f"Winner response:{winner.response}")
        iter += 1



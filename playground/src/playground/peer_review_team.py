# peer_review_team.py

import peer_review_config as cfg
from peer_review_config import AVAILABLE_TOOLS, USE_REAL_THREADS, QueryAnalysisFormat, SimpleResponseFormat, PeerReviewResponseFormat, ImprovementPointsFormat, my_name
from peer_review_prompts import get_query_analysis_prompt, get_response_generation_prompt, get_peer_review_prompt, get_iterations_prompt, get_review_improvement_points_prompt
from peer_review_utils import dict_from_str 

import time
from dotenv import load_dotenv
import logging
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_anthropic import ChatAnthropic
#from langchain_community.chat_models import ChatOpenAI
import asyncio
if USE_REAL_THREADS: from concurrent.futures import ThreadPoolExecutor 
from langchain_core.language_models import BaseChatModel

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
        self.agent_models = []
        self.leader_model = None


    def initialize_team(self, leader_llm_name: str, agent_llm_names: List[str])-> Tuple[str, List[str]]:
        # Initialize actual LLM instances with Langchain. Return: actual leader name, actual agent names, 
        try:
            self.leader_model = init_chat_model(model= leader_llm_name)
            logging.info(f"{my_name()}: Successfully created a model for the leader: { leader_llm_name}")
        except Exception as e:  # need to be more specific about the exception type
            logging.error(f"{my_name()}: Can't create a model for the leader: { leader_llm_name}")
            raise e

        for agent_name in agent_llm_names:
            try:
                self.agent_models.append(init_chat_model(agent_name))
                logging.info(f"{my_name()}: Successfully created a model for agent: {agent_name}")
            except Exception as e:
                logging.error(f"{my_name()}: Can't create a model for agent: {agent_name}")
                continue
        
        # return actual names:
        return get_model_name(self.leader_model), [get_model_name(agent) for agent in self.agent_models]
        

    #====================================================================
    #----------------------- ANALYZE_QUERY() ----------------------------
    #====================================================================
    def analyze_query(self, query: str) -> Tuple[str, List[str], Optional[str]]:
        """
        Analyze query and return type, recommended tools, and improved query.

        Returns:
            Tuple[str, List[str], Optional[str]]: A tuple containing:
                - query_type (str): The type of the query.
                - recommended_tools (List[str]): A list of recommended tools.
                - improved_query (Optional[str]): An improved version of the query, or None.
        """
        logging.info(f"{my_name()}: Analyzing query: {query}")
        prompt = get_query_analysis_prompt(query, AVAILABLE_TOOLS)

        # enforce json output:
        # structured_model = self.leader_model.with_structured_output(QueryAnalysisFormat)

        response = ""

        try:
            llm_response = self.leader_model.invoke(input=prompt)
            logging.info(f"{my_name()}: Raw response from leader model: {response}")

            # Clean up json and return object: 
            response = dict_from_str(llm_response.content, QueryAnalysisFormat)
            if response is None:
                logging.error(f"{my_name()}: Failed to parse response from leader model: {response}")
                raise ValueError("Failed to parse response from leader model")
            
            logging.info(f"{my_name()}: Parsed response from leader model: {response}")
            
            # Access the attributes directly from the Pydantic object
            query_type = response.query_type
            recommended_tools = response.recommended_tools
            improved_query = response.improved_query

            logging.info(f"{my_name()}: Parsed response: query_type: {query_type}, recommended_tools: {recommended_tools}, improved_query: {improved_query}")

            return query_type, recommended_tools, improved_query

        except ValidationError as e:
            logging.error(f"{my_name()}: error: {e}")
            return None, None, None      # have to write recovery code!                                                              
    

    #====================================================================
    #---------------------- GENERATE_RESPONSES() ------------------------
    #====================================================================

    def generate_responses(self, query: str, query_type: str, criteria: List[str]) -> SimpleResponseFormat:
        # Generate the prompt for the models
        prompt = get_response_generation_prompt(query, query_type, criteria)
        
        # Log the query details for debugging
        logging.info(f"{my_name()}: {query}, {query_type}, {criteria}")

        responses = asyncio.run(self.async_generate_responses(self.agent_models, prompt))

        # no Json validation required, as this is just initial responses in free text 

        logging.info(f"{my_name()}\n completed with {responses} ")

        return responses


    #====================================================================
    #---------------------- ASYNC_GENERATE_RESPONSES() ------------------------
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

        # Define the helper function to call a model asynchronously and return its name and response.
        # Required to bind responses to model names 
        #--------------------------------------------------------------
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
                end_time = time.time()
                logging.info(f"{my_name()}: model {name} started at {start_time}. Execution took {end_time - start_time} seconds")
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
                logging.info(f"{my_name()}: Received response from {name}. It took {time.time() - start_time} seconds.")
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
    #-----------------------GENERATE_PEER_REVIEWS() ---------------------
    #====================================================================
    def generate_peer_reviews(self, 
        query: str, criteria: List[str], responses: dict[str, str])-> PeerReviewResponseFormat:
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

        peer_review_prompt = get_peer_review_prompt(
                query=query,
                responses=responses,
                criteria=criteria
            )

        try:
            reviews = asyncio.run( self.async_generate_responses(
                    # we ask the Leader to review - but only if same model is not there already 
                    agents=self.agent_models + [self.leader_model if not self.leader_model in self.agent_models else None],  
                    prompt=peer_review_prompt,
                    )
            )
            logging.info(f"{my_name()}: Async peer review generated {len(responses)} results")
        except Exception as e:
            logging.error(f"{my_name()}: Error: {e}")
            raise e
        
        parsed_reviews = {}   # will store validated dictionaries
        
        # response contains { model_name : {score:int, improvement_points:list[str]}}. So we have to give the value:
        for reviewer_name, review in reviews.items():
            logging.info(f"{my_name()}: parsing reviewer_name: {reviewer_name}, review: {review}")
            parsed_review = dict_from_str(review, PeerReviewResponseFormat)
            if parsed_review is None:
                logging.error(f"{my_name()}: failed to parse JSON for {reviewer_name}.")
                continue # Skip to the next agent
            else:
                try:
                    parsed_reviews[reviewer_name] = parsed_review 
                    logging.info(f"{my_name()}: parsed_review for {reviewer_name}: {parsed_review}") 
                    # remove self-reviews if required: 
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
            rationalize improvement points 
            return the improved list
        """
        logging.info(f"{my_name()}: starting")
        prompt = get_review_improvement_points_prompt(query, improvement_points)

        response = ""

        try:
            response = self.leader_model.invoke(input=prompt)
            logging.info(f"{my_name()}: prompt {prompt}. \n response from leader model: {response.content}")
            
            # Parse the string response into a list using dict_from_str with RootModel
            parsed_response = dict_from_str(response.content, ImprovementPointsFormat)
            if parsed_response is None:
                logging.error(f"{my_name()}: Failed to parse improvement points: {response.content}")
                return improvement_points  # Fallback to original points
        
            # Return the list of strings directly from the root
            return parsed_response.root
            
        except ValidationError as e:
            logging.error(f"{my_name()}: error: {e}")
            return ""        


#====================================================================
#----------------------- ANALYZE_PEER_REVIEWS() ---------------------
#====================================================================
    def analize_peer_reviews(self, query: str, peer_reviews: PeerReviewResponseFormat)->Tuple[dict, str, int, List[str]]:
        """
            Accepts PeerReviewsRepsonseFormat object: {reviewing_agent: {'reviewed_agent': {improvement_points, score}}
            Calculates the averages and finds the winner. 
            Returns: see below         
        """
        logging.info(f"{my_name()} starting")

        score_table = {}                # agent_name : list[int] 
        improvement_points_table = {}   # agent_name : list[str]
        avg_scores = {}                 # agent_name : int
        
        winner_avg_score = 0
        winner_improvement_points = []

        for reviewer_name, review in peer_reviews.items():
            for reviewed, inner_review in review.root.items():
                if reviewed not in score_table:
                    score_table[reviewed] = []
                score_table[reviewed].append(inner_review.score)
                if reviewed not in improvement_points_table:
                    improvement_points_table[reviewed] = []
                improvement_points_table[reviewed].extend(inner_review.improvement_points)
                    
        for name, scores in score_table.items():
            avg_scores[name] = int(sum(scores) / len(scores))

        logging.info(f"\n{my_name()}: avg_scores: {avg_scores}")

        # find the winner with the highest avg:  
        winner = max(avg_scores, key=avg_scores.get)
        winner_avg_score = avg_scores[winner]

        # enhance improvement points list:
        winner_improvement_points = self._review_improvement_points(
            query = query, improvement_points = improvement_points_table[winner])

        logging.info(f"\n{my_name()}: winner: {winner}, winner_avg_score: {winner_avg_score}")
        logging.info(f"\n{my_name()}: improvement_points type: {type(winner_improvement_points)}")
        logging.info(f"\n{my_name()}: winner_improvement_points: {winner_improvement_points}")
        
        return avg_scores, winner, winner_avg_score, winner_improvement_points
            
    #====================================================================
    #------------------GENERATE_ITERATIVE_IMPROVEMENT() -----------------
    #====================================================================    
    def generate_iterative_improvement(
        self, 
        query: str, 
        criteria:list[str], 
        improvement_points:list[str], 
        response: str, 
        user_feedback: str = None
        )-> SimpleResponseFormat:    
        """
            Similar to generate_response. Prompt is the only difference 
        """
        # Log the query details for debugging
        logging.info(f"{my_name()}: starting")

        prompt = get_iterations_prompt(query=query, criteria=criteria, response=response, improvement_points = improvement_points, user_feedback=user_feedback)

        responses = asyncio.run(self.async_generate_responses(self.agent_models, prompt))

        # no Json validation required, as this is just initial responses in free text 
        logging.info(f"{my_name()}\n completed with prompt {prompt} and responses: {responses} ")

        return responses
    

#====================================================================
#---------------------------- __MAIN__ -----------------------------
#====================================================================
if __name__ == "__main__":
    # Test instantiation
    team = AgentTeam("gemini/gemini-2.0-flash", ["gemini/gemini-2.0-flash"])
    query_type, recommended_tools, improved_query = team.analyze_query("What is the best moisturizer?")
    print(f"Query Type: {query_type}")
    print(f"Recommended Tools: {recommended_tools}")
    print(f"Improved Query: {improved_query}")
    print(team.generate_responses("Test query", "OTHER", ["accuracy"], ["Agent1", "Agent2"]))

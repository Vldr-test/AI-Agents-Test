
import os
import json
import json5
from json_repair import repair_json
import re
import inspect
from pprint import pprint
from dotenv import load_dotenv
import logging
from crewai.flow import Flow, listen, start
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase
from langchain_openai import ChatOpenAI
from crewai.llm import LLM as CrewAILLM
from pydantic import BaseModel, RootModel, ValidationError  
import asyncio
from typing import Type, List, Dict, Optional

# from langchain.output_parsers import PydanticOutputParser, OutputFixingParser


#**********************************************************************************************************************
#************************************************ GLOBAL LOGGING  *****************************************************
#**********************************************************************************************************************

VERBOSE = False

LOG_LEVEL=logging.INFO
os.environ["CREWAI_SUPPRESS_TELEMETRY"] = "true"  # CrewAI telemetry off
os.environ["LITELLM_SUPPRESS_LOGGING"] = "true"   # LiteLLM debug logs off

logging.basicConfig(
    level=LOG_LEVEL,
    handlers=[logging.StreamHandler()],  force=True  #Forces reconfiguration
    )

llm_logger = logging.getLogger("LiteLLM")
llm_logger.setLevel(LOG_LEVEL)  # Set to WARNING to reduce noise

crew_logger = logging.getLogger("crewai")
crew_logger.setLevel(LOG_LEVEL)

# will be used to report current function name: my_name()
my_name = lambda: inspect.currentframe().f_back.f_code.co_name

#**********************************************************************************************************************
#************************************************** CONFIGURATIONS ****************************************************
#**********************************************************************************************************************


# Global definition of a leader
LEADER_MODEL = "gpt-4" # "gpt-3.5-turbo"  # Can be changed to any model

LEADER_BASE_NAME = "LEAD" 
AGENT_BASE_NAME = "AGNT"   

# These are the names of models we would like to use for generating responses (and later, peer reviews)
AGENT_MODELS = [
    # "gpt-4",
    "gpt-3.5-turbo",
    "anthropic/claude-3-haiku-20240307",    # this format is required 
    "gemini/gemini-2.0-flash",              # this format is required 
    "xai/grok"
]


MODEL_PROVIDER_MAP = {
    "gpt-": "openai",
    "gemini/": "google",
    "anthropic/": "anthropic",
    "xai/": "xai"
    # in future, add DeepSeek, etc.
}



LLM_PROVIDERS = {
    "openai": {"api_key": os.getenv("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY", "llm_class": ChatOpenAI, "temperature": 0.7},
    "google": {"api_key": os.getenv("GOOGLE_API_KEY"), "env_var": "GOOGLE_API_KEY", "llm_class": CrewAILLM, "temperature": 0.7},
    "anthropic": {"api_key": os.getenv("ANTHROPIC_API_KEY"), "env_var": "ANTHROPIC_API_KEY", "llm_class": CrewAILLM, "temperature": 0.7},
    "xai": {"api_key": os.getenv("XAI_API_KEY"), "env_var": "XAI_API_KEY", "llm_class": CrewAILLM, "temperature": 1.0}
}

QUERY_TYPES = {
    "CREATIVE_WRITING": ["creativity", "originality", "coherence"],
    "SOFTWARE_PROGRAMMING": ["accuracy", "efficiency", "readability"],
    "MATH": ["precision", "clarity", "logic"],
    "TRANSLATION": ["accuracy", "fluency", "clarity", "grammar"],
    "SUMMARIZATION": ["coverage_of_key_points", "conciseness", "clarity"],
    "REAL_TIME_DATA_QUERY": ["accuracy", "factual_correctness"],
    "OTHER": ["quality"]    # will be filled in dynamically 
}

SELF_REVIEW = False    # if set to True, agents will review their own work :) 

#================================================= DATA STRUCTURES ======================================================

# unstructured LLM response 
class AgentResponseFormat(BaseModel):
  agent_name: str  
  response : str

#-----------------------------------------------------------------------------------------------------------------------

# response from query_analysis
class QueryAnalysisFormat(BaseModel):
  query_type: str
  # criteria: List[str]
  recommended_tools: Optional[List[str]] = None  # this is for future when we will start using tools 

#--------------------------------------------------------------------------------------------------------------------------

# internal class
class InnerPeerReviewFormat(BaseModel):
  improvement_points: List[str]
  score: int 


class PeerReviewFormat(BaseModel):
    agent_name: str
    reviews: Dict[str,          # peer_name
        InnerPeerReviewFormat]

#-------------------------------------------------------------------------------------------------------------------------

# internal class
class InnerPeerReviewScoresFormat(BaseModel):
  peer_name: str
  score: int

# Final response 
class FinalResponseFormat(BaseModel):
  winner_name: str  
  winner_response: str
  winner_avg_score: float 
  winner_improvement_points: List[str]
  peer_review_scores: Dict[ 
        str,  # agent_name 
        InnerPeerReviewScoresFormat
    ]

#---------------------------------------------------------------------------------------------------------------------------


#****************************************** PROMPTS FOR ALL TASKS ************************************************

def get_analysis_prompt(query, format=QueryAnalysisFormat):
    # Don't include schema_json directly—use a description instead
    return (
        f"Classify the query '{query}' into one of these types: [{', '.join(QUERY_TYPES.keys())}]. "
        f"Return a simple JSON object with two fields: 'query_type' (a string from the list above) and 'recommended_tools' (a list of strings for LangChain tools, or an empty list []). "
        f"Do not return a schema or properties. Do not include any text or markers."
    )

def get_generation_prompt(query, query_type, criteria):
    return (
        f"Generate a {query_type} response for '{query}'. "
        f"Focus on {', '.join(criteria)}. "
        "Respond in the same language as the query (e.g., if the query is in Russian, use Russian), unless explicitly requested otherwise. "
        "Return the response as a plain string — no Unicode escape characters like '\\u0412': you should use 'В' instead of '\\u0412'. "
        "Don't include any markers."
    )

def get_peer_review_prompt(query, responses, criteria):
    # Use a descriptive instruction instead of raw schema to avoid confusion
    return (
        f"Review the responses to the query '{query}'.\n"
        f"Responses: {json.dumps(responses, ensure_ascii=False, indent=2)}\n"
        f"Evaluate based on these criteria: {', '.join(criteria)}.\n"
        f"Create a single JSON object where:\n"
        f"- Keys are the agent names from the 'Responses' (e.g., 'Agent-OPENAI-gpt-3.5-turbo').\n"
        f"- Values are objects with:\n"
        f"  - 'improvement_points': a list of 2-5 full-sentence improvement suggestions.\n"
        f"  - 'score': an integer from 1 (lowest) to 10 (highest), fair but harsh.\n"
        f"Return ONLY the JSON object, no extra text.\n"
        f"**No prose, no explanations, no extra text outside the JSON.** "
    )


def get_final_response_prompt(agent_names, query, peer_reviews, criteria):
    agent1 = agent_names[0] if agent_names else "Agent1"
    return (
        f"Return ONLY a single JSON dictionary — NO prose, NO extra text, NO newlines, NO markers. "
        f"From peer reviews by {agent_names} for '{query}': {json.dumps(peer_reviews)}. "
        f"For each agent, calculate the average score from all reviews. "
        f"Average means summing all scores for an agent and dividing by the total number of reviews. "
        f"Include a 'scores_table' with each agent's scores from every reviewer. "
        f"Pick the winner with the highest average score, and set 'final_score' to that average. "
        f"Format as: {{'winner': '{agent1}', 'final_score': 7.8, 'scores_table': {{'Agent1': {{'Agent1': 8, 'Agent2': 7}}, ...}}}}"
    )


#**********************************************************************************************************************
#*************************************************** HELPER FUNCTIONS  ************************************************
#**********************************************************************************************************************

# handles working with model, provider, and agent's name. Returns: provide, name
def create_agent_name(model_name, base_name=""):
    provider_name = next((prov for prefix, prov in MODEL_PROVIDER_MAP.items() if model_name.startswith(prefix)), None)
    if not provider_name:
        logging.error(f"Unknown provider for model: {model_name}")
        return None, None
    return provider_name, f'{base_name}-{provider_name.upper()}-{model_name}' 

def short_name(agent_name):
    """Extracts the short model name from a full agent name"""
    short_model_name = agent_name

# ---- CHANGES ----
    # Extract model part based on AGENT_BASE_NAME (e.g., "AGNT-")
    if agent_name.startswith(AGENT_BASE_NAME):
        # Split on "-" and take the model part (after provider)
        parts = agent_name.split("-", 2)  # Max 3 parts: AGNT, PROVIDER, MODEL
        if len(parts) > 2:
            short_model_name = parts[2]  # e.g., "gpt-3.5-turbo", "anthropic/claude-3-haiku-20240307"
    
    # Clean up provider prefixes with "/" (e.g., "anthropic/claude-3-haiku-20240307" -> "claude-3-haiku-20240307")
    short_model_name = short_model_name.split("/")[-1]
    
    # Remove trailing version digits (e.g., "20240307" from "claude-3-haiku-20240307")
    short_model_name = re.sub(r'-\d+$', '', short_model_name)  # Matches "-20240307", not "3.5"
    
    # Remove trailing "-" if present (e.g., "claude-3-haiku-" -> "claude-3-haiku")
    short_model_name = short_model_name.rstrip("-")

    return short_model_name


#--------------------------------------------------- CREATE_AGENT_FROM_NAME() -------------------------------------------
#  Accepts model_name. Returns an initialized agent + a unique name
def create_agent_from_name(model_name, base_name=""):
    # find a provider and the name for the agent:
    provider, agent_name = create_agent_name(model_name, base_name)
    if not provider:
        logging.error(f"Unknown provider for model: {model_name}")
        return None, None

    config = LLM_PROVIDERS[provider]
    if not config["api_key"]:
        logging.error(f"Missing API key ({config['env_var']}) in .env for {provider}")
        return None, None
    try:
        llm = config["llm_class"](model=model_name, api_key=config["api_key"], temperature=config["temperature"])

        agent = Agent(
            role="agent",
            goal="Generate responses and review others' responses.",
            backstory="Experienced thinker known for fair opinions.",
            verbose=VERBOSE,
            llm=llm,
            allow_delegation=False,
            tools=[]
        )

        # Crew expects one more attribure, apparently:
        agent.config = {}  # Add at the start of the class if not initialized
        agent.config[agent_name] =  {}

        logging.info(f"{my_name()}: Initialized agent {agent_name} with model: {model_name}")
        return agent, agent_name

    except Exception as e:
        logging.error(f"Failed to initialize agent for {provider}/{model_name}: {str(e)} - Skipping")
        return None, None

#--------------------------------------------------- EXEC_SYNC() -------------------------------------------
# creates a one-agent Crew based on the prompt, and executes synchronously 
def exec_sync(agent, agent_name, prompt, expected_output):
        
        task = Task(                 # create a task for the pseudo-crew 
            description=prompt,  
            agent=agent,
            expected_output=expected_output
        )
        pseudo_crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=VERBOSE
        )
        if pseudo_crew is None:
            logging.error(f"Can't create Pseudo Crew for {agent_name}")
            raise RuntimeError(f"Can't create Crew leader for {agent_name}. Execution will be stopped")
        
        logging.info(f"Crew created for {agent_name}")
        results = pseudo_crew.kickoff()  # Synchronous
        return results.tasks_output[0].raw 


#--------------------------------------------------- EXEC_ASYNC() -------------------------------------------
# creates multiple one-agent Crews based on the prompt, and executes asynchronously 
async def exec_async(agents, agent_names, prompt, expected_output):
    tasks = [] # will store async tasks 
    for agent, agent_name in zip(agents, agent_names):  
        # creating a new task:
        task = Task(
            description=prompt, 
            agent=agent,
            expected_output=expected_output
        )
        
        # create a new pseudo-crew:
        pseudo_crew=Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=VERBOSE
        )
        
        tasks.append(asyncio.create_task(pseudo_crew.kickoff_async()))
        logging.info(f"pseudo-crew launched for {agent_name}") 
     
    # "for" loop ends. IMPROVEMENT: has to be done for each agent separately, not to waste all responses when one fails
    results_with_exceptions = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for i, result in enumerate(results_with_exceptions):
        if isinstance(result, Exception):
            logging.error(f"{my_name()}: Task {i} raised an exception: {result}")
        else:
            results.append(result)
         
    logging.info(f"Aysnc peer review Generated results: {len(results)} items - {[r.tasks_output[0].raw[:50] for r in results]}")
    return results

#---------------------------------------------- PARSING and VALIDATION  --------------------------------------------------------

def dict_from_str(llm_output: str, Pydantic_format: Optional[Type[BaseModel]] = None):
    """
    Tries to turn an LLM's output (a string) into a valid dict and validate it with a Pydantic objmodel if provided.
    Args:
        llm_output: The raw string from an LLM (could be JSON, could be messy).
        Pydantic_format: The Pydantic model we want to use for validation (optional).
    Returns: a valid dict validated with Pydantic (but NOT pydantic:) OR None  
    """
 
    json_obj = json_to_dict(llm_output) # Try to fix the JSON string and get the parsed JSON object
    if json_obj is None:
        logging.error(f"{my_name()}: Failed to fix JSON string: {llm_output}")
        return None

    if Pydantic_format:
        try:
            pydantic_obj = Pydantic_format.parse_obj(json_obj)  # Validate the JSON object with Pydantic 
            return json_obj
        except ValidationError as e:
            logging.error(f"{my_name()}: Pydantic validation failed: {e}")
            return None
    else:
        return json_obj  # without validation 
    
         
# simplest cleanup - locating outward '{' '}'
def cleanup_json(text: str) -> str:
    start = text.find('{')
    end = text.rfind('}') + 1
    if start == -1 or end == -1:
        logging.warning(f"{my_name()}: No valid JSON found for {text}")
        return None
    return text[start:end]

# deeper cleenup with recurrance 
def cleanup_json_re(text: str) -> str:
    json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)  # Capture the matched JSON string
    else: 
        logging.warning(f"{my_name()}: No valid JSON found for {text}")
        return None

# accepts a malformed JSON string from LLM output; returns a parsed dictionary or None
# accepts a malformed JSON string from LLM output; returns a parsed dictionary or None
def json_to_dict(text: str) -> Optional[dict]:
    """Converts a potentially malformed JSON string from LLM output into a dictionary."""
    json_str = cleanup_json(text)
    if json_str is None:
        json_str = cleanup_json_re(text)
        if json_str is None:
            logging.warning(f"{my_name()}: cleanup_json_re() failed to produce valid JSON for {text}")
            try:
                json_obj = json5.loads(text)
                json_str = json.dumps(json_obj)
            except ValueError as e:
                logging.warning(f"{my_name()}: json5 failed to produce valid json: {e} for {text}. Attempting repair_json()")
                try:
                    json_str = repair_json(text)
                except json.JSONDecodeError as e:
                    logging.warning(f"{my_name()}: repair_json failed to produce valid JSON: {e} for {text}")
                    return None
    
    if json_str is None:  # Redundant but explicit
        return None
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.error(f"{my_name()}: JSON final parsing failed for {text}. Error: {str(e)}")
        return None


#-------------------------------------- REMOVE_SELF_REVIEWS() --------------------------------------------------------------

# accepts  "validated_dict" in PeerReviewFormat, validates it, removes self-reviews, and returns a cleaned-up version. 
def remove_self_reviews(validated_dict, pydantic_format: Optional[Type[BaseModel]] = None):
    """Remove self-reviews from validated_dict, ensuring Pydantic compliance."""
    cleaned_up_dict = validated_dict.copy()
    if not SELF_REVIEW:
        for agent_name in list(cleaned_up_dict.keys()): # these are names of the agents that have done peer reviews 
            reviews = cleaned_up_dict[agent_name]["reviews"]
            if agent_name in reviews:
                logging.info(f"{my_name()}: Removing self-review for {agent_name}")
                del reviews[agent_name]
            if not reviews:
                logging.info(f"{my_name()}: No valid peer reviews left for {agent_name} after self-review removal")
                del cleaned_up_dict[agent_name]

    # Validate the entire cleaned-up dict after filtering, outside the loop
    if pydantic_format and cleaned_up_dict:  # Only validate if there’s data
        temp_dict = cleaned_up_dict.copy()  # Work on a temp copy to preserve original if needed
        for agent_name in list(temp_dict.keys()):
            try:
                pydantic_format.parse_obj(temp_dict[agent_name])
            except ValidationError as e:
                logging.error(f"{my_name()}: Post-filter validation failed for {agent_name}. Error: {e}")
                del temp_dict[agent_name]
        cleaned_up_dict = temp_dict  # Update only if all validations pass
        if not cleaned_up_dict:
            logging.error(f"{my_name()}: No entries remain after Pydantic validation")
    else:
        logging.info(f"{my_name()}: No reviews to validate after self-review filtering")

    return cleaned_up_dict

#**********************************************************************************************************************
#****************************************** WORKFLOW CLASS DEFINITION  ************************************************
#**********************************************************************************************************************

class TeamworkFlow(Flow):

    def __init__(self, query):
        super().__init__()
        self.state["query"] = query
        self.leader_agent = None
        self.leader_agent_name = None
        self.agents = []
        self.agent_names = []

        # create a leader: 
        leader_name = LEADER_MODEL
        self.leader_agent, self.leader_agent_name = create_agent_from_name(leader_name, LEADER_BASE_NAME)
        if self.leader_agent == None:
            logging.error(f"{my_name()}: Can't create Leader for leader_name: {leader_name}")
            raise RuntimeError(f"{my_name()}: Can't create Leader for leader_name: {leader_name}. Execution will be stopped")

        # create agents:  
        agent_models = AGENT_MODELS
        for model_name in agent_models:
            agent, agent_name = create_agent_from_name(model_name, AGENT_BASE_NAME)
            if agent is None:  
                logging.warning(f"Skipping {model_name} because the agent could not be created.")
                continue  # Skip to the next model, so that all agents are valid. If not, CrewAI fails if agent is None

            self.agents.append(agent)
            self.agent_names.append(agent_name)
            logging.info(f"{my_name()}: Created agent: {agent_name}")    


    def kickoff(self):
        logging.info("Flow kickoff started.")
        super().kickoff()
        logging.info("Flow kickoff completed.")

   
     #--------------------------------------------------- ANALYZE_QUERY() -----------------------------------------------------
    @start()
    def analyze_query(self):
        
        logging.info(f"{my_name()}: Starting analyze_query")
        self.state["analysis_state"] = {}
        
        query = self.state["query"]

        response = exec_sync(self.leader_agent, self.leader_agent_name, 
                    prompt=get_analysis_prompt(query), 
                    expected_output="one single QUERY_TYPE"
                    ) 
        # later, parse the output. Returns a valid dict matching Pydantic QueryAnalysisFormat - or None if JSON can't be recovered 
        result = dict_from_str(response, Pydantic_format=QueryAnalysisFormat)
        
        if result is None:
            logging.error(f"\n{my_name()}: Leader LLM failed to analyze the query")
            raise RuntimeError(f"\n{my_name()}: Leader LLM failed to analyze the query. Execution will be stopped")      
        
        query_type = result["query_type"].upper()
        if query_type not in QUERY_TYPES:
            logging.warning(f"{my_name()}: Invalid query_type '{query_type}', defaulting to 'OTHER'")
            query_type = "OTHER"
        self.state["analysis_state"]["query_type"] = query_type
        self.state["analysis_state"]["criteria"] = QUERY_TYPES[query_type]
        # self.state["analysis_state"]["criteria"] = result.criteria           # let the LLM override criteria in future 
        
        # Recommending TOOLS:
        self.state["analysis_state"]["recommended_tools"] = result["recommended_tools"]             

        return self.state  # Return state to ensure flow continuation

   
    #--------------------------------------------------- GENERATE_RESPONSES() -----------------------------------------------------
    @listen("analyze_query")
    async def generate_responses(self):
        """ Instantiates agents using globabl list, runs them, and adds their responses to state['responses'] """
        
        logging.info(f"{my_name()}: Starting generate_responses - Agents available: {len(self.agents)}")

        # initializing variables
        tasks = [] # will store async tasks 
          
        task_prompt = get_generation_prompt(
           query=self.state["query"],
           query_type=self.state["analysis_state"]["query_type"],
           criteria=self.state["analysis_state"]["criteria"]
        )
        self.state["generate_responses_state"] = {} 

        results = await exec_async(agents=self.agents, agent_names=self.agent_names, prompt=task_prompt, expected_output="A detailed response based on the description.")
        
        # analyze the response and fill in the state object. Since "results" are just agent responses, no JSON checkup required
        # potential issues: "result" could be None or empty 
        self.state["generate_responses_state"] = {
            agent_name: result.tasks_output[0].raw 
                for agent_name, result in zip(self.agent_names, results) if result.tasks_output is not None
        }

        logging.info(f"{my_name()}: state after generate_responses: {self.state["generate_responses_state"]}")

        return self.state  # Return state to ensure flow continuation"


#--------------------------------------------------- PEER_REVIEW() -----------------------------------------------------
    @listen("generate_responses")
    async def peer_review(self):
        
        logging.info(f"{my_name()}: Starting peer review")
        self.state["peer_review_state"] = {}
        if not self.state.get("generate_responses_state"):
            logging.error(f"{my_name()}: No generated responses available")
            return self.state

        self.state["peer_review_state"] = {}
        criteria = self.state["analysis_state"]["criteria"]
        responses = self.state["generate_responses_state"]
        logging.info(f"Available agents: {self.agent_names}")

        peer_review_prompt = get_peer_review_prompt(
            query=self.state["query"],
            responses=responses,
            criteria=criteria
        )
        logging.info(f"{my_name()}: starting peer review. Prompt: {peer_review_prompt}")

        try:
            results = await exec_async(
                agents=self.agents,
                agent_names=self.agent_names,
                prompt=peer_review_prompt,
                expected_output="a single JSON object with improvement points and scores"
            )
            logging.info(f"{my_name()}: Async peer review generated results: {len(results)} items")
        except Exception as e:
            logging.error(f"{my_name()}: Error in exec_async: {e}")
            return self.state

        validated_dict = {}   # will store validated dictionaries

        for agent_name, result in zip(self.agent_names, results):
            
            if isinstance(result, Exception):
                logging.error(f"{my_name()}: Failed to get review from {agent_name}: {str(result)}")
                continue
            
            raw_output = result.tasks_output[0].raw if result.tasks_output else "No output"
            
            logging.info(f"{my_name()}: Processing review for {agent_name}. Raw output: {raw_output}")

            json_dict = dict_from_str(raw_output)
            if json_dict is None:
                logging.error(f"{my_name()}: failed to parse JSON for {agent_name}. Raw output: {raw_output}")
                continue # Skip to the next agent

            # wrap it up properly with the "agent_name" and "reviews" keys
            validated_dict[agent_name] = {"agent_name": agent_name, "reviews": json_dict}   
        
        # FOR loop ends. Remove self-reviews if needed - and  check Pydantic compliance: 
        logging.info(f"{my_name()}: Validated peer reviews : validated_dict")
        
        validated_dict = remove_self_reviews(validated_dict, pydantic_format=PeerReviewFormat)
        if not validated_dict:
            logging.error(f"{my_name()}: No valid peer reviews found")
           
        else: 
            self.state["peer_review_state"] = validated_dict
        
        logging.info(f"\n{my_name()}: final peer review state: {self.state["peer_review_state"]}")
        # start("final_response")
        # await self._execute_listeners("final_response")
        await final_response(self)
        return self.state               # Return state to ensure flow continuation"

#--------------------------------------------------- FINAL_RESPONSE() -----------------------------------------------------
@listen("peer_review")
async def final_response(self):
    
    logging.info(f"{my_name()}: Starting final response calculation.")

    self.state["final_response_state"] = {} 
    peer_reviews_dict = self.state["peer_review_state"]

    if not peer_reviews_dict:
        logging.error(f"{my_name()}: No peer reviews available; cannot determine winner")
        self.state["final_response_state"] = {"winner": {}, "peer_review_scores": {}}
        return self.state

    scores_table = {}
    # Gather all unique agents (reviewers and reviewed)
    all_agents = set(peer_reviews_dict.keys())
    for review in peer_reviews_dict.values():
        all_agents.update(review["reviews"].keys())

    # Initialize scores_table
    for agent_name in all_agents:
        scores_table[agent_name] = {
            "scores_from_peers": {},
            "avg_score_from_peers": 0,
            "improvements_from_peers": [],
            "scores_given": {},
            "avg_score_given": 0
        }

    # Populate scores_table
    for agent_name, review in peer_reviews_dict.items():
        reviews = review["reviews"]
        for peer_name, review_data in reviews.items():
            scores_table[agent_name]["scores_given"][peer_name] = review_data["score"]
            scores_table[peer_name]["scores_from_peers"][agent_name] = review_data["score"]
            scores_table[peer_name]["improvements_from_peers"].extend(review_data["improvement_points"])

    # Calculate averages
    for agent_name in scores_table:
        scores_from = scores_table[agent_name]["scores_from_peers"]
        scores_given = scores_table[agent_name]["scores_given"]
        scores_table[agent_name]["avg_score_from_peers"] = (
            sum(scores_from.values()) / len(scores_from) if scores_from else 0
        )
        scores_table[agent_name]["avg_score_given"] = (
            sum(scores_given.values()) / len(scores_given) if scores_given else 0
        )

    # Pick the winner
    winner_name = max(scores_table, key=lambda x: scores_table[x]["avg_score_from_peers"]) if any(scores_table[a]["scores_from_peers"] for a in scores_table) else next(iter(all_agents))
    winner_avg_score = scores_table[winner_name]["avg_score_from_peers"]
    winner_response = self.state["generate_responses_state"].get(winner_name, "Response not found")
    winner_improvement_points = scores_table[winner_name]["improvements_from_peers"]

    peer_review_winner = {
        "winner": winner_name,
        "winner_score": winner_avg_score,
        "winner_response": winner_response,
        "winner_improvement_points": winner_improvement_points
    }

    self.state["final_response_state"]["winner"] = peer_review_winner
    self.state["final_response_state"]["peer_review_scores"] = scores_table
    return self.state

#**********************************************************************************************************************
#*********************************************** MAIN FUNCTION ********************************************************
#**********************************************************************************************************************

# Helper
def print_peer_review_table(peer_scores, full_agent_names, model_names):
    """Prints a formatted table of peer review scores."""
    print("\nCross-Review Table (Scores Given by Row Agent to Column Agent):")
    header = "Model Name".ljust(20) + " | " + " | ".join(name.center(12) for name in model_names)
    print(header)
    print("-" * len(header))

    for row_idx, row_agent in enumerate(model_names):
        full_row_name = full_agent_names[row_idx]
        scores = peer_scores.get(full_row_name, {}).get("scores_given", {})
        row = [row_agent.ljust(20)]
        for col_idx, col_agent in enumerate(model_names):
            full_col_name = full_agent_names[col_idx]
            if full_row_name == full_col_name and not SELF_REVIEW:
                score = "x".center(12)
            else:
                score = str(scores.get(full_col_name, "-")).center(12)
            row.append(score)
        print(" | ".join(row))

if __name__ == "__main__":
    logging.getLogger().setLevel(LOG_LEVEL)
    flow = TeamworkFlow(
        # "Напишите короткое хайку о сложности программирования агентов с ИИ"
        "Напишите историю о роботе, который спасает принцессу из замка (не более 300 слов)"
        )
    flow.kickoff()
    logging.info(f"Events dispatched: {flow.listeners.keys()}")
    
    final_state = flow.state.get("final_response_state", {})
    if not final_state:
        logging.error(f"{my_name()}: No final response state available")
        print("No final response state available.")
    else:
        json_str = json.dumps(final_state, indent=2, ensure_ascii=False)
        formatted_output = json_str.replace("\\n", "\n").replace('\\"', '"')
        print(f"\nFinal state: {formatted_output}")

        peer_scores = final_state.get("peer_review_scores", {})
        if not peer_scores:
            print("No peer review scores available.")
        else:
            # create a mapping from full agent names to short names
            full_agent_names = list(peer_scores.keys())     # these are present in the peer_scores
            logging.info(f"Full agent names: {full_agent_names}")

            agent_to_model = {}
            model_names = [short_name(full_name) for full_name in full_agent_names]
            logging.info(f"Model names: {model_names}")

            print("\nCross-Review Table (Scores Given by Row Agent to Column Agent):")
            print_peer_review_table(peer_scores, full_agent_names, model_names)



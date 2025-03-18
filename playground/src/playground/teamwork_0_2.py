
import os
import json
import json5
from json_repair import repair_json
import re
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
from typing import Type, List, Dict, Union, Optional
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser


#**********************************************************************************************************************
#************************************************ GLOBAL LOGGING  *****************************************************
#**********************************************************************************************************************

VERBOSE = False

LOG_LEVEL=logging.WARNING
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


#**********************************************************************************************************************
#************************************************** CONFIGURATIONS ****************************************************
#**********************************************************************************************************************


# Global definition of a leader
LEADER_MODEL = "gpt-4" # "gpt-3.5-turbo"  # Can be changed to any model

# These are the names of models we would like to use for generating responses (and later, peer reviews)
AGENT_MODELS = [
    # "gpt-4",
    "gpt-3.5-turbo",
    "anthropic/claude-3-haiku-20240307",  # Updated model name
    "gemini/gemini-2.0-flash",
    "xai/grok"
]

MODEL_PROVIDER_MAP = {
    "gpt-": "openai",
    "gemini/": "google",
    "anthropic/": "anthropic",
    "xai/": "xai"
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


INVALID_JSON_STR = "Invalid JSON:"

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

# this is what we expect from the LLM output, EXCEPT the "reviews" keyword. LLM will return only the Dict!
class RawPeerReviewOutput(BaseModel):
    reviews: Dict[str, InnerPeerReviewFormat] = {}


class PeerReviewFormat(BaseModel):
    agent_name: str
    reviews: Dict[str, InnerPeerReviewFormat]

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

"""
def get_analysis_prompt(query, format = QueryAnalysisFormat):

    format_instructions = format.schema_json(indent=2)
    
    return (
        f"Classify the query '{query}' into EXACTLY one of these types: [{', '.join(QUERY_TYPES.keys())}]. "
        f"If the query falls in between these types, select the one that is the best." 
        f"Return a filled in JSON object that matches this schema: {format_instructions}"
        f"For the field 'recommended_tools' generate a list of those LangChain tools that will help answer the query, or an empty list if none" 
        f"Don't include any text or markers, and don't copy the schema."
    )
"""

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
        f"If you don’t understand, return just the word 'PANIC' and nothing else."
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


#--------------------------------------------------- CREATE_AGENT_FROM_NAME() -------------------------------------------
#  Accepts model_name. Returns an initialized agent + a unique name
def create_agent_from_name(model_name, base_name=""):
    # find a provider:
    provider = next((prov for prefix, prov in MODEL_PROVIDER_MAP.items() if model_name.startswith(prefix)), None)
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

        agent_name = base_name + "-" + provider.upper() + "-" + model_name.replace('/', '-')

        # Crew expects one more attribure, apparently:
        agent.config = {}  # Add at the start of the class if not initialized
        agent.config[agent_name] =  {}

        logging.info(f"Initialized agent {agent_name} with model: {model_name}")
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
     
    # "for" loop ends
    results = await asyncio.gather(*tasks)
    logging.info(f"Aysnc peer review Generated results: {len(results)} items - {[r.tasks_output[0].raw[:50] for r in results]}")
    return results

#---------------------------------------------- PARSING and VALIDATION  --------------------------------------------------------

def pydantic_from_str(llm_output: str, Pydantic_format: Optional[Type[BaseModel]] = None, llm=None):
    """
    Tries to turn an LLM's output (a string) into a valid Pydantic object or a clean JSON string.
    Logs the original output for debugging purposes.

    Args:
        llm_output: The raw string from an LLM (could be JSON, could be messy).
        Pydantic_format: The Pydantic model we want to parse into (optional).
        llm: An LLM instance for fixing bad output (optional, e.g., the leader's LLM).

    Returns: a valid Pydantic object OR None if the object has been passed
             a valid json string OR {Invalid JSON}+llm_output if there is no Pydantic object
    """

    # Log the original output for debugging - this ensures we have the full context
    # even if processing fails later
    logging.info(f"Original LLM output: {llm_output}")

    # If we have a Pydantic model to aim for
    if Pydantic_format is not None:

        # Step 0: Ugly hack :) - Locate the start of a potential JSON object
        json_start_index = llm_output.find('{')
        if extracted_json = cleanup_json(llm_output):
            try:
                # Step 1: Validate the extracted string as proper JSON
                json.loads(extracted_json)
                logging.info(f"Extracted and validated JSON: {extracted_json}")

            except json.JSONDecodeError:
                # If the extracted JSON is invalid, log and try to repair it; 
                logging.error(f"Extracted JSON is invalid: {extracted_json}")
                extracted_json = fix_json(extracted_json)
                if extracted_json is None: 
                    # give up and log
                    return None
                else:
                    try:
                        # Try parsing the repaired JSON into the Pydantic model
                        return Pydantic_format.parse_raw(extracted_json)
                    except ValidationError as e:
                        
        else:
                # If no valid JSON structure is found after the '{', log and exit
                logging.error(f"No valid JSON structure found after '{{': {llm_output[json_start_index:]}")
                return None
    else:
        # If no '{' is found at all, log and exit
            logging.error(f"No '{{' found in LLM output: {llm_output}")
            return None

        # Step 2: Try the simplest approach—parse the string directly into the Pydantic model
        try:
            parser = PydanticOutputParser(pydantic_object=Pydantic_format)
            return parser.parse(llm_output)  # If this works, we’re done — return the Pydantic object
        except ValidationError as e:
            # If Step 2 fails (e.g., JSON is malformed or doesn’t match the model), log it
            logging.warning(f"Pydantic parsing failed for '{llm_output}': {e}")
            
            # Step 3: If we have an LLM, use it to fix the output
            if llm:
                try:
                    # Create a fixer that uses the LLM to repair the bad output
                    fixing_parser = OutputFixingParser.from_llm(
                        parser=PydanticOutputParser(pydantic_object=Pydantic_format), 
                        llm=llm
                    )
                    return fixing_parser.parse(llm_output)  # Return the fixed Pydantic object if it works
                except Exception as e:
                    # If the LLM fix fails (e.g., network issue or LLM can’t fix it), log it
                    logging.warning(f"OutputFixingParser failed: {e}")
            
            # Step 4: Brute force—try to repair the JSON without an LLM
            fixed_string = fix_json(llm_output)  # Get a hopefully valid JSON string
            if not fixed_string.startswith(INVALID_JSON_STR):  # Check if repair worked
                try:
                    # Try parsing the repaired JSON into the Pydantic model
                    return Pydantic_format.parse_raw(fixed_string)
                except ValidationError as e:
                    # If it still doesn’t fit the model, log and give up
                    logging.warning(f"Brute force recovery failed: {e}")
            else:
                # If JSON repair failed, log and skip further Pydantic parsing
                logging.warning(f"JSON repair failed, skipping Pydantic parsing")
            return None  # All attempts failed, so return None to signal defeat
    
    # If no Pydantic model is provided, just clean up the JSON and return it as a string
    else:
        try:
            # Try to parse and re-dump the string as valid JSON
            return json.dumps(json.loads(llm_output), ensure_ascii=False)
        except json.JSONDecodeError:
            # If it’s not valid JSON, fall back to repairing it
            return fix_json(llm_output)
         

def cleanup_json(text: str) -> str:
    start = text.find('{')
    end = text.rfind('}') + 1
    if start == -1 or end == -1:
        logging.warning(f"No valid JSON found }")
        return None
    return text[start:end]

def cleanup_json1(text: str) -> str:
    json_start_index = text.find('{')
    if json_start_index != -1:
        # Attempt to extract the first valid JSON object using regex
        # The regex handles nested structures to avoid cutting off mid-object
        json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text[json_start_index:], re.DOTALL)
        if json_match:
            extracted_json = json_match.group(0)  # Capture the matched JSON string
            try:
                # Step 1: Validate the extracted string as proper JSON
                json.loads(extracted_json)
                logging.warning(f"Extracted and validated JSON: {extracted_json}")
                text = extracted_json  # Update llm_output to the validated JSON
            except json.JSONDecodeError:
                # If the extracted JSON is invalid, log and exit early
                logging.error(f"Extracted JSON is invalid: {extracted_json}")
                return None
        else:
            # If no valid JSON structure is found after the '{', log and exit
            logging.error(f"No valid JSON structure found after '{{': {text[json_start_index:]}")
            return None

# accepts a broken JSON str; returns a valid json string OR "invalid JSON: {llm_output}"
def fix_json(llm_output: str) -> str:
   
    try:
        parsed = json5.loads(llm_output)
        return json.dumps(parsed, ensure_ascii=False)
    except json5.JSON5DecodeError:
        logging.warning(f"json5 failed to produce valid json. Attempting repair_json()")
        repaired = repair_json(llm_output)
        try:
            json.loads(repaired)  # Validate the repair
            return repaired
        except json.JSONDecodeError:
            logging.warning(f"repair_json failed to produce valid JSON: {repaired}")
            return f"{INVALID_JSON_STR}{llm_output}"


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
        self.leader_agent, self.leader_agent_name = create_agent_from_name(leader_name, "Leader")
        if self.leader_agent == None:
            logging.error(f"Can't create Leader for leader_name: {leader_name}")
            raise RuntimeError(f"Can't create Leader for leader_name: {leader_name}. Execution will be stopped")

        # create agents:  
        agent_models = AGENT_MODELS
        for model_name in agent_models:
            agent, agent_name = create_agent_from_name(model_name, "Agent")
            if agent is None:  
                logging.warning(f"Skipping {model_name} because the agent could not be created.")
                continue  # Skip to the next model, so that all agents are valid. If not, CrewAI fails if agent is None

            self.agents.append(agent)
            self.agent_names.append(agent_name)
            logging.info(f"Created agent: {agent_name}")    


    def kickoff(self):
        logging.info("Flow kickoff started.")
        super().kickoff()
        logging.info("Flow kickoff completed.")

   
     #--------------------------------------------------- ANALYZE_QUERY() -----------------------------------------------------
    @start()
    def analyze_query(self):
        
        logging.info("Starting analyze_query")
        self.state["analysis_state"] = {}
        
        query = self.state["query"]

        response = exec_sync(self.leader_agent, self.leader_agent_name, 
                    prompt=get_analysis_prompt(query), 
                    expected_output="one single QUERY_TYPE"
                    ) 
        # later, parse the output. Returns a valid Pydantic QueryAnalysisResponseClass or None if JSON can't be recovered 
        result = pydantic_from_str(response, Pydantic_format=QueryAnalysisFormat, llm=self.leader_agent.llm)
        
        if result is None:
            logging.error(f"\n Leader LLM failed to analyze the query")
            raise RuntimeError(f"\nLeader LLM failed to analyze the query. Execution will be stopped")      
        
        query_type = result.query_type.upper()
        if query_type not in QUERY_TYPES:
            logging.warning(f"Invalid query_type '{query_type}', defaulting to 'OTHER'")
            query_type = "OTHER"
        self.state["analysis_state"]["query_type"] = query_type
        self.state["analysis_state"]["criteria"] = QUERY_TYPES[query_type]
        # self.state["analysis_state"]["criteria"] = result.criteria           # let the LLM override criteria in future 
        
        # Recommending TOOLS:
        self.state["analysis_state"]["recommended_tools"] = result.recommended_tools             

        return self.state  # Return state to ensure flow continuation

    #--------------------------------------------------- GENERATE_RESPONSES() -----------------------------------------------------
    @listen("analyze_query")
    async def generate_responses(self):
        """ Instantiates agents using globabl list, runs them, and adds their responses to state['responses'] """
        
        logging.info(f"Starting generate_responses - Agents available: {len(self.agents)}")

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
            agent_name: result.tasks_output[0].raw # if not isinstance(result, str) else f"Invalid JSON: {result}"
                for agent_name, result in zip(self.agent_names, results)
        }

        logging.info(f"State after generate_responses: {self.state["generate_responses_state"]}")

        return self.state  # Return state to ensure flow continuation"

#--------------------------------------------------- PEER_REVIEW() -----------------------------------------------------
    @listen("generate_responses")
    async def peer_review(self):
        self.state["peer_review_state"] = {}
        criteria = self.state["analysis_state"]["criteria"]
        responses = self.state["generate_responses_state"]
        logging.info(f"Available agents: {self.agent_names}")

        peer_review_prompt = get_peer_review_prompt(
            query=self.state["query"],
            responses=responses,
            criteria=criteria
        )
        logging.info(f"Starting peer review. Prompt: {peer_review_prompt}")

        try:
            results = await exec_async(
                agents=self.agents,
                agent_names=self.agent_names,
                prompt=peer_review_prompt,
                expected_output="a single JSON object with improvement points and scores"
            )
            logging.info(f"Async peer review generated results: {len(results)} items")
        except Exception as e:
            logging.error(f"Error in exec_async: {e}")
            return self.state

        for agent_name, result in zip(self.agent_names, results):
            raw_output = result.tasks_output[0].raw if result.tasks_output else "No output"
            logging.info(f"Processing review for {agent_name}. Raw output: {raw_output}")

            try:
                # Step 1: Wrap the raw output string with "reviews"
                # Assume raw_output is a JSON string like '{"Agent-OPENAI-...": {"improvement_points": [...], "score": 6}}'
                wrapped_output = {"reviews": json.loads(raw_output)}  # Parse here to embed in dict
                wrapped_output_str = json.dumps(wrapped_output)  # Convert back to string for pydantic_from_str

                # Step 2: Validate with pydantic_from_str
                validated_review = pydantic_from_str(
                    wrapped_output_str,
                    Pydantic_format=RawPeerReviewOutput,
                    llm=self.leader_agent.llm
                )

                if validated_review is None:
                    logging.warning(f"Validation failed for {agent_name}. Raw output: {raw_output}")
                    continue

                # Step 3: Filter reviews (exclude self-reviews unless SELF_REVIEW is True)
                reviews = {
                    peer_name: review.dict()  # Convert to plain dict
                    for peer_name, review in validated_review.reviews.items()
                    if peer_name in responses and (peer_name != agent_name or SELF_REVIEW)
                }
                logging.info(f"Filtered reviews for {agent_name}: {reviews}")

                if reviews:
                    # Step 4: Create PeerReviewFormat and store it
                    peer_review = PeerReviewFormat(
                        agent_name=agent_name,
                        reviews=reviews
                    )
                    self.state["peer_review_state"][agent_name] = peer_review.dict()
                    logging.info(f"Stored review for {agent_name}: {self.state['peer_review_state'][agent_name]}")
                else:
                    logging.warning(f"No valid reviews after filtering for {agent_name}")

            except json.JSONDecodeError as e:
                logging.warning(f"JSON parsing failed for {agent_name}: {e}. Raw output: {raw_output}")
            except Exception as e:
                logging.warning(f"Error processing review for {agent_name}: {e}. Raw output: {raw_output}")

        logging.info(f"State after peer_review: {self.state['peer_review_state']}")
        return self.state

#--------------------------------------------------- FINAL_RESPONSE() -----------------------------------------------------
    @listen("peer_review")
    def final_response(self):
        # the Leader goes over the peer reviews, collects the stats and picks up the winner 
        self.state["final_response_state"] = {} 

        prompt = get_final_response_prompt(
            agent_names=self.agent_names,
            query=self.state["query"],
            peer_reviews=self.state["peer_review_state"],
            criteria=self.state["analysis_state"]["criteria"]
        )

        logging.info(f"Starting final analysis. Prompt: {prompt}")
     
        result = exec_sync(self.leader_agent, self.leader_agent_name, 
                prompt=prompt, 
                expected_output="a single JSON object with a winner and its average score"
                ) 
  
        obj = pydantic_from_str(result, Pydantic_format=FinalResponseFormat, llm=self.leader_agent.llm )
        
        if obj is None:
            logging.error(f'Final response failed: {result.tasks_output[0].raw}') 
            return self.state
        
        self.state["final_response_state"] = {
                "winner": obj.winner_name, 
                "winner_avg_score": obj.winner_avg_score,
                "winner_response": obj.winner_response,    
                "winner_improvement_points": obj.winner_improvement_points, # now they come without a reference to the agent who wrote them? 
                "peer_review_scores": obj.peer_review_scores
        }

        return self.state  # Return state to ensure flow continuation"
    

#**********************************************************************************************************************
#*********************************************** MAIN FUNCTION ********************************************************
#**********************************************************************************************************************

if __name__ == "__main__":

    # set logging level globally:
    logging.getLogger().setLevel(LOG_LEVEL)

    # flow = TeamworkFlow("Write a Python function to find the longest palindromic substring...")
    flow = TeamworkFlow("Напишите короткое хайку о сложности программирования агентов с ИИ")
    # flow = TeamworkFlow("Please give me a few practical use cases for using CrewAI and LangChain 'tools'") 
    flow.kickoff()
   
    json_str = json.dumps(flow.state, indent=2, ensure_ascii=False)
    formatted_output = json_str.replace("\\n", "\n").replace('\\"', '"')  # Replace escaped newlines with real ones and remove backslashes
    print(f"\nFinal state: {formatted_output}")
    # pprint(formatted_output, indent=2, width=80)
"""
    # Get the relevant objects from json:
    
    json_str = flow.state["peer_review_state"]  

    # re-create the PeerReviewFormat 
    peer_review_obj = PeerReviewFormat.parse_raw(json_str)
    peer_reviews = peer_review_obj.reviews
     
    scores_dict = {}  # agent_name -> list of (peer_name, score)
    improvement_points_dict = {}  # agent_name -> list of improvement points

    for review in peer_reviews:
        reviewed_agent = review.agent_name
        if reviewed_agent not in scores_dict:
            scores_dict[reviewed_agent] = []
            improvement_points_dict[reviewed_agent] = []
        
        for peer_name, review_data in review.reviews.items():
            scores_dict[reviewed_agent].append((peer_name, review_data.score))
            improvement_points_dict[reviewed_agent].extend(review_data.improvement_points)

    # Calculate average scores
    avg_scores = {}
    for agent_name, scores in scores_dict.items():
        total_score = sum(score for _, score in scores)
        avg_scores[agent_name] = total_score / len(scores)

    # Pick the winner
    winner_name = max(avg_scores, key=avg_scores.get)
    winner_avg_score = avg_scores[winner_name]

    # Get winner's response from agent_responses
    winner_response = next(agent.response for agent in agent_responses if agent.agent_name == winner_name)

    # Build peer_review_scores in the required format
    peer_review_scores = {
        agent_name: [
            InnerPeerReviewScoresFormat(peer_name=peer, score=score) for peer, score in scores
        ]
        for agent_name, scores in scores_dict.items()
    }

    # Construct the FinalResponseFormat
    final_output = FinalResponseFormat(
        winner_name=winner_name,
        winner_response=winner_response,
        winner_avg_score=winner_avg_score,
        winner_improvement_points=improvement_points_dict[winner_name],
        peer_review_scores=peer_review_scores
    )

    # Print the result
    print(json.dumps(final_output.dict(), indent=2, ensure_ascii=False))
"""

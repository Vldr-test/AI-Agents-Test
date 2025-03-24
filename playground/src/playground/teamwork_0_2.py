
import os
import json
import json5
import demjson3

from json_repair import repair_json
import re
import inspect
from pprint import pprint, pformat
from dotenv import load_dotenv
import logging
from crewai.flow import Flow, listen, start
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase
from langchain_openai import ChatOpenAI
from crewai.llm import LLM as CrewAILLM
from pydantic import BaseModel, ValidationError  
import asyncio
from typing import Type, List, Dict, Optional

import streamlit as st

# from langchain.output_parsers import PydanticOutputParser, OutputFixingParser


#===============================================================================================================================
#----------------------------------------------------- UI & LOGGING --------------------------------------------------------------
#===============================================================================================================================

USE_STREAMLIT=False

VERBOSE = False

LOG_LEVEL=logging.ERROR
os.environ["CREWAI_SUPPRESS_TELEMETRY"] = "true"  # CrewAI telemetry off
os.environ["LITELLM_SUPPRESS_LOGGING"] = "true"   # LiteLLM debug logs off

logging.basicConfig(
    level=LOG_LEVEL,
    handlers=[logging.StreamHandler()],  force=True  #Forces reconfiguration
    )

llm_logger = logging.getLogger("LiteLLM")
llm_logger.setLevel(LOG_LEVEL)  

crew_logger = logging.getLogger("crewai")
crew_logger.setLevel(LOG_LEVEL)

# will be used to report current function name: my_name()
my_name = lambda: inspect.currentframe().f_back.f_code.co_name

#===============================================================================================================================
#-------------------------------------------------- GLOBAL CONFIGURATION -------------------------------------------------------
#===============================================================================================================================


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


#===============================================================================================================================
#---------------------------------------------------- AGENTS' TOOLS ------------------------------------------------------------
#===============================================================================================================================
import crewai_tools
from crewai_tools import YoutubeVideoSearchTool, SerperDevTool, WebsiteSearchTool, CodeInterpreterTool  # need to check they match the AVAILABLE_TOOLS table

class ToolWrapper:
    def __init__(self, tool):
        self.tool = tool
        self.logger = logging.getLogger(f"ToolWrapper.{tool.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        # handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def __call__(self, *args, **kwargs):
        self.logger.info(f"Calling tool with args: {args}, kwargs: {kwargs}")
        print(f"Calling tool with args: {args}, kwargs: {kwargs}")
        result = self.tool(*args, **kwargs)
        self.logger.info(f"Tool returned: {result}")
        print(f"Tool returned: {result}")
        return result


AVAILABLE_TOOLS = {
        "YoutubeVideoSearchTool": 
            { 
            "tool_description": YoutubeVideoSearchTool().description, # "A RAG tool aimed at searching within YouTube videos, ideal for video data extraction.",
            "tool_factory": ToolWrapper(YoutubeVideoSearchTool)
            }, 
        "SerperDevTool" :{ 
            "tool_description":SerperDevTool().description, # "Designed to search the internet and return the most relevant results.",
            "tool_factory": ToolWrapper(SerperDevTool)
            },
#        "WebsiteSearchTool" : { 
#            "tool_description": WebsiteSearchTool().description, # "Performs a RAG (Retrieval-Augmented Generation) search within the content of a website.",
#           "tool_factory": ToolWrapper(WebsiteSearchTool)
#            },
        "CodeInterpreterTool" : {
            "tool_description": CodeInterpreterTool().description,
            "tool_factory": ToolWrapper(CodeInterpreterTool)
        }
    }

tool_logger = logging.getLogger("crewai.tools")
tool_logger.setLevel(logging.DEBUG)
logging.getLogger("crewai.tools.CodeInterpreterTool").setLevel(logging.DEBUG)
logging.getLogger("crewai.tools.YoutubeVideoSearchTool").setLevel(logging.DEBUG)
logging.getLogger("crewai.tools.SerperDevTool").setLevel(logging.DEBUG)
logging.getLogger("crewai.tools.WebsiteSearchTool").setLevel(logging.DEBUG)



#===============================================================================================================================
#-------------------------------------------------- PYDANTIC DATA MODELS -------------------------------------------------------
#===============================================================================================================================

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
class WinnerFormat(BaseModel):
  winner_name: str  
  winner_response: str
  winner_avg_score: float 
  winner_improvement_points: List[str]
  peer_review_scores: Dict[ 
        str,  # peer_agent_name 
        int   # score 
    ]


#===============================================================================================================================
#-------------------------------------------- PROMPTS GENERATION FOR ALL TASKS--------------------------------------------------
#===============================================================================================================================


def get_analysis_prompt(query, available_tools, format=QueryAnalysisFormat):
    tool_list = [tool_name for tool_name in AVAILABLE_TOOLS.keys()]
    prompt = (
        f"Classify the query '{query}' into one of these types: [{', '.join(QUERY_TYPES.keys())}]. "
        f"Return a simple JSON object with two fields: "
        f"- 'query_type': a string from the list above "
        f"- 'recommended_tools': a list of tool names from [{', '.join(tool_list)}], or [] if none apply. "
        f"Do not return a schema or properties. Do not include any text or markers."
    )
    logging.info(f"{my_name()}: Analysis prompt: {prompt}")
    return prompt

def get_generation_prompt(query, query_type, criteria):
    return (
        f"Generate a {query_type} response for '{query}'. "
        f"Focus on {', '.join(criteria)}. "
        f"Respond in the same language as the query (e.g., if the query is in Russian, use Russian), unless explicitly requested otherwise. "
        f"Return the response as a plain string — no Unicode escape characters like '\\u0412': you should use 'В' instead of '\\u0412'. "
        f"Don't include any markers. IMPORTANT: try to use tools if you have them. "
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

def get_second_pass_prompt(query, criteria, winner_response, improvement_points):
    return ( 
        f"Based on the query '{query}' and the criteria {', '.join(criteria)}, improve the following response:\n"
            f"{winner_response}\n"
            f", taking the following improvement points into account: {', '.join(improvement_points)}\n"
            f"Return the response as a plain string — no Unicode escape characters like '\\u0412': you should use 'В' instead of '\\u0412'. "
            f"**No prose, no explanations, no extra text outside the response.** "
        )   

################################################################################################################################
#                                                    HELPER FUNCTIONS                                                           
################################################################################################################################

#------------------------------------------------------ INFORM_USER() ----------------------------------------------
def inform_user(message:str, data: Optional[dict] = None, severity = "INFO"):


    """Inform the user with a message and optional dictionary data, formatted clearly. 
        Args:
            message (str): The message to display.
            data (dict, optional): Dictionary to format and display.
            severity (str, optional): Severity level (e.g., 'info', 'warning', 'error') for styling.
    """
    use_streamlit = USE_STREAMLIT


    def format_data(data, indent_level: int = 0) -> str:
        """Recursively format data with keys on separate lines and indented values."""
        indent = "    " * indent_level  # Four  spaces per level
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                lines.append(f"\n {indent}{key}:")
                if isinstance(value, (dict, list)):
                    lines.append(format_data(value, indent_level + 1))
                else:
                    lines.append(f"{indent}  {str(value)}")
            return " ".join(lines)
        elif isinstance(data, list):
            return " ".join(f"{indent}  - {str(x)}" for x in data)
        else:
            return f"{indent}{str(data)}"
    
    # Base output with message
    output = message
    if severity:
        output = f"[{severity.upper()}] {message}"

    # Add formatted data if provided
    if data is not None:
        formatted_data = format_data(data)
        output += f"\n\n{formatted_data}\n"

    # Output based on environment
    if use_streamlit:
        if severity == "error":
            st.error(output)
        elif severity == "warning":
            st.warning(output)
        elif severity == "info":
            st.info(output)
        elif severity == "success":
            st.success(output)
        else:
            st.write(output)
    else:
        print(output)

#===============================================================================================================================
#-------------------------------------------------- CREATING & NAMING AGENTS ---------------------------------------------------
#===============================================================================================================================

#------------------------------------------------------ CREATE_AGENT_NAME() ----------------------------------------------------
# handles working with model, provider, and agent's name. Returns: provide, name
def create_agent_name(model_name, base_name=""):
    provider_name = next((prov for prefix, prov in MODEL_PROVIDER_MAP.items() if model_name.startswith(prefix)), None)
    if not provider_name:
        logging.error(f"Unknown provider for model: {model_name}")
        return None, None
    return provider_name, f'{base_name}-{provider_name.upper()}-{model_name}' 

#---------------------------------------------------------- SHORT_NAME() -------------------------------------------------------
def short_name(agent_name):
    """Extracts the short model name from a full agent name"""
    short_model_name = agent_name

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
 
def create_agent_from_name(model_name, base_name="", tools=None):
    """ Accepts model_name. Returns an initialized agent + a unique agent's name """
    provider, agent_name = create_agent_name(model_name, base_name)
    if not provider:
        logging.error(f"Unknown provider for model: {model_name}")
        return None, None

    if tools is None:
        tools = []
      
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
            tools=tools
        )

        # Crew expects one more attribure, apparently:
        agent.config = {}  # Add at the start of the class if not initialized
        agent.config[agent_name] =  {}

        logging.info(f"{my_name()}: Initialized agent {agent_name} with model: {model_name}")
        return agent, agent_name

    except Exception as e:
        logging.error(f"{my_name()}: Failed to initialize agent for {provider}/{model_name}: {str(e)} - Skipping")
        inform_user(f"Failed to initialize agent for {provider}/{model_name}: {str(e)}", severity = "ERROR" )
        return None, None


#===============================================================================================================================
#----------------------------------------------------- CREWS EXECUTION ---------------------------------------------------------
#===============================================================================================================================

#--------------------------------------------------- EXEC_SYNC() -------------------------------------------

def exec_sync(agent, agent_name, prompt, expected_output):
    """ creates a one-agent Crew based on the prompt, and executes synchronously. Used only once for the Leader as of now""" 
   
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
        raise RuntimeError(f"Can't create Pseudo Crew for {agent_name}. Execution will be stopped")
        
    logging.info(f"Crew created for {agent_name}")
    try:
        results = pseudo_crew.kickoff()  # Synchronous
    except Exception as e:
        logging.error(f"{my_name()} - pseudo_crew for {agent_name} failed: {e}")
        inform_user(f"Couldn't launch crew for {agent_name}", severity="ERROR" )
        
    return results.tasks_output[0].raw 


#--------------------------------------------------- EXEC_ASYNC() -------------------------------------------

async def exec_async(agents, agent_names, prompt, expected_output):
    """ creates multiple one-agent Crews based on the prompt, and executes asynchronously """
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
        
        
        tasks.append(pseudo_crew.kickoff_async())  # earlier was tasks.append(asyncio.create_task(pseudo_crew.kickoff_async()))
        logging.info(f"pseudo-crew launched for {agent_name}") 
     
    # "for" loop ends. IMPROVEMENT: has to be done for each agent separately, not to waste all responses when one fails
    results_with_exceptions = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for i, result in enumerate(results_with_exceptions):
        if isinstance(result, Exception):
            # task = tasks[i]  # Get the original task from the list
            agent_name = agent_names[i] 
            logging.error(f"{my_name()}: Task raised an exception for {agent_name} : {result}")
            inform_user(f"Task {i} for {agent_name} failed to execute {agent_name}", severity="ERROR")
        else:
            results.append(result)
         
    logging.info(f"Aysnc peer review generated results: {len(results)} items - {[r.tasks_output[0].raw[:50] for r in results]}")
    return results

#===============================================================================================================================
#------------------------------------------ PARSING and VALIDATNG LLM OUTPUT----------------------------------------------------
#===============================================================================================================================

#--------------------------------------------------- DICT_FROM_STR() -----------------------------------------------------------

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
            pydantic_obj = Pydantic_format.model_validate(json_obj)  
            return json_obj
        except ValidationError as e:
            logging.error(f"{my_name()}: Pydantic validation failed: {e}")
            return None
    else:
        return json_obj  # without validation 
    
#--------------------------------------------------- CLEANUP_JSON() -----------------------------------------------------------       
def cleanup_json(text: str) -> str:
    """simplest cleanup - locating outward '{' '}' """
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

#===============================================================================================================================
#---------------------------------------------------- JSON_TO_DICT() -----------------------------------------------------------       
#===============================================================================================================================
def json_to_dict(text: str) -> Optional[dict]:
    """Converts a potentially malformed JSON string from LLM output into a dictionary or None."""
    
    text = text.replace("```json", "").replace("```", "").strip()
    # first try it simple: 
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logging.error(f"{my_name()}: JSON simple parsing failed for: {text}. Error: {str(e)}")
        try:
            return json5.loads(text)
        except ValueError as e:
            logging.error(f"{my_name()}: json5 parsing failed for: {text}. Error: {str(e)}")
            try:
                # demjson3 can attempt to repair and parse malformed JSON
                return demjson3.decode(text)
            except demjson3.JSONDecodeError as e:
                logging.error(f"{my_name()}: demjson3 simple parsing failed for: {text}. Error: {str(e)}")
        return None



################################################################################################################################
#------------------------------------------------- WORKFLOW CLASS DEFINITION ---------------------------------------------------
################################################################################################################################

#------------------------------------------------------ CONSTRUCTOR() ----------------------------------------------------------
class TeamworkFlow(Flow):

    def __init__(self, query):
        super().__init__()
        self.state["query"] = query
        self.leader_agent = None
        self.leader_agent_name = None
        self.agents = []
        self.agent_names = []
        self.tools_names = []
        self.tools=[]

        # create a leader: 
        leader_name = LEADER_MODEL
        self.leader_agent, self.leader_agent_name = create_agent_from_name(leader_name, LEADER_BASE_NAME)
        if self.leader_agent == None:
            logging.error(f"{my_name()}: Can't create Leader for leader_name: {leader_name}")
            raise RuntimeError(f"{my_name()}: Can't create Leader for leader_name: {leader_name}. Execution will be stopped")


    def kickoff(self):
        logging.info("Flow kickoff started.")
        super().kickoff()
        logging.info("Flow kickoff completed.")

#===============================================================================================================================
#------------------------------------------------------ ANALYZE_QUERY() --------------------------------------------------------
#===============================================================================================================================
    @start()
    def analyze_query(self):
        
        logging.info(f"{my_name()}: Starting analyze_query")
        self.state["analysis_state"] = {}
        
        query = self.state["query"]
        available_tools = {tool: AVAILABLE_TOOLS[tool]["tool_description"] for tool in AVAILABLE_TOOLS}

        logging.info(f"{my_name()}: available tools: {available_tools}")

        response = exec_sync(self.leader_agent, self.leader_agent_name, 
                    prompt=get_analysis_prompt(query, available_tools=available_tools), 
                    expected_output="one single QUERY_TYPE"
                    ) 
        # parse the output. Returns a valid dict matching Pydantic QueryAnalysisFormat - or None if JSON can't be recovered 
        result = dict_from_str(response, Pydantic_format=QueryAnalysisFormat)
        
        if result is None:
            logging.error(f"\n{my_name()}: Leader LLM failed to analyze the query")
            inform_user("Leader LLM failed to analyze the query", severity = "ERROR")
            raise RuntimeError(f"\n{my_name()}: Leader LLM failed to analyze the query. Execution will be stopped")      
        
        query_type = result["query_type"].upper()
        if query_type not in QUERY_TYPES:
            logging.warning(f"{my_name()}: Invalid query_type '{query_type}', defaulting to 'OTHER'")
            query_type = "OTHER"
        self.state["analysis_state"]["query_type"] = query_type
        self.state["analysis_state"]["criteria"] = QUERY_TYPES[query_type]
        # self.state["analysis_state"]["criteria"] = result.criteria           # let the LLM override criteria in future 
        
        # create the recommended TOOLS:
        recommended_tools_names = result["recommended_tools"]  # list of tools names
        for tool_name in recommended_tools_names:      
            if tool_name not in AVAILABLE_TOOLS.keys():
                logging.error(f"{my_name()}: Unknown tool recommended '{tool_name}'")
                inform_user(f"Unknown tool recommended '{tool_name}'", severity = "ERROR")
                continue
            tool = AVAILABLE_TOOLS[tool_name]["tool_factory"]()
            self.tools.append(tool)
            self.tools_names.append(tool_name)
            logging.info(f"{my_name()}: Added tool: {tool_name}")

       
        # Create agents:  
        
        agent_models = AGENT_MODELS
        for model_name in agent_models:
            agent, agent_name = create_agent_from_name(model_name, AGENT_BASE_NAME, self.tools)
            if agent is None:  
                logging.warning(f"Skipping {model_name} because the agent could not be created.")
                inform_user(f"Skipping {model_name} because the agent could not be created.", severity ="WARNING")
                continue  # Skip to the next model, so that all agents are valid. If not, CrewAI fails if agent is None

            self.agents.append(agent)
            self.agent_names.append(agent_name)
            logging.info(f"{my_name()}: Created agent: {agent_name}")   

        logging.info(f"{my_name()}: state after analyze_query: {self.state["analysis_state"]}")

        inform_user(
            message=f"Query analysis done:"
                    f"Query type: {query_type}. Recommended tools: {recommended_tools_names} \n"
                    f"Created agents: {', '.join(self.agent_names)}\n", severity = "SUCCESS"
        )
        return self.state  # Return state to ensure flow continuation


 #===============================================================================================================================  
 #--------------------------------------------------- GENERATE_RESPONSES() ------------------------------------------------------
 #===============================================================================================================================
    @listen("analyze_query")
    async def generate_responses(self):
        """ Instantiates agents using globabl list, runs them, and adds their responses to state['responses'] """
        
        logging.info(f"{my_name()}: Starting generate_responses - Agents available: {len(self.agents)}")
        inform_user(f"{len(self.agents)} agents start working on responses")

        # initializing variables
        tasks = [] # will store async tasks 
          
        task_prompt = get_generation_prompt(
           query=self.state["query"],
           query_type=self.state["analysis_state"]["query_type"],
           criteria=self.state["analysis_state"]["criteria"]
        )
        self.state["generate_responses_state"] = {} 

        try:
            results = await exec_async(agents=self.agents, agent_names=self.agent_names, prompt=task_prompt, expected_output="A detailed response based on the description.")
        except Exception as e:
            logging.error(f"{my_name()}: Error in exec_async: {e}. agent_names: {self.agent_names}, prompt: {task_prompt}")
            inform_user(f"Error while generating responses: {e}", severity = "ERROR")
            return self.state        
        
        # analyze the response and fill in the state object. Since "results" are just agent responses, no JSON checkup required
        self.state["generate_responses_state"] = {
            agent_name: result.tasks_output[0].raw 
                for agent_name, result in zip(self.agent_names, results) if result.tasks_output is not None
        }

        logging.info(f"{my_name()}: state after generate_responses: {self.state["generate_responses_state"]}")
        inform_user("Responses generated", data = self.state['generate_responses_state'], severity = "SUCCESS") 

        return self.state  # Return state to ensure flow continuation"

#===============================================================================================================================
#------------------------------------------------------- PEER_REVIEW() ---------------------------------------------------------
#===============================================================================================================================
    @listen("generate_responses")
    async def peer_review(self):

        self.state["peer_review_state"] = {}
        if not self.state.get("generate_responses_state"):
            logging.error(f"{my_name()}: No generated responses available")
            inform_user("Agents did not provide responses", severity = "ERROR")
            return self.state

        inform_user("Starting peer review")

        self.state["peer_review_state"] = {}
        criteria = self.state["analysis_state"]["criteria"]
        responses = self.state["generate_responses_state"]

        peer_review_prompt = get_peer_review_prompt(
                query=self.state["query"],
                responses=responses,
                criteria=criteria
            )
        logging.info(f"{my_name()}: peer review prompt: {peer_review_prompt}")

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
            inform_user(f"Error while doing peer review: {e}", severity = "ERROR")
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
        validated_dict =  self._remove_self_reviews(validated_dict, pydantic_format=PeerReviewFormat)
            
        if not validated_dict:
            logging.error(f"{my_name()}: No valid peer reviews found")
            logging.error(f"{my_name()}: No peer reviews available; cannot determine winner")
            self.state["peer_review_state"]["winner"] = None
            self.state["peer_review_state"]["peer_review_scores"] = None
            inform_user("No peer reviews available; cannot determine winner", severity = "ERROR")
        else: 
            self.state["peer_review_state"] = validated_dict
            # analyse the results and fill in the data structures:  
            self._analyze_peer_review()   
            
        return self.state             # Return state to ensure flow continuation"

#---------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------- _REMOVE_SELF_REVIEWS() --------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
# accepts  "validated_dict" in PeerReviewFormat, validates it, removes self-reviews, and returns a cleaned-up version. 
    def _remove_self_reviews(self, validated_dict, pydantic_format: Optional[Type[BaseModel]] = None):
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
                    pydantic_format.model_validate(temp_dict[agent_name])
                except ValidationError as e:
                    logging.error(f"{my_name()}: Post-filter validation failed for {agent_name}. Error: {e}")
                    del temp_dict[agent_name]
            cleaned_up_dict = temp_dict  # Update only if all validations pass
            if not cleaned_up_dict:
                logging.error(f"{my_name()}: No entries remain after Pydantic validation")
        else:
            logging.info(f"{my_name()}: No reviews to validate after self-review filtering")

        return cleaned_up_dict

#---------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------- _ANALYZE_PEER_REVIEW () -----------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
    def _analyze_peer_review(self):
    
        logging.info(f"{my_name()}: Starting peer review analysis.")

        peer_reviews_dict = self.state["peer_review_state"]

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
        winner_response = self.state["generate_responses_state"][winner_name]
        winner_improvement_points = scores_table[winner_name]["improvements_from_peers"]

        # consolidate the winner's improvement points: 
        # winner_improvement_points = consolidate_improvement_points(winner_improvement_points, self.leader_agent)
        
        scores_table[winner_name]["improvements_from_peers"] = winner_improvement_points

        peer_review_winner = {
            "winner_name": winner_name,
            "winner_score": winner_avg_score,
            "winner_response": winner_response,
            "winner_improvement_points": winner_improvement_points
        }

        self.state["peer_review_state"]["winner"] = peer_review_winner
        self.state["peer_review_state"]["peer_review_scores"] = build_peer_review_table(scores_table, list(all_agents))
        inform_user(f"Peer review done. The winner is: {winner_name} with a score of {winner_avg_score}.", severity = "SUCCESS" )
        inform_user("Peer review scores:", data = self.state["peer_review_state"]["peer_review_scores"], severity = "SUCCESS")
        return self.state

#================================================================================================================================
#------------------------------------------------------ SECOND_PASS() -----------------------------------------------------------
#================================================================================================================================

    @listen("peer_review")
    async def second_pass(self):
    
        logging.info(f"{my_name()}: Starting the second pass.")
       
        self.state["second_pass_state"] = {} 
    
        try:
            winner_dict = self.state["peer_review_state"]["winner"]
            winner_name = winner_dict["winner_name"]
            winner_response = winner_dict["winner_response"]
            winner_improvement_points = winner_dict["winner_improvement_points"]
             # find the agent who won the first round: 
            winner_agent = self.agents[self.agent_names.index(winner_name)] # if winner_name in self.agent_names else None 
        except Exception as e:
            logging.error(f"{my_name()}: No winner available")
            inform_user(f"State mananagement internal error: {e}", severity="ERROR")
            return self.state

        inform_user(f"Starting the second pass for the winner {winner_name}. Points for improvement recommended by peers:", data = winner_improvement_points)


        # create a new task and a new pseudo-crew for the winner:
        task_prompt = get_second_pass_prompt(
            query=self.state["query"],
            criteria=self.state["analysis_state"]["criteria"],
            winner_response=winner_response, 
            improvement_points=winner_improvement_points
        )

        task = Task(
            description=task_prompt,
            agent=winner_agent,
            expected_output="A detailed response based on the description."
        )
        pseudo_crew = Crew(
            agents=[winner_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=VERBOSE
        )

        try:
            results = await pseudo_crew.kickoff_async()
            # results = pseudo_crew.kickoff_async()
        except Exception as e:
            logging.error(f"{my_name()}: Error in pseudo_crew.kickoff_async(): {e}") 
            inform_user(f"Error while doing second pass - winner could not improve their work: {e}", severity = "ERROR")
            self.state["second_pass_state"]["response"] = None

        self.state["second_pass_state"]["response"] = results.tasks_output[0].raw
        inform_user(f"Second pass done. Final response by {winner_dict['winner_name']}:", data = self.state["second_pass_state"]["response"], severity = "SUCCESS")
        return self.state


#**********************************************************************************************************************
#*********************************************** MAIN FUNCTION ********************************************************
#**********************************************************************************************************************

# Helper
def build_peer_review_table(peer_scores, full_agent_names):
    """Prints a formatted table of peer review scores."""
    # Preprocess peer_scores to ensure all scores are strings
    for row_agent in peer_scores:
        if "scores_given" in peer_scores[row_agent]:
            peer_scores[row_agent]["scores_given"] = {
                col_agent: str(score) if score != "-" else "-"
                for col_agent, score in peer_scores[row_agent]["scores_given"].items()
            }

    # Then build table_data
    table_data = {}
    for row_agent in full_agent_names:
        scores = peer_scores.get(row_agent, {}).get("scores_given", {})
        table_data[short_name(row_agent)] = {
            short_name(col_agent): scores.get(col_agent, "-") if row_agent != col_agent or SELF_REVIEW else "x"
            for col_agent in full_agent_names
        }
    
    return table_data


if __name__ == "__main__":

    use_streamlit = USE_STREAMLIT           # os.getenv("USE_STREAMLIT", "false").lower() == "true"
    
    #query = "Напишите смешную историю о молодом крокодиле, который спасает принцессу из замка (не более 300 слов)"
    #query = "What time is it now?"
    #query = "Резюмируйте содержание видео: https://www.youtube.com/watch?v=JGwWNGJdvx8"
    query = "summarize the content of the website: https://www.epam.com as of today (March 2025)"
    #query = (f"write a python program that calculates the factorial of 'n'. Run it in a Python interpreter to make sure it works properly. "  
    #        f"Inform me how you tested it")
    flow = TeamworkFlow(query)
    final_state = {}

    try:
        flow.kickoff()
        logging.info(f"Events dispatched: {flow._listeners.keys()}")
        final_state=flow.state.get("second_pass_state", {})
    except Exception as e:
        logging.error(f"Flow failed: {str(e)}")

    logging.info(f"Final state: {flow.state}")    
  
"""
    # presenting final results: 

    if USE_STREAMLIT:

        st.title("Teamwork Flow")
            
        if st.button("Run"):
            st.subheader("Winner")
            winner = final_state["winner"]
            st.write(f"**Agent**: {short_name(winner['winner'])}")
            st.write(f"**Score**: {winner['winner_score']}")
            st.write("**Story**:")
            st.write(winner['winner_response'])
            peer_scores = final_state.get("peer_review_scores")
            st.table(peer_scores)
        else:
            st.error("No result")
    else:
        winner = final_state["peer_review_state"]["winner"]
        print(f"Winner: {short_name(winner['winner'])}")
        print(f"Score: {winner['winner_score']}")
        print("Story:")
        print(winner['winner_response'])
        peer_scores = final_state.get("peer_review_scores", {})
        if peer_scores:
            print(peer_scores)"
"""
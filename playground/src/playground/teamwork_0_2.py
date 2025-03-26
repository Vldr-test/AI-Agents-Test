
import os
import json
import json5
import demjson3
# from json_repair import repair_json
import time
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
from langchain_google_genai import ChatGoogleGenerativeAI

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
LEADER_MODEL = "gemini/gemini-2.0-flash"  # Can be changed to any model like openai/gpt-3.5-turbo

LEADER_BASE_NAME = "LEAD" 
AGENT_BASE_NAME = "AGNT"   

# These are the names of models we would like to use for generating responses (and later, peer reviews)
AGENT_MODELS = [
    # "gpt-4",
    "openai/gpt-4",
    "anthropic/claude-3-5-sonnet-latest",     
    "gemini/gemini-2.0-flash",              
    "deepseek/deepseek-chat" 
    # "deepseek/deepseek-reasoner"   doesn't work as expected 
]



MODEL_PROVIDER_MAP = {
    "gpt-": "openai",
    "openai/": "openai",
    "google/": "google",
    "gemini/": "google",
    "anthropic/": "anthropic",
    "xai/": "xai",
    "deepseek/": "deepseek"
    # in future, add DeepSeek, etc.
}

# temperature is just a default one, will be set separately after the QUERY_TYPE is known 
LLM_PROVIDERS = {
    "openai": {"api_key": os.getenv("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY", "llm_class": CrewAILLM, "temperature": 0.7},
    "google": {"api_key": os.getenv("GOOGLE_API_KEY"), "env_var": "GOOGLE_API_KEY", "llm_class": CrewAILLM, "temperature": 0.7},
    "anthropic": {"api_key": os.getenv("ANTHROPIC_API_KEY"), "env_var": "ANTHROPIC_API_KEY", "llm_class": CrewAILLM, "temperature": 0.7},
    "xai": {"api_key": os.getenv("XAI_API_KEY"), "env_var": "XAI_API_KEY", "llm_class": CrewAILLM, "temperature": 0.7},
    "deepseek": {"api_key": os.getenv("DEEPSEEK_API_KEY"), "env_var": "DEEPSEEK_API_KEY", "llm_class": CrewAILLM, "temperature": 0.7 }
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

QUERY_TYPES_TEMPERATURE = {
    "CREATIVE_WRITING": {
        "openai": 0.9,       # high creativity recommended 
        "google": 0.9,       # higher end (0.7-1.0) for creative tasks 
        "anthropic": 1.0,    # closer to 1.0 for generative tasks 
        "deepseek": 1.5,     # official recommendation for creative writing
        "xai": 1.0,          # no official value; high (range 0-1) for creativity
        "meta": 1.0,         # up to 1.0 (default 0.9) for more diverse output 
        "mistral": 0.8       # higher end (default 0.7) for creative output 
    },
    "SOFTWARE_PROGRAMMING": {
        "openai": 0.0,       # use 0 for deterministic code generation
        "google": 0.0,       # low (0-0.3) for deterministic outputs like code 
        "anthropic": 0.0,    # closer to 0 for analytical tasks  
        "deepseek": 0.0,     # coding/math recommended at 0.0 
        "xai": 0.0,          # no official data; assume near 0 for code accuracy 
        "meta": 0.0,         # no official; community uses ~0 for coding 
        "mistral": 0.0       # no official; use near 0 for correct code (Nemo uses 0.3)  
    },
    "MATH": {
        "openai": 0.0,       # 0-0.2 for focused tasks like math""
        "google": 0.0,       # treat like classification  
        "anthropic": 0.0,    # analytical task  
        "deepseek": 0.0,     # math grouped with coding at 0.0 
        "xai": 0.0,          # no official; likely 0 for precise calculation 
        "meta": 0.0,         # no official; low temp for deterministic output 
        "mistral": 0.0       # no official; low temp to avoid errors 
    },
    "TRANSLATION": {
        "openai": 0.0,       # no explicit value; ~0-0.2 for faithful translation 
        "google": 0.0,       # no explicit; use low (0-0.3) for deterministic tasks 
        "anthropic": 0.0,    # no explicit; treat as analytical -> low temp 
        "deepseek": 1.3,     # recommended higher temperature for translation 
        "xai": 0.0,          # no official; presumably low to stick to source text 
        "meta": 0.0,         # no official; low temp to avoid mistranslation (default 0.9) 
        "mistral": 0.0       # no official; low temp for literal accuracy 
    },
    "SUMMARIZATION": {
        "openai": 0.2,       # generally keep low for factual summary (0-0.3) 
        "google": 0.2,       # start low (~0.2), increase if summary too generic 
        "anthropic": 0.2,    # no explicit; low for accurate summaries (Claude default 1.0) 
        "deepseek": 1.0,     # not given; ~1.0 used for data analysis tasks (proxy) 
        "xai": 0.0,          # no official; assume low for coherence (decisive mode) 
        "meta": 0.0,         # no official; likely low to avoid hallucination 
        "mistral": 0.3       # Mistral Nemo-Instruct recommends 0.3 for guided tasks 
    },
    "REAL_TIME_DATA_QUERY": {
        "openai": 0.0,       # use 0 to avoid creativity when using external data 
        "google": 0.0,       # no direct quote; best to use 0 for RAG to ensure accuracy 
        "anthropic": 0.0,    # no explicit; assumed 0 to stick strictly to retrieved info 
        "deepseek": None,    # not specified by DeepSeek (likely would be low)
        "xai": None,         # not specified by xAI (likely low for accuracy)
        "meta": None,        # not specified by Meta (community uses 0 for RAG)
        "mistral": None      # not specified by Mistral (usually set ~0 for RAG)
    },
    "CLASSIFICATION": {
        "openai": 0.0,       # 0 for deterministic classification 
        "google": 0.0,       # Google recommends temp=0 for classification tasks 
        "anthropic": 0.0,    # analytical/multiple-choice -> temp ~0 
        "deepseek": 0,       # no specific guidance (would default to 0)
        "xai": 0,            # no specific guidance (likely 0)
        "meta": 0,           # no specific guidance (likely 0)
        "mistral": 0.1       # no specific guidance (likely low)
    },
    "OTHER": {
        "openai": 1.0,       # default API temperature is 1.0 
        "google": 0.7,       # default ~1.0 for models (but best practice start ~0.2) 
        "anthropic": 1.0,    # default is 1.0 (range 0-1) 
        "deepseek": 1.0,     # default 1.0, but general conversation mode uses 1.3 
        "xai": 0.7,          # default not explicitly mentioned (range 0-1) 
        "meta": 0.9,         # default temperature for LLaMA models is 0.9 
        "mistral": 0.3       # default 0.7; Nemo-Instruct model recommends 0.3 
    }
}




SELF_REVIEW = False    # if set to True, agents will review their own work :) 


#===============================================================================================================================
#---------------------------------------------------- AGENTS' TOOLS ------------------------------------------------------------
#===============================================================================================================================
import crewai_tools
from crewai_tools import YoutubeVideoSearchTool, SerperDevTool, WebsiteSearchTool, CodeInterpreterTool  # need to check they match the AVAILABLE_TOOLS table



AVAILABLE_TOOLS = {
        "YoutubeVideoSearchTool": 
            { 
            "tool_description": YoutubeVideoSearchTool().description, # "A RAG tool aimed at searching within YouTube videos, ideal for video data extraction.",
            "tool_factory": YoutubeVideoSearchTool
            }, 
        "SerperDevTool" :{ 
            "tool_description":SerperDevTool().description, # "Designed to search the internet and return the most relevant results.",
           "tool_factory": SerperDevTool
            },
#        "WebsiteSearchTool" : { 
#            "tool_description": WebsiteSearchTool().description, # "Performs a RAG (Retrieval-Augmented Generation) search within the content of a website.",
#           "tool_factory": WebsiteSearchTool
#            },
        "CodeInterpreterTool" : {
            "tool_description": CodeInterpreterTool().description,
            "tool_factory": CodeInterpreterTool
        }
    }




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
  recommended_tools: Optional[List[str]] = None    
  edited_query: Optional[str] = None             # this is for future when we will change both query and temperature based on the query_type 

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
        f"Classify the query '{query}' into ONE and ONLY ONE of these types: [{', '.join(QUERY_TYPES.keys())}]. "
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


def get_second_pass_prompt(query, criteria, winner_response, improvement_points, user_feedback = None):
    return ( 
        f"Based on the initial query '{query}' and the criteria {', '.join(criteria)}, improve the following original response:\n"
            f"{winner_response}\n"
            f"Take into account the following improvement points into account: {', '.join(improvement_points)},\n"
            f", and also user's feedback on the original response and improvement points: {user_feedback}\n"
            f"Compare your new response to the original and make sure that your new response is better than the original.\n"
            f"Rework your new response if required, until it is better than the original.\n"
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
 
def create_agent_from_name(model_name, base_name="", query_type = None, tools=None):
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
    
        # Set up LLM parameters
        llm_params = {
            "model": model_name,
            "api_key": config["api_key"],
            "temperature" : QUERY_TYPES_TEMPERATURE[query_type][provider] if query_type else config["temperature"]
        }

        llm = config["llm_class"](**llm_params)
        
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
    for attempt in range(3):
        try:
            results = pseudo_crew.kickoff()  # Synchronous
            break
        except Exception as e:
            logging.error(f"{my_name()} - pseudo_crew for {agent_name} failed: {e}. Attempt {attempt + 1}")
            time.sleep(2)   # wait before retrying 
            if attempt == 2: # last attempt 
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
            expected_output=expected_output,
            output_format = 'json'
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
            inform_user(f"Task {i} for {agent_name} failed to execute", severity="ERROR")
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
    
    # Simple repair: if it starts with '{' but doesn't end with '}', append '}'. This is an ugly manual hack :( 
    if text.startswith('{') and not text.endswith('}'): text += '}'
    
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
            logging.error(f"\n{my_name()}: Leader LLM failed to analyze the query.")
            inform_user("Leader LLM failed to analyze the query. Falling back to a generic type 'OTHER'.", severity = "ERROR")
            self.state["analysis_state"]["query_type"] = query_type
            self.state["analysis_state"]["criteria"] = QUERY_TYPES[query_type]
            self.tools = []
            self.tools_names = []
            # raise RuntimeError(f"\n{my_name()}: Leader LLM failed to analyze the query. Execution will be stopped")      
        else:
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
            agent, agent_name = create_agent_from_name(model_name, AGENT_BASE_NAME, query_type= query_type, tools = self.tools)
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

        user_feedback = input("\nPlease provide your comments on the text and improvement points and / or press Enter to continue:")


        # create a new task and a new pseudo-crew for the winner:
        task_prompt = get_second_pass_prompt(
            query=self.state["query"],
            criteria=self.state["analysis_state"]["criteria"],
            winner_response=winner_response, 
            improvement_points=winner_improvement_points, 
            user_feedback=user_feedback
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
    # query = "summarize the content of the website: https://www.epam.com as of today (March 2025) and present the EPAM latest stock price"
    query = "please find the latest information on Leo Lozner. Hint - he is related to EPAM"
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
  


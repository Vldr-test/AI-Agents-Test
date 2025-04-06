# peer_review_config.py 
# global configuration for the project 

#===============================================================================================================================
#-------------------------------------------------- GLOBAL CONFIGURATION -------------------------------------------------------
#===============================================================================================================================

USE_REAL_THREADS = True             # controls threading model; if False, uses Python asyncio threads 

import logging
import inspect  # Added for my_name lambda
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, RootModel, ValidationError  
import asyncio
from typing import Type, List, Dict, Optional, Tuple, Union, Any


AVAILABLE_LEADER_LLMS = [
    "google_genai:gemini-2.0-flash", 
    "gpt-4o-mini"
]

# Available agent LLMs for responses and peer reviews
AVAILABLE_AGENT_LLMS = [
    "gpt-4",
    "claude-3-5-sonnet-latest",     
    "google_genai:gemini-2.0-flash",              
    "deepseek-chat"
]

QUERY_TYPES = {
    "CREATIVE_WRITING": ["creativity", "originality", "coherence"],
    "SOFTWARE_PROGRAMMING": ["accuracy", "efficiency", "readability"],
    "MATH": ["precision", "clarity", "logic"],
    "TRANSLATION": ["accuracy", "fluency", "clarity", "grammar"],
    "SUMMARIZATION": ["coverage_of_key_points", "conciseness", "clarity"],
    "REAL_TIME_DATA_QUERY": ["accuracy", "factual_correctness"],
    "OTHER": ["quality"]
}
 
from langchain_community.utilities import WikipediaAPIWrapper, SerpAPIWrapper, OpenWeatherMapAPIWrapper 
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, OpenWeatherMapQueryRun
from langchain_experimental.tools import PythonREPLTool #, YoutubeVideoSearchTool
from langchain_community.agent_toolkits import FileManagementToolkit  
from langchain_core.tools import Tool, BaseTool

import os  

#=======================================================================================
#------------------------------ TOOLS CONFIGURATION ------------------------------------
#=======================================================================================

#def serp_api_factory(api_key: str) -> Tool:
#    """A wrapper for the SerpAPI search engine. Required since SerpAPIWrapper does not have 
#       'description' field, etc.
#  """
#    tool = SerpAPIWrapper(serpapi_api_key = api_key)
#    return Tool(name="SerperAPI search tool", 
#                func=tool.run, 
#                description="Search the internet for current information"
#        )

class HITL_tool(BaseTool):
    name: str = "HITL_tool" # Give it a clear name
    description: str = ( # Crucial for the LLM to know when to use it
        "Use this tool when you need clarification, confirmation, or additional input "
        "directly from the human user. Ask a clear question for the human based on your current task. "
    )

    # Synchronous version
    def _run(self, query: str) -> str:
        """Synchronously ask the human for input."""
        print(f"\n Agent asks: {query}") # Display the question from the LLM
        user_response = input("Your Input: ")    # Get input from console
        return user_response


ALL_TOOLS = {
   
    #"YoutubeVideoSearchTool":  { 
    # A RAG tool aimed at searching within YouTube videos, ideal for video data extraction.
    #   "API_KEY": None, 
    #   "tool_factory" : YoutubeVideoSearchTool
    #}, 

    "HITL_tool": {
    # A tool for human-in-the-loop interaction, allowing the agent to ask the user for input.
        "API_KEY": None, 
        "tool_factory": lambda: HITL_tool()
    }, 
    

    "WikipediaQueryRun" : {
    # A tool for querying Wikipedia pages, useful for retrieving structured information.
        "API_KEY": None,
        "tool_factory": lambda: WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))
        # Import: from langchain_community.tools import WikipediaQueryRun
        #         from langchain_community.utilities import WikipediaAPIWrapper
        },

    # There is a complicated error this tool is causing; for the time being, Tavily is used instead.
    #"SerperAPI": { 
    # Designed to search the internet and return the most relevant results.
    #    "API_KEY": "SERPER_API_KEY", 
    #    "tool_factory": lambda api_key: serp_api_factory(api_key=api_key)
    #    },
    # Import: from langchain_community.utilities import SerpAPIWrapper
    #        from langchain.tools import Tool

    
    "TavilySearchResults" : {
    # A tool for searching and retrieving results from Tavily, a specialized search engine.
        "API_KEY": "TAVILY_API_KEY", 
        "tool_factory": lambda api_key: TavilySearchResults(tavily_api_key=api_key, max_results=5)
        },
    # Import: from langchain_community.tools.tavily_search import TavilySearchResults

    "DuckDuckGoSearchRun" : {
        "API_KEY": None, 
        "tool_factory": lambda: DuckDuckGoSearchRun()
        }, 
    # Import: from langchain_community.tools import DuckDuckGoSearchRun

    "PythonREPLTool" : {
        # A tool for executing Python code in a REPL (Read-Eval-Print Loop) environment.
        "API_KEY": None, 
        "tool_factory": lambda: PythonREPLTool()
    }, 
    # Import: from langchain_community.tools import PythonREPLTool

    "OpenWeatherMap": {
        "API_KEY": "OPENWEATHERMAP_API_KEY",  # Requires an API key from openweathermap.org
        "tool_factory": lambda api_key: OpenWeatherMapQueryRun(
            api_wrapper=OpenWeatherMapAPIWrapper(openweathermap_api_key=api_key)
        )
        # Import: from langchain_community.tools import OpenWeatherMapQueryRun
        #         from langchain_community.utilities import OpenWeatherMapAPIWrapper
    }

    # These is not one tool, but a set of tools for file management. It is not instantiated propertly yet.
    #"FileManagement": {
    #    "API_KEY": None,
    #    "tool_factory": lambda: FileManagementToolkit(
    #        root_dir=os.getenv("AGENT_FILE_ROOT_DIR", "/tmp/agent_files") # Read from env
    #    ).get_tools() # Decide below if you want all tools
    #}
    # Import: from langchain_community.agent_toolkits import FileManagementToolkit
}


QUERY_TYPES_TEMPERATURE = {
    "CREATIVE_WRITING": {
        "openai": 0.9,       # high creativity recommended 
        "gemini": 0.9,       # higher end (0.7-1.0) for creative tasks 
        "anthropic": 1.0,    # closer to 1.0 for generative tasks 
        "deepseek": 1.5,     # official recommendation for creative writing
        "xai": 1.0,          # no official value; high (range 0-1) for creativity
        "meta": 1.0,         # up to 1.0 (default 0.9) for more diverse output 
        "mistral": 0.8       # higher end (default 0.7) for creative output 
    },
    "SOFTWARE_PROGRAMMING": {
        "openai": 0.0,       # use 0 for deterministic code generation
        "gemini": 0.0,       # low (0-0.3) for deterministic outputs like code 
        "anthropic": 0.0,    # closer to 0 for analytical tasks  
        "deepseek": 0.0,     # coding/math recommended at 0.0 
        "xai": 0.0,          # no official data; assume near 0 for code accuracy 
        "meta": 0.0,         # no official; community uses ~0 for coding 
        "mistral": 0.0       # no official; use near 0 for correct code (Nemo uses 0.3)  
    },
    "MATH": {
        "openai": 0.0,       # 0-0.2 for focused tasks like math""
        "gemini": 0.0,       # treat like classification  
        "anthropic": 0.0,    # analytical task  
        "deepseek": 0.0,     # math grouped with coding at 0.0 
        "xai": 0.0,          # no official; likely 0 for precise calculation 
        "meta": 0.0,         # no official; low temp for deterministic output 
        "mistral": 0.0       # no official; low temp to avoid errors 
    },
    "TRANSLATION": {
        "openai": 0.0,       # no explicit value; ~0-0.2 for faithful translation 
        "gemini": 0.0,       # no explicit; use low (0-0.3) for deterministic tasks 
        "anthropic": 0.0,    # no explicit; treat as analytical -> low temp 
        "deepseek": 1.3,     # recommended higher temperature for translation 
        "xai": 0.0,          # no official; presumably low to stick to source text 
        "meta": 0.0,         # no official; low temp to avoid mistranslation (default 0.9) 
        "mistral": 0.0       # no official; low temp for literal accuracy 
    },
    "SUMMARIZATION": {
        "openai": 0.2,       # generally keep low for factual summary (0-0.3) 
        "gemini": 0.2,       # start low (~0.2), increase if summary too generic 
        "anthropic": 0.2,    # no explicit; low for accurate summaries (Claude default 1.0) 
        "deepseek": 1.0,     # not given; ~1.0 used for data analysis tasks (proxy) 
        "xai": 0.0,          # no official; assume low for coherence (decisive mode) 
        "meta": 0.0,         # no official; likely low to avoid hallucination 
        "mistral": 0.3       # Mistral Nemo-Instruct recommends 0.3 for guided tasks 
    },
    "REAL_TIME_DATA_QUERY": {
        "openai": 0.0,       # use 0 to avoid creativity when using external data 
        "gemini": 0.0,       # no direct quote; best to use 0 for RAG to ensure accuracy 
        "anthropic": 0.0,    # no explicit; assumed 0 to stick strictly to retrieved info 
        "deepseek": None,    # not specified by DeepSeek (likely would be low)
        "xai": None,         # not specified by xAI (likely low for accuracy)
        "meta": None,        # not specified by Meta (community uses 0 for RAG)
        "mistral": None      # not specified by Mistral (usually set ~0 for RAG)
    },
    "CLASSIFICATION": {
        "openai": 0.0,       # 0 for deterministic classification 
        "gemini": 0.0,       # Google recommends temp=0 for classification tasks 
        "anthropic": 0.0,    # analytical/multiple-choice -> temp ~0 
        "deepseek": 0,       # no specific guidance (would default to 0)
        "xai": 0,            # no specific guidance (likely 0)
        "meta": 0,           # no specific guidance (likely 0)
        "mistral": 0.1       # no specific guidance (likely low)
    },
    "OTHER": {
        "openai": 1.0,       # default API temperature is 1.0 
        "gemini": 0.7,       # default ~1.0 for models (but best practice start ~0.2) 
        "anthropic": 1.0,    # default is 1.0 (range 0-1) 
        "deepseek": 1.0,     # default 1.0, but general conversation mode uses 1.3 
        "xai": 0.7,          # default not explicitly mentioned (range 0-1) 
        "meta": 0.9,         # default temperature for LLaMA models is 0.9 
        "mistral": 0.3       # default 0.7; Nemo-Instruct model recommends 0.3 
    }
}

# Configure logging
logging.basicConfig(level=logging.ERROR, handlers=[logging.StreamHandler()], force=True)

# Function to get current function name for logging
my_name = lambda: inspect.currentframe().f_back.f_code.co_name


#===============================================================================================================================
#-------------------------------------------------- PYDANTIC DATA MODELS -------------------------------------------------------
#===============================================================================================================================

# unstructured LLM response. Maps agent names to their responses directly 
class SimpleResponseFormat(RootModel[dict[str, str]]):
    pass

#-----------------------------------------------------------------------------------------------------------------------

# response from query_analysis
class QueryAnalysisResponseFormat(BaseModel):
    query_type: str                         # type of query; e.g. "SOFTWARE_PROGRAMMING", "CREATIVE_WRITING", etc.
    query_class: str                        # "COMPLEX" or "SIMPLE"
    prompt_for_agents: str                  # improved prompt that includes tools, etc. 
    recommended_tools: Optional[List[str]] = None     # names of tools to be used by agents 

  
#--------------------------------------------------------------------------------------------------------------------------

# internal class
class InnerPeerReviewFormat(BaseModel):
    improvement_points: List[str]
    score: int 


class PeerReviewResponseFormat(RootModel[dict[str, InnerPeerReviewFormat]]):
    pass
#------------------------------------------------------------------------------------------------------

# plain list of strings for improvement points: 
class ImprovementPointsFormat(RootModel):
    root: List[str]
# peer_review_config.py 
# global configuration for the project 

#===============================================================================================================================
#-------------------------------------------------- GLOBAL CONFIGURATION -------------------------------------------------------
#===============================================================================================================================

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


AVAILABLE_TOOLS = {
        "YoutubeVideoSearchTool":  { 
            "tool_description":"A RAG tool aimed at searching within YouTube videos, ideal for video data extraction.",
            "tool_factory": None # YoutubeVideoSearchTool
            }, 
        "SerperDevTool": { 
            "tool_description":"Designed to search the internet and return the most relevant results.",
           "tool_factory": None  # SerperDevTool
            },
#        "WebsiteSearchTool" : { 
#            "tool_description": "Performs a RAG (Retrieval-Augmented Generation) search within the content of a website.",
#           "tool_factory": None # WebsiteSearchTool
#            },
        "CodeInterpreterTool" : {
            "tool_description": "A tool for interpreting and running Python code, ideal for programming tasks.",
            "tool_factory": None # CodeInterpreterTool
        }
    }


# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()], force=True)

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
class QueryAnalysisFormat(BaseModel):
  query_type: str
  # criteria: List[str]
  recommended_tools: Optional[List[str]] = None    
  improved_query: Optional[str] = None             # this is for future when we will change both query and temperature based on the query_type 

#--------------------------------------------------------------------------------------------------------------------------

# internal class
class InnerPeerReviewFormat(BaseModel):
  improvement_points: List[str]
  score: int 


class PeerReviewResponseFormat(RootModel[dict[str, InnerPeerReviewFormat]]):
    pass

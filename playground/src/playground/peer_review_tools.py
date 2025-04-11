# peer_review_tools.py

from langchain.tools import Tool, BaseTool
import logging
import os
from dotenv import load_dotenv
from langchain_community.utilities import WikipediaAPIWrapper, SerpAPIWrapper, OpenWeatherMapAPIWrapper 
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, OpenWeatherMapQueryRun
from langchain_experimental.tools import PythonREPLTool #, YoutubeVideoSearchTool
from langchain_community.agent_toolkits import FileManagementToolkit  
from langchain_core.tools import Tool, BaseTool
from typing import Type, List, Dict, Optional, Tuple, Union, Any
from peer_review_config import my_name



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



#-------------------------------------------------------------------------------
#------------------------------ LOAD_ALL_TOOLS() -------------------------------
#-------------------------------------------------------------------------------
def load_all_tools()-> Tuple[List[BaseTool], str]:
    
    """
        Go over the list of ALL_TOOLS, try to instantiate them. 
        Return the list of available tools 
    """
    
    available_tools = []
    
    load_dotenv()
    
    # go over the list of all tools and check if they are available:
    for tool_name in ALL_TOOLS.keys():
        try:
            api_key = None
            # check if the tool requires an API key: 
            api_key_name = ALL_TOOLS[tool_name]["API_KEY"]
            if api_key_name:
                # check if the API key is in the environment variables: 
                api_key = os.getenv(api_key_name)
                if not api_key: 
                    logging.error(f"{my_name()}: {tool_name}' API_KEY not found in .env file")
                    continue
            
            # if the tool doesn't require an API key or the API key is available, load it: 
            # logging.info(f"{my_name()}: Loading tool: {tool_name}, {ALL_TOOLS[tool_name]}")
            tool = ALL_TOOLS[tool_name]["tool_factory"](api_key=api_key) if api_key else ALL_TOOLS[tool_name]["tool_factory"]()
            
            # note that from here on we use the names returned by the tools, hence tool.name, not tool_name :) 
            available_tools.append(tool)
            logging.info(f"{my_name()}: Successfully instantiated the tool: {tool_name}")
        except Exception as e:
            logging.error(f"{my_name()}: Can't load tool: {tool.name}, {e}") # shall we better instantiate one by one? 
            continue
    
    tools_descriptions = "\n\n".join([
        f"***Tool_name***: {tool.name}   ***Tool_description***: {tool.description}***" for tool in available_tools if hasattr(tool, 'name') and hasattr(tool, 'description')  
    ])   

    return available_tools, tools_descriptions
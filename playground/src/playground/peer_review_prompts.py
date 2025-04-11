# peer_review_prompts

from peer_review_config import QUERY_TYPES, my_name
import json
import logging
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chains import LLMChain
from pydantic import BaseModel, ValidationError  
from typing import Type, List, Dict, Optional, Tuple, Union, Any





#===========================================================================
#---------------------- PROMPTS GENERATION FOR ALL TASKS -------------------
#===========================================================================


#---------------------------------------------------------------------------
#----------------------- GET_QUERY_ANALYSIS_PROMPT_1() ---------------------
#---------------------------------------------------------------------------
def get_query_analysis_prompt_1(tools_descriptions_str: str, query_types_str: str) -> ChatPromptTemplate:
    """
    Generates the ChatPromptTemplate for the query analysis phase.

    This version embeds static context (tool descriptions, query types) directly
    into the prompt string, expecting only 'input' and 'agent_scratchpad'
    variables at runtime.

    Args:
        tools_descriptions_str: A formatted string listing available tools and their descriptions.
        query_types_str: A formatted string listing the possible query types.

    Returns:
        A ChatPromptTemplate_1 object to be used by Executor (!)
    """
    # logging.info(f"{my_name()}:  tools: {tools_descriptions_str} and \n query types: {query_types_str}")
    # Define the system message content
    system_message_content = (
        "You are an expert in analyzing user queries and running the tools for finding relevant context. \n"
        "You work with the user using HITL_tool, to clarify all additional information. \n"
        "You work with other search tools to gather all relevant information.\n"
    )

    # Define the human message template string using an f-string.
    # Static context (tools_descriptions_str, query_types_str) is embedded directly.
    # '{input}' remains as the placeholder for the dynamic user query.
    human_message_template_string = f"""Your task is to collect all relevant information from the user and external tools, 
    , based on the user’s query ***{{input}}***.
    For that, you MUST execute the following steps:
    1. Classify the query into one and only one query_type from this list: ***{query_types_str}***.
    2. Define the query_class as 'SIMPLE' or 'COMPLEX', based on the anticipated response complexity.
    3. CALL the internal 'HITL_tool' to get clarifications from the user. 
    4. CALL one or several tools from this list: ***{tools_descriptions_str}*** to get additional context. 

    RETURN FORMAT:
    You MUST return a JSON object with exactly THREE fields, no extra text:
    * 'query_type' (a string): the query type from the list: ***{query_types_str}***.
    * 'query_class': 'COMPLEX' if you want to use the team of agents, or 'SIMPLE' if you want to do it yourself.
    * 'improved_query' (a string): an improved version of the original query, 
    based on the user information from HITL_tool, as well as other tools' output.

    Always keep the language of the query (e.g a query in Spanish could be improved only in Spanish, if at all)
    Do NOT include any extra text, markers, or schema in your reply, — just the JSON.
"""

    # Create the ChatPromptTemplate object
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message_content),
        ("human", human_message_template_string),
        MessagesPlaceholder(variable_name="agent_scratchpad") # Essential for agent memory/history
    ])
    
    # No formatting happens here; return the template object itself
    return prompt_template

#---------------------------------------------------------------------------
#----------------------- GET_QUERY_ANALYSIS_PROMPT_2() ---------------------
#---------------------------------------------------------------------------
def get_query_analysis_prompt_2(improved_query:str, tools_descriptions_str: str) -> str:
    """
    Generates the ChatPromptTemplate for agents' prompt generation .

    This version embeds static context (tool descriptions, query types) directly
    into the prompt string, expecting only 'input' and 'agent_scratchpad'
    variables at runtime.

    Args:
        tools_descriptions_str: A formatted string listing available tools and their descriptions.
        
    Returns:
        A ChatPromptTemplate object ready to be used by the agent.
    """
    # Define the human message template string using an f-string.
    # Static context (tools_descriptions_str, query_types_str) is embedded directly.
    # '{input}' remains as the placeholder for the dynamic user query.
    prompt = f"""Your task is to create a prompt for the AI agent, 
    based on the query ***{improved_query}***.
    For that, you MUST execute the following steps: \n
    1. Analyze the query and convert it into the best prompt for AI agents\n. 
    2. Select one or several tools from this list: ***{tools_descriptions_str}*** for the AI agent to get additional context. \n

    RETURN FORMAT:
    You MUST return a JSON object with exactly TWO fields, no extra text:\n
    * 'prompt_for_agents' (a string): the prompt for the AI agents, based on the query.\n
    You can keep the original query if you believe it is a good prompt for the agents. \n 
    * 'recommended_tools' (a list of strings): a list of recommended tools for the agents to execute.\n

    Always keep the language of the query (e.g a query in Spanish could be improved only in Spanish, if at all)
    Do NOT include any extra text, markers, or schema in your reply, — just the JSON.
    """
    # No formatting happens here; return the string
    return prompt


#---------------------------------------------------------------------------
#---------------------- GET_RESPONSE_GENERATION_PROMPT() -------------------
#---------------------------------------------------------------------------
def get_response_generation_prompt(query):
    # Define the template with explicit message objects
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a response generator. Your task is to create a response for the user’s query. "
            "Match the query’s language (e.g., Russian query = Russian response) unless told otherwise. "
            "Return a plain string—no Unicode escapes (use 'В' not '\\u0412'). "
            "Use tools if available. Do NOT include markers or extra text—just the response."
        ),
        ("human", "Respond to this query: ***{query}***")
    ])
    
    # Format with inputs
    formatted_prompt = template.format_messages(
        query=query,
    )
    # logging.info(f"{my_name}: {formatted_prompt}")
    return formatted_prompt

#---------------------------------------------------------------------------
#-------------------------- GET_PEER_REVIEW_PROMPT() -----------------------
#---------------------------------------------------------------------------
def get_peer_review_prompt(query, responses):
    # Define the template with explicit message objects
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a peer reviewer. Your job is to evaluate responses to a query. "
        ),
        ("human", "Review these responses to the query ***{query}***:\n"
            "{responses}. For each response, provide: "
            "- 'improvement_points': a list of 2-5 actionable suggestions. "
            "- 'score': an integer from 1 (lowest) to 100 (highest). Be fair but harsh. "
            "Return a JSON object where keys are agent names and values are objects with 'improvement_points' and 'score'. "
            "only use currency symbols inside quoted strings."
            "Do NOT include any extra text — just the JSON."
        )
    ])
    
    # Format with inputs
    formatted_prompt = template.format_messages(
        query=query,
        responses=responses
    )
    # logging.info(f"{my_name}: {formatted_prompt}")

    return formatted_prompt

#---------------------------------------------------------------------------
#---------------------------- GET_ITERATION_PROMPT() -----------------------
#---------------------------------------------------------------------------
def get_iteration_prompt(query, response, improvement_points):
    # Define the template with explicit message objects
    template = ChatPromptTemplate.from_messages([
        ("system", 
            "You are an expert editor skilled in enhancing text incrementally while preserving its core. "
            "Your task is to refine the given response based on specific improvement points."
        ),
        ("human",
           "Refine this response ***{response}*** to the query ***{query}*** :\n"
            "Address these  improvement points:\n"
            "{improvement_points}\n\n"
            "You MUST:\n"
            "- Reuse the original response as the base, modifying it to incorporate each improvement point.\n"
            "- Ensure the new response is noticeably different and improved, not a copy.\n"
            "- Verify before returning that all improvement points are addressed and the result is better.\n"
            "- Keep what’s already strong (like clarity or key points) and only tweak what requires improvement."
            "Return only the improved response as a plain string—no Unicode escapes (use 'В' not '\\u0412'), no extra text."
        )
    ])
    
    # Format improvement points as a bulleted list for clarity
    formatted_improvements = '\n'.join(f"- {point}" for point in improvement_points)
    
    formatted_prompt = template.format_messages(
        query=query,
        response=response,
        improvement_points=formatted_improvements
    )
    # logging.info(f"{my_name}: {formatted_prompt}")
    return formatted_prompt

#---------------------------------------------------------------------------
#------------------ GET_REVIEW_IMPROVEMENT_POINTS_PROMPT() -----------------
#---------------------------------------------------------------------------
def get_review_improvement_points_prompt(query: str, improvement_points: list[str])-> list[str]:
    # shorten and rationalize a list of improvement points 
    template = ChatPromptTemplate.from_messages([
        ("system", 
            "You are an experienced editor. Your job is to improve other people's responses."
            "You are very good at incremental improvements, respecting the original version, but making it better. "
            "You know how to adress improvement points and user feedback on the original response. "
        ),
        ("human",
            "Analize and rationalize this list of improvement points: ***{improvement_points}***. "
            "To give you some context, these list is based on the response to the following query: ***{query}***. "
            "Combine similar improvement points into one. "
            "Make sure that all improvement points are clear, concise, and actionable. "
            "Make the final list as short as possible - ideally, not more than three improvement points. " 
            "Return a plain JSON list of refined improvement points. "
            "Do NOT include any extra text; return ONLY the JSON list! "
            "Return a plain string — no Unicode escapes (use 'В' not '\\u0412'). "
        )
    ])

    # Format with inputs
    formatted_prompt = template.format_messages(
        query=query,
        improvement_points=", ".join(improvement_points)
    )
    # logging.info(f"{my_name}: {formatted_prompt}")

    return formatted_prompt

#---------------------------------------------------------------------------
#---------------------- GET_FEEDBACK_REQUEST_PROMPT() -------------------
#---------------------------------------------------------------------------
def get_feedback_request_prompt(winner_response: str, winner_score: int, improvement_points: list[str] )-> ChatPromptTemplate:
    
    # Define the system message content
    system_message_content = (
        "You are an expert in interactions with human using the HITL_tool to clarify user's feedback. \n"
    )

    # Define the human message template string using an f-string.
    human_message_template_string = f"""Your task is to check with the user if the user has any 
     feedback on the final response . For that, you should:
      - present the user the final winner response ***{winner_response}***, 
      winner score ***{winner_score}*** and improvement points ***{improvement_points}***. 
      - then, check for user's feedback, possibly engaging into a dialog. 
      Let the user know that if he or she types: "break", "stop", "enough" or similar, 
      you will stop the interaction. 
    
    RETURN FORMAT:
    You MUST return a result of your dialog with the user, based on the HITL_tool input or a word "break" 
    if the user decides to stop the work. 

    Do NOT include any extra text, markers, or schema in your reply, — just the user input.
"""

    # Create the ChatPromptTemplate object
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message_content),
        ("human", human_message_template_string),
        MessagesPlaceholder(variable_name="agent_scratchpad") # Essential for agent memory/history
    ])
    
    # No formatting happens here; return the template object itself
    return prompt_template
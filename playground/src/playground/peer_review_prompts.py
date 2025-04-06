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



#===============================================================================================================================
#-------------------------------------------------- PYDANTIC DATA MODELS -------------------------------------------------------
#===============================================================================================================================

class AgentResponseFormat(BaseModel):
    agent_name: str  
    response: str

    
class InnerPeerReviewFormat(BaseModel):
    improvement_points: List[str]
    score: int 

class PeerReviewFormat(BaseModel):
    agent_name: str
    reviews: Dict[str, InnerPeerReviewFormat]

#===============================================================================================================================
#-------------------------------------------- PROMPTS GENERATION FOR ALL TASKS--------------------------------------------------
#===============================================================================================================================


#===========================================================================
#------------------------- GET_QUERY_ANALYSIS_PROMPT -----------------------
#===========================================================================
def get_query_analysis_prompt(tools_descriptions_str: str, query_types_str: str) -> ChatPromptTemplate:
    """
    Generates the ChatPromptTemplate for the query analysis phase.

    This version embeds static context (tool descriptions, query types) directly
    into the prompt string, expecting only 'input' and 'agent_scratchpad'
    variables at runtime.

    Args:
        tools_descriptions_str: A formatted string listing available tools and their descriptions.
        query_types_str: A formatted string listing the possible query types.

    Returns:
        A ChatPromptTemplate object ready to be used by the agent.
    """
    logging.info(f"{my_name()}:  tools: {tools_descriptions_str} and \n query types: {query_types_str}")
    # Define the system message content
    system_message_content = (
        "You are an expert query classifier and prompt engineer. \n"
        "Your job is to analyze the user query and gather all missing information.\n"
    )

    # Define the human message template string using an f-string.
    # Static context (tools_descriptions_str, query_types_str) is embedded directly.
    # '{input}' remains as the placeholder for the dynamic user query.
    human_message_template_string = f"""Your task is to collect all relevant information from the user and external tools, 
    , based on the user’s query ***{{input}}***.
  For that, you MUST execute the following steps:
  1. Classify the query into one and only one query_type from the list: ***{query_types_str}***.
  2. Define the query_class as 'SIMPLE' or 'COMPLEX', based on the anticipated response complexity.
  3. CALL the internal 'HITL_tool' to get clarifications from the user. NEVER recommend the HITL_tool to your agents - do it yourself to create a better prompt! 
  4. CALL one or a few of the tools {tools_descriptions_str} 
  5. Create a single prompt_for_agents, using:
     - the original user's query.
     - Additional user input from HITL_tool call (if available).
     - (optional) results from other internal tools that you called. 
     - (optional) the list of tools from ***{tools_descriptions_str}*** that  agents should execute. 
     Use the EXACT tool names from this list of tools, without changing them for readibility.
     E.g. use 'WikipediaQueryRun", do NOT use a short form like 'Wikipedia'. 
  Use best prompt engineering practices to create a prompt that will help your agents to do their job.

RETURN FORMAT:
You MUST return a JSON object with exactly FOUR fields, no extra text:
* 'query_type' (a string)
* 'query_class': 'COMPLEX' if you want to use the team of agents, or 'SIMPLE' if you want to do it yourself.
* 'prompt_for_agents' (a string). Could be a copy of the original query, or an improved version of it, based on the user input and the tools output.
* 'recommended_tools' (a list of strings) from {tools_descriptions_str} that you want the agents to use. 
   Use the EXACT tool names from this list of tools, without changing them for readibility. 
   E.g. use 'WikipediaQueryRun", do NOT use a short form like 'Wikipedia'. 
Always keep the language of the query (e.g a query in Spanish could be improved only in Spanish, if at all)
Do NOT include any extra text, markers, or schema in your reply, — just the JSON.
"""

    # Create the ChatPromptTemplate object
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message_content),
        ("human", human_message_template_string),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # Essential for agent memory/history
    ])
    
    # No formatting happens here; return the template object itself
    return prompt_template



def get_response_generation_prompt(query, query_type, criteria):
    # Define the template with explicit message objects
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a response generator. Your task is to create a response for the user’s query. "
            "Focus on these criteria: {criteria}. "
            "Match the query’s language (e.g., Russian query = Russian response) unless told otherwise. "
            "Return a plain string—no Unicode escapes (use 'В' not '\\u0412'). "
            "Use tools if available. Do NOT include markers or extra text—just the response."
        ),
        ("human", "Respond to this query: ***{query}***")
    ])
    
    # Format with inputs
    formatted_prompt = template.format_messages(
        query=query,
        query_type=query_type,
        criteria=", ".join(criteria)
    )
    logging.info(f"{my_name}: {formatted_prompt}")
    return formatted_prompt

def get_peer_review_prompt(query, responses, criteria):
    # Define the template with explicit message objects
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a peer reviewer. Your job is to evaluate responses to a query based on these criteria: {criteria}. "
        ),
        ("human", "Review these responses to the query ***{query}***:\n"
            "{responses}. For each response, provide: "
            "- 'improvement_points': a list of 2-5 actionable suggestions. "
            "- 'score': an integer from 1 (lowest) to 100 (highest). Be fair but harsh. "
            "Return a JSON object where keys are agent names and values are objects with 'improvement_points' and 'score'. "
            "Do NOT include extra text—just the JSON."
        )
    ])
    
    # Format with inputs
    formatted_prompt = template.format_messages(
        query=query,
        responses=json.dumps(responses, ensure_ascii=False, indent=2),
        criteria=", ".join(criteria)
    )
    logging.info(f"{my_name}: {formatted_prompt}")

    return formatted_prompt

def get_iterations_prompt(query, criteria, response, improvement_points, user_feedback=None):
    # Define the template with explicit message objects
    template = ChatPromptTemplate.from_messages([
        ("system", 
            "You are an expert editor skilled in enhancing text incrementally while preserving its core. "
            "Your task is to refine the given response based on specific improvement points."
        ),
        ("human",
           "Refine this response to the query ***{query}*** based on criteria ***{criteria}***:\n"
            "Original Response: ***{response}***\n\n"
            "Address these specific improvement points:\n"
            "{improvement_points}\n\n"
            "If provided, prioritize this user feedback: ***{user_feedback}***. "
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
        criteria=", ".join(criteria),
        response=response,
        improvement_points=formatted_improvements,
        user_feedback=user_feedback if user_feedback else "None"
    )
    logging.info(f"{my_name}: {formatted_prompt}")
    return formatted_prompt


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
    logging.info(f"{my_name}: {formatted_prompt}")

    return formatted_prompt
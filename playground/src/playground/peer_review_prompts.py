# peer_review_prompts

from peer_review_config import QUERY_TYPES, AVAILABLE_TOOLS, my_name
import json
import logging
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
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

class QueryAnalysisFormat(BaseModel):
    query_type: str
    recommended_tools: Optional[List[str]] = None    
    edited_query: Optional[str] = None

class InnerPeerReviewFormat(BaseModel):
    improvement_points: List[str]
    score: int 

class PeerReviewFormat(BaseModel):
    agent_name: str
    reviews: Dict[str, InnerPeerReviewFormat]

#===============================================================================================================================
#-------------------------------------------- PROMPTS GENERATION FOR ALL TASKS--------------------------------------------------
#===============================================================================================================================

def get_query_analysis_prompt(query, available_tools):
    # Prepare data for the prompt
    tool_list = [tool_name for tool_name in available_tools.keys()]
    query_types = list(QUERY_TYPES.keys())

    # Define the template with explicit message objects
    template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert query classifier and prompt engineer."),
        ("human", "Your job is to analyze the user’s query ***{query}*** and classify it into ONE type from this list: ***{query_types}***. "
            "You must also recommend tools from this list: ***{tool_list}***, or return an empty list if none apply. "
            "Return a JSON object with three fields: "
            "* 'query_type' (a string) "
            "* 'recommended_tools' (a list of strings), and "
            "* an optional 'improved_query' (a string). "
            "If you believe that the initial query is already well worded, do NOT use 'improved_query'\n"
            "In any case always keep the language of the query " 
            " (e.g a query in Spanish could be improved only in Spanish, if at all) \n"
            "Do NOT include any extra text, markers, or schema in your reply, — just the JSON.")
    ])
    
    # Format the template with the query
    formatted_prompt = template.format_messages(query=query, query_types = ", ".join(query_types), tool_list = ", ".join(tool_list))
    logging.info(f"get_query_analysis_prompt: {formatted_prompt}")
    return formatted_prompt


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
            "You are an experienced editor. Your job is to improve responses to queries provided by others"
            "You receive the original query and the response that you HAVE to improve. "
            "For that please use explicit improvement points, optional user feedback on the original response,"
            "and a list of criteria."  
        ),
        ("human",
            "Improve this response: ***{response}*** on the query ***{query}***, based on criteria {criteria}***. "
            "Consider these improvement points: ***{improvement_points}***. "
            "If provided, give priority to this user feedback: {user_feedback}. "
            "Ensure the new response is better than the original! "
            "Return a plain string — no Unicode escapes (use 'В' not '\\u0412'). "
            "Do NOT include extra text — just the response."
        )
    ])
    
    # Format with inputs
    formatted_prompt = template.format_messages(
        query=query,
        criteria=", ".join(criteria),
        response= response,
        improvement_points=", ".join(improvement_points),
        user_feedback=user_feedback if user_feedback else "None"
    )
    logging.info(f"{my_name}: {formatted_prompt}")

    return formatted_prompt

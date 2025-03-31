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
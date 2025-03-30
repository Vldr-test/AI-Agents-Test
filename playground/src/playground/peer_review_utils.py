

import os
import json
import json5
import demjson3
import logging
from typing import Type, List, Dict, Optional, Tuple, Union, Any
from pydantic import BaseModel, ValidationError  
import re
import inspect
from peer_review_config import QueryAnalysisFormat, my_name




#===============================================================================================================================
#---------------------------------------------------- DICT_FROM_STR() ----------------------------------------------------------       
#===============================================================================================================================

def dict_from_str(llm_output: str, Pydantic_format: Optional[Type[BaseModel]] = None):
    """
    Tries to turn an LLM's output (a string) into a valid dict and validate it with a Pydantic objmodel if provided.
    Args:
        llm_output: The raw string from an LLM (could be JSON, could be messy).
        Pydantic_format: The Pydantic model we want to use for validation (optional).
    Returns: a valid dict or a Pydantic object, or None   
    """
 
    json_obj = json_to_dict(llm_output) # Try to fix the JSON string and get the parsed JSON object
    if json_obj is None:
        logging.error(f"{my_name()}: Failed to fix JSON string: {llm_output}")
        return None

    if Pydantic_format:
        try:
            pydantic_obj = Pydantic_format.model_validate(json_obj)  
            return pydantic_obj
        except ValidationError as e:
            logging.error(f"{my_name()}: Pydantic validation failed: {e}")
            return None
    else:
        return json_obj  # without validation 
    

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



import os
import json
import logging
from pathlib import Path
from typing import Any, List, Dict, Union

import json5
import demjson3

from typing import Type, Optional, Union
from pydantic import BaseModel, ValidationError
import re
import ast
from peer_review_config import my_name

# --- Configuration for Truncation (Adjust values as needed) ---
# Max characters to read from a single file
MAX_SINGLE_FILE_LENGTH = 200000
# Max total characters for the final combined string
MAX_TOTAL_LENGTH = 500000
# Max items to process from a list
MAX_LIST_ITEMS = 50
# Separator used when joining content from lists
LIST_ITEM_SEPARATOR = "\n\n---\n\n"


# ===========================================================================================
#                                    ARTIFACTS_TO_STR()  
# ===========================================================================================
def artifacts_to_str(artifacts: Any) -> str:
    """
    Converts various artifact types (str, list, dict, Path, List[Path])
    into a single string suitable for LLM prompts. Reads file content for Path objects.
    Omits metadata headers and uses large truncation limits.

    Args:
        artifacts: The input data (str, list, dict, Path, List[Path], etc.).

    Returns:
        A single string representation of the artifact(s).
    """
    if artifacts is None:
        return "" # Return empty string for None input

    final_string_parts = []
    current_total_length = 0
    overall_truncated = False

    # --- Handle List Input ---
    if isinstance(artifacts, list):
        items_processed = 0
        for item in artifacts:
            if items_processed >= MAX_LIST_ITEMS:
                logging.warning(f"{my_name()} Reached max list item limit ({MAX_LIST_ITEMS}). Truncating list.")
                overall_truncated = True
                break

            item_str = ""
            item_truncated = False
            available_length = MAX_TOTAL_LENGTH - current_total_length - len(LIST_ITEM_SEPARATOR)

            if available_length <= 0:
                 overall_truncated = True
                 break # Stop processing if no space left

            if isinstance(item, str):
                if len(item) > available_length:
                    item_str = item[:available_length]
                    item_truncated = True
                else:
                    item_str = item

            elif isinstance(item, Path):
                # Read file content, limit by available length AND single file limit
                read_limit = min(available_length, MAX_SINGLE_FILE_LENGTH)
                item_str, item_truncated = _read_file_content(item, read_limit)
                if item_str is None: item_str = "" # Handle read failure

            elif isinstance(item, dict):
                try:
                    json_str = json.dumps(item, ensure_ascii=False) # No indent for brevity
                    if len(json_str) > available_length:
                        item_str = json_str[:available_length]
                        item_truncated = True
                    else:
                        item_str = json_str
                except TypeError:
                    logging.warning(f"{my_name()}Could not serialize item of type {type(item)} to JSON. Using str().")
                    item_str_fallback = str(item)
                    if len(item_str_fallback) > available_length:
                         item_str = item_str_fallback[:available_length]
                         item_truncated = True
                    else:
                         item_str = item_str_fallback

            else: # Handle other types as strings
                item_str_fallback = str(item)
                if len(item_str_fallback) > available_length:
                    item_str = item_str_fallback[:available_length]
                    item_truncated = True
                else:
                    item_str = item_str_fallback

            final_string_parts.append(item_str)
            current_total_length += len(item_str)
            if item_truncated:
                overall_truncated = True
                logging.warning(f"{my_name()}Item {items_processed+1} truncated due to length limits.")
                # Stop processing if an item hit the available length exactly,
                # as adding separators or more items will exceed limit.
                if len(item_str) == available_length:
                     break
            # Add separator length calculation for next iteration (if not the first item)
            if items_processed > 0:
                 current_total_length += len(LIST_ITEM_SEPARATOR)

            items_processed += 1


        final_output = LIST_ITEM_SEPARATOR.join(final_string_parts)
        if overall_truncated:
             # Add truncation notice at the very end if needed
             trunc_msg = "\n... [Content Truncated]"
             # Check if adding the message exceeds limit, unlikely but possible
             if current_total_length + len(trunc_msg) <= MAX_TOTAL_LENGTH:
                  final_output += trunc_msg
             else: # If even the truncation message doesn't fit, just return what we have
                  pass

        return final_output


    # --- Handle Non-List Inputs ---
    elif isinstance(artifacts, str):
        if len(artifacts) > MAX_TOTAL_LENGTH:
            overall_truncated = True
            final_output = artifacts[:MAX_TOTAL_LENGTH]
        else:
            final_output = artifacts

    elif isinstance(artifacts, Path):
         # Read file, limit by MAX_TOTAL_LENGTH (as it's the only item) and single file limit
         read_limit = min(MAX_TOTAL_LENGTH, MAX_SINGLE_FILE_LENGTH)
         content, overall_truncated = _read_file_content(artifacts, read_limit)
         final_output = content if content is not None else ""

    elif isinstance(artifacts, dict):
         try:
             json_str = json.dumps(artifacts, indent=2, ensure_ascii=False) # Use indent for single dict
             if len(json_str) > MAX_TOTAL_LENGTH:
                 overall_truncated = True
                 final_output = json_str[:MAX_TOTAL_LENGTH]
             else:
                 final_output = json_str
         except TypeError:
             logging.warning(f"{my_name()}Could not serialize dict to JSON. Using str().")
             str_repr = str(artifacts)
             if len(str_repr) > MAX_TOTAL_LENGTH:
                  overall_truncated = True
                  final_output = str_repr[:MAX_TOTAL_LENGTH]
             else:
                  final_output = str_repr

    else: # Fallback for other types
         str_repr = str(artifacts)
         if len(str_repr) > MAX_TOTAL_LENGTH:
             overall_truncated = True
             final_output = str_repr[:MAX_TOTAL_LENGTH]
         else:
             final_output = str_repr

    if overall_truncated:
        final_output += "\n... [Content Truncated]"

    return final_output

# ----------------------------------------------------------------------
#                    _READ_FILE_CONTENT() helper 
# ----------------------------------------------------------------------
def _read_file_content(file_path: Path, max_len: int) -> Tuple[Optional[str], bool]:
    """
    Internal helper to read file content up to max_len.
    Returns (content_string, was_truncated). Handles errors.
    """
    try:
        if file_path.is_file():
            with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_len + 1) # Read extra char to detect truncation
            if len(content) > max_len:
                return content[:max_len], True # Content, Truncated = True
            else:
                return content, False # Content, Truncated = False
        else:
            logging.error(f"{my_name()} File not found or is not a file during read: {file_path}")
            return f"[File Not Found: {file_path}]", False # Return error message as content
    except Exception as e:
        logging.error(f"{my_name()} Error reading file {file_path}: {e}")
        return f"[Error Reading File: {file_path}]", False # Return error message as content
# --- Example Usage ---
# from pathlib import Path
# # Create dummy files for testing
# with open("test1.txt", "w") as f: f.write("Content of file 1.")
# with open("test2.py", "w") as f: f.write("print('Hello from file 2')")
# with open("long_file.txt", "w") as f: f.write("X" * (MAX_SINGLE_FILE_LENGTH + 100))

# simple_string = "This is a simple string."
# long_string = "This is a very long string..." * 100000 # Assume exceeds MAX_TOTAL_LENGTH
# path_obj1 = Path("test1.txt")
# path_obj2 = Path("test2.py")
# long_path = Path("long_file.txt")
# non_existent_path = Path("not_a_real_file.txt")
# simple_list = ["Item 1", "Item 2", "Item 3"]
# path_list = [path_obj1, path_obj2, non_existent_path, long_path]
# mixed_list = ["First item", path_obj1, {"key": "value"}, 123]
# simple_dict = {"name": "Example", "value": 100, "nested": {"a": True}}


#====================================================================
#-------------------------- DICT_FROM_STR() -------------------------
#====================================================================
def dict_from_str(llm_output: str, Pydantic_format: Optional[Type[BaseModel]] = None):
    """
    Converts an LLM's output string into a dict/list, using Pydantic for validation only.
    Returns the validated root data (if Pydantic is used) or the parsed object.
    """
    parsed_obj = json_to_dict(llm_output)
    if parsed_obj is None:
        logging.error(f"{my_name()}: Failed to parse string: {llm_output[:200]}...")
        return None

    if Pydantic_format:
        try:
            pydantic_obj = Pydantic_format.model_validate(parsed_obj)
            return pydantic_obj.model_dump()
        except ValidationError as e:
            logging.error(f"{my_name()}: Pydantic validation failed: {e}")
            return None
        except Exception as e:
            logging.error(f"{my_name()}: Unexpected error during validation: {e}")
            return None
    else:
        return parsed_obj   # return the parsed dict or list without validation


#====================================================================
#-------------------------- JSON_TO_DICT() --------------------------
#====================================================================
def json_to_dict(text: str) -> Optional[Union[dict, list]]:
    """Converts a potentially malformed JSON or Python literal string into a dictionary or list."""
    if not isinstance(text, str):
        logging.error(f"{my_name()}: Input is not a string, received type {type(text)}")
        return None

    text = text.strip()
    if not text or text in ['{}', '[]']:
        logging.warning(f"{my_name()}: Empty or trivial input received.")
        return {} if text == '{}' else []

    # Remove markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Preprocess common LLM issues
    text = re.sub(r',(\s*[}\]])', r'\1', text)  # Remove trailing commas
    # text = re.sub(r"(?<!\\)'", '"', text.replace("\\'", "'"))  # Single to double quotes
    # text = text.replace('\\', '\\\\')  # Escape backslashes
    # Preprocess common LLM issues
    # text = text.replace('doesn"t', "doesn't").replace('can"t', "can't")
    # text = re.sub(r'("(?:[^"\\]|\\.)*?)(\$|\*|\+)([^"]*")', r'\1\\\2\3', text)  # Escape $, *, +
    # text = re.sub(r"(?<!\\)'", '"', text.replace("\\'", "'"))
    # text = re.sub(r',(\s*[}\]])', r'\1', text)
    # text = text.replace('\\', '\\\\')

    # Repair incomplete structures
    if text.startswith('{') and not text.endswith('}'):
        text += '}'
    elif text.startswith('[') and not text.endswith(']'):
        text += ']'

    parsers = [json.loads, json5.loads, demjson3.decode]
    parser_names = ["json", "json5", "demjson3"]
    errors = []

    for parser, name in zip(parsers, parser_names):
        try:
            return parser(text)
        except Exception as e:
            errors.append(f"{name}: {str(e)}")

    # Try parsing as Python literal
    try:
        evaluated_obj = ast.literal_eval(text)
        if isinstance(evaluated_obj, (dict, list)):
            logging.info(f"{my_name()}: Successfully parsed as Python literal.")
            return evaluated_obj
        logging.warning(f"{my_name()}: ast.literal_eval resulted in non-dict/list type: {type(evaluated_obj)}")
        return None
    except (ValueError, SyntaxError,TypeError, MemoryError, RecursionError) as e:
        errors.append(f"ast.literal_eval: {str(e)}")

    logging.error(f"{my_name()}: All parsing attempts failed for: {text}... Errors: {'; '.join(errors)}")
    return None


# --------------------------------------------------------------------
#                       GET_MODEL_NAME() HELPER 
# --------------------------------------------------------------------
def get_model_name(llm: BaseChatModel)->str:
    """ from the model, return the model name """ 
    if isinstance(llm, ChatGoogleGenerativeAI):
        name = llm.model.split("/")[1]  # e.g. models/gemini-2.0-flash 
        return name
    elif hasattr(llm, "model_name"):    # OpenAI, Anthropic, DeepSeek 
        return llm.model_name
    elif hasattr(llm, "model"):         # anyone besides Google that might use "model" instead of "model_name"
        return llm.model
    else:
        return "unknown"
    

import logging
import json
from typing import Any

# Import BaseMessage if you want explicit type checking, otherwise hasattr is fine
# from langchain_core.messages import BaseMessage

def extract_llm_response(response: Any) -> str:
    """
    Robustly extracts string content from various LangChain Runnable outputs.
    Handles BaseMessage types (.content), strings, dictionaries (common keys),
    and falls back to str() representation.
    Args:
        response: The output from a LangChain Runnable's invoke/ainvoke.
    Returns:
        The extracted string content, or a representation if direct extraction fails.
    """
    if response is None:
        return "[No Response Received]"

    # 1. Check for LangChain BaseMessage structure (AIMessage, HumanMessage etc.)
    # if isinstance(response, BaseMessage): # More explicit check
    if hasattr(response, 'content') and isinstance(response.content, str):
        logging.debug(f"{my_name()} Extracting content from '.content' attribute.")
        return response.content

    # 2. Check if it's already a string
    if isinstance(response, str):
        logging.debug(f"{my_name()} Response is already a string.")
        return response

    # 3. Check if it's a dictionary - look for common output keys
    if isinstance(response, dict):
        logging.debug(f"{my_name()} Response is a dict, checking common keys.")
        # Prioritize common keys often used for final output
        common_keys = ["output", "result", "answer", "text", "response"]
        for key in common_keys:
            if key in response and isinstance(response[key], str):
                logging.debug(f"{my_name()} Found string content in dict key: '{key}'")
                return response[key]

        # If no common string key, represent the dict as JSON string
        logging.warning(f"{my_name()} Response is dict, but no common string key found. Serializing to JSON.")
        try:
             # Using ensure_ascii=False for better handling of non-Latin chars
             # Use compact separators for LLM context efficiency
             return json.dumps(response, ensure_ascii=False, separators=(',', ':'))
        except TypeError as e:
             logging.error(f"{my_name()} Could not serialize dict to JSON: {e}. Using str().")
             return str(response) # Fallback if dict isn't JSON serializable

    # 4. Fallback for any other type (int, float, list, custom objects)
    logging.warning(f"{my_name()} Response type {type(response).__name__} not explicitly handled for content extraction. Using str().")
    
    return str(response)
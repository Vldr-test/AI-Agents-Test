

import os
import json
import json5
import demjson3
import logging
from pathlib import Path
from typing import Any, List, Dict, Union, Tuple, Optional, Literal, Type # Import standard types from typing
from pydantic import BaseModel, ValidationError
import re
import ast
from langchain_core.language_models import BaseChatModel

from config import my_name


# --- Configuration for Truncation (Adjust values as needed) ---
# Max characters to read from a single file
MAX_SINGLE_FILE_LENGTH = 200000
# Max total characters for the final combined string
MAX_TOTAL_LENGTH = 500000
# Max items to process from a list
MAX_LIST_ITEMS = 50
# Separator used when joining content from lists
LIST_ITEM_SEPARATOR = "\n\n---\n\n"

# Default max length for the output string to prevent overly long representations
DEFAULT_MAX_STR_LEN = 20000



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


# ====================================================================
#                            DICT_FROM_STR()  
# ====================================================================
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


# --------------------------------------------------------------------
#                       JSON_TO_DICT()  
# --------------------------------------------------------------------
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
    

# --------------------------------------------------------------------
#                       EXTRACT_LLM_RESPONSE()
# --------------------------------------------------------------------
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




# --------------------------------------------------------------------
#                       TO_STRING() HELPER 
# --------------------------------------------------------------------
# Separator for use when strong_formatting is True
STRONG_SEPARATOR = "\n\n------------------------------\n\n" # A more visible separator
# Limit items *processed* from huge lists in strong formatting to prevent massive slowdown/output
MAX_LIST_ITEMS_TO_PROCESS = 100

def to_string(
    data: Any,
    strong_formatting: bool = False # Flag for enhanced visual separation
    ) -> str:
    """
    Converts common Python types into a string representation. NO TRUNCATION.

    If strong_formatting is True, applies clear visual separators between
    top-level list items or dictionary key-value pairs, using simple str()
    for the items/values themselves for simplicity.
    Otherwise, uses more compact representations (compact JSON for dicts/models,
    standard str() for lists/tuples/other).

    Args:
        data: The input data of any type.
        strong_formatting: If True, use verbose separators between top-level list/dict items.

    Returns:
        Full string representation of the data. Returns "" for None.
        Returns an error message string if conversion fails unexpectedly.
    """
    if data is None:
        return ""

    content_str = "[Conversion Error]" # Default

    try:
        # --- Apply Strong Formatting if requested and applicable ---
        if strong_formatting:
            if isinstance(data, (list, tuple)):
                parts = []
                item_type = "Item"
                items_to_process = data[:MAX_LIST_ITEMS_TO_PROCESS] # Limit items *processed*

                for i, item in enumerate(items_to_process):
                    # Format item content simply using str() - prevents deep recursion/complexity
                    item_content_str = str(item)
                    part = f"{item_type} {i+1}:\n{item_content_str}"
                    parts.append(part)

                # Join parts with the strong separator
                content_str = STRONG_SEPARATOR.join(parts)

                # Add note if list items were limited during processing
                if len(data) > len(items_to_process):
                     content_str += f"{STRONG_SEPARATOR}...({len(data) - len(items_to_process)} more list items not shown)"

                return content_str # Return directly after strong formatting

            elif isinstance(data, dict):
                parts = []
                for key, value in data.items():
                    # Format value simply using str()
                    value_str = str(value)
                    # Use repr() for key to handle non-string keys safely
                    part = f"Key: {repr(key)}\nValue:\n{value_str}"
                    parts.append(part)

                # Join parts with the strong separator
                content_str = STRONG_SEPARATOR.join(parts)
                return content_str # Return directly after strong formatting

            # If strong_formatting=True but data isn't list/dict, fall through to standard formatting

        # --- Standard Formatting (if strong_formatting is False or not list/dict) ---
        if isinstance(data, str):
            content_str = data
        elif isinstance(data, bytes):
            content_str = data.decode('utf-8', errors='replace')
        elif isinstance(data, (int, float, bool)):
            content_str = str(data)
        elif isinstance(data, Path):
             content_str = str(data.resolve()) # Return path string
        elif isinstance(data, BaseModel): # Check for Pydantic models
             try: # Use compact JSON representation
                 content_str = data.model_dump_json(indent=None)
             except Exception: content_str = str(data) # Fallback
        elif isinstance(data, dict): # Handles dict when strong_formatting=False
            try: # Use compact JSON, handle non-serializable with str
                content_str = json.dumps(data, ensure_ascii=False, default=str, separators=(',', ':'))
            except TypeError: content_str = str(data) # Fallback
        elif isinstance(data, (list, tuple)): # Handles list/tuple when strong_formatting=False
             content_str = str(data) # Standard list/tuple representation
        else: # Fallback for any other type
            content_str = str(data)

        # --- NO TRUNCATION APPLIED ---

        return content_str

    except Exception as e:
        # Catch unexpected errors during the conversion process itself
        logger = logging.getLogger(__name__) # Use standard logging if available
        error_msg = f"[Error converting type {type(data).__name__} to string: {e}]"
        if logger.hasHandlers() and logger.isEnabledFor(logging.ERROR):
             logger.error(error_msg, exc_info=True)
        else: # Basic print if no logging configured
             print(f"ERROR: {error_msg}")
        return error_msg

"""
def json_to_dict(text: str) -> Optional[Union[dict, list]]:
    # Converts a JSON or Python‐literal string into a dict or list.
    if not isinstance(text, str):
        logging.error(f"{my_name()}: Input is not a string ({type(text)})")
        return None

    text = text.strip()
    if not text or text in ("{}", "[]"):
        return {} if text == "{}" else []

    # Remove markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # 1) Quick Python‐eval pass
    try:
        val = ast.literal_eval(text)
        if isinstance(val, (dict, list)):
            logging.info(f"{my_name()}: Parsed via ast.literal_eval")
            return val
    except Exception:
        pass

    # 2) Normalize “smart” quotes and unescaped apostrophes
    #    Only convert single quotes that appear to be JSON delimiters.
    #    This regex looks for a single‐quoted chunk preceded/followed by JSON punctuation.
    text = re.sub(
        r"(?<=[:\[\{]\s*)'([^']*?)'(?=\s*[,\]\}])",
        r'"\1"',
        text
    )

    # 3) Try the JSON engines
    for parser, name in [(json.loads, "json"), (json5.loads, "json5"), (demjson3.decode, "demjson3")]:
        try:
            logging.debug(f"{my_name()}: Trying {name}")
            return parser(text)
        except Exception as e:
            logging.debug(f"{my_name()}: {name} failed: {e}")

    # 4) Final fallback: try to salvage a Python‐style list by hand
    if text.startswith("[") and text.endswith("]"):
        # extract any double‐quoted chunks
        items = re.findall(r'"([^"]*)"', text)
        if items:
            logging.info(f"{my_name()}: Fallback list‑extract succeeded")
            return items

    logging.error(f"{my_name()}: All parsing attempts failed for: {text[:200]}…")
    return None
"""


#
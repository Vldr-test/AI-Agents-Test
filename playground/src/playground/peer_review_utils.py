import json
import json5
import demjson3
import logging
from typing import Type, Optional, Union
from pydantic import BaseModel, ValidationError
import re
import ast
from peer_review_config import my_name

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
    # text = re.sub(r"(?<!\\)'", '"', text.replace("\\'", "'"))  # Single to double quotes
    text = re.sub(r',(\s*[}\]])', r'\1', text)  # Remove trailing commas
    # text = text.replace('\\', '\\\\')  # Escape backslashes

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
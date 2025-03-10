# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv

# these expect to find a .env file at the directory above the lesson.                                                                                                                     # the format for that file is (without the comment)                                                                                                                                       #API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService
def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

def get_serper_api_key():
    load_env()
    openai_api_key = os.getenv("SERPER_API_KEY")
    return openai_api_key

# --- LLM Inspection   
def agent_inspect_llm(agent):
    print("\n--- LLM Information ---")
    # If the passed object has an 'llm' attribute, inspect that.
    if hasattr(agent, 'llm'):
        llm = agent.llm
        print("LLM Object:", llm)
        print("LLM Class:", llm.__class__.__name__)
        # Check for model attribute names.
        if hasattr(llm, 'model_name'):
            print("LLM Model Name:", llm.model_name)
        elif hasattr(llm, 'model'):
            print("LLM Model:", llm.model)
        else:
            print("LLM Model: Not available for this LLM type.")
    else:
        print("No LLM attribute found. Object type:", agent.__class__.__name__)
    print("--- End LLM Information ---\n")
# --- End LLM Inspection Code

# break line every 80 characters if line is longer than 80 characters
# don't break in the middle of a word
def pretty_print_result(result):
  parsed_result = []
  for line in result.split('\n'):
      if len(line) > 80:
          words = line.split(' ')
          new_line = ''
          for word in words:
              if len(new_line) + len(word) + 1 > 80:
                  parsed_result.append(new_line)
                  new_line = word
              else:
                  if new_line == '':
                      new_line = word
                  else:
                      new_line += ' ' + word
          parsed_result.append(new_line)
      else:
          parsed_result.append(line)
  return "\n".join(parsed_result)

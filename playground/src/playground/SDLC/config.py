
#===========================================================================================
#                            file config.py: GLOBAL CONFIGURATION 
#===========================================================================================

USE_REAL_THREADS = True             # controls threading model; if False, uses Python asyncio threads 
MAX_ITERATIONS = 2                  # max number of self improving iterations. 
SELF_REVIEWS_ALLOWED = True         # self-reviews allowed during peer-reviews 

import logging
import inspect  # Added for my_name lambda
import os

from schemas import (
    Phase, 
    Step,
    PHASE_SEQUENCE,
    STEP_SEQUENCE
)


PHASE_TEAM_ALLOCATION = { 
    Phase.DEFINE: {
        "orchestrator": "product_manager",
        "working_team": ["product_manager", "lead_product_owner"],
        "working_team_lead": "product_manager",
        "reviewing_team": ["lead_product_owner"],
        "reveiwing_team_lead": ["lead_product_owner"]
    },
    Phase.DESIGN: {},
    Phase.REFINE_BACKLOG: {},

    Phase.PLAN_SPRINT: {},
    Phase.CODE: {
        "orchestrator": "scrum_master",
        "working_team": "developers",
    #   "working_team_lead": "product_manager"
        "reviewing_team": ["developers", "scrum_master", "lead_architect"]
    #   "reveiwing_team_lead": ["lead_product_owner"]
    },
    Phase.TEST: {}, 
    Phase.DEPLOY: {},
    Phase.SPRINT_REVIEW: {}
}

# this is the actual execution sequence to follow. Comment out what is not needed: 
BEST_EXECUTION_SEQUENCE = [Phase.DEFINE, Phase.DESIGN, Phase.REFINE_BACKLOG, 
                      Phase.PLAN_SPRINT, Phase.CODE, Phase.TEST, Phase.DEPLOY]

# this is the actual execution sequence for the run: 
EXECUTION_SEQUENCE = [Phase.CODE]

# Configures agents / teams per role 
AGENT_CONFIG = {
    "product_manager": {"model" : "openai:gpt-4o-mini" }, 
    
    # --------------------
    "lead_product_owner": {"model": "openai:gpt-4o-mini"},
    
    "product_owners":  {
                        "po1" : {"model": "openai:gpt-4o-mini"}, 
                        "po2" : {"model": "google_genai:gemini-2.0-flash"}
                        }, 

    # --------------------
    "lead_architect" : {"model" : "openai:gpt-4o-mini"},
    "architects": None, 

    # --------------------
    "scrum_master":    {"model" : "openai:gpt-4o-mini"},  

     # --------------------   
    "lead_developer":  {"model" : "google_genai:gemini-2.0-flash"},  
     
    "developers" :     {
                        "dev1" : {"model" : "openai:gpt-4o-mini"}, 
                        "dev2" : {"model" : "google_genai:gemini-2.0-flash"}, 
                        "dev3" : {"model" : "deepseek:deepseek-chat"}
                        }, 
    
    # --------------------
    "lead_tester" :   {"model" : "google_genai:gemini-2.0-flash"},  
     
    "testers" :       {
                        "tester1" : {"model" : "openai:gpt-4o-mini"}, 
                        "tester2" : {"model" : "google_genai:gemini-2.0-flash"}, 
                        "tester3" : {"model" : "deepseek:deepseek-chat"}
                        } 
}

"""
# temperature is just a default one, will be set separately after the QUERY_TYPE is known 
LLM_PROVIDERS = {
    "openai": {"api_key": os.getenv("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY", "llm_class": CrewAILLM, "temperature": 0.7},
    "google": {"api_key": os.getenv("GOOGLE_API_KEY"), "env_var": "GOOGLE_API_KEY", "llm_class": CrewAILLM, "temperature": 0.7},
    "anthropic": {"api_key": os.getenv("ANTHROPIC_API_KEY"), "env_var": "ANTHROPIC_API_KEY", "llm_class": CrewAILLM, "temperature": 0.7},
    "xai": {"api_key": os.getenv("XAI_API_KEY"), "env_var": "XAI_API_KEY", "llm_class": CrewAILLM, "temperature": 0.7},
    "deepseek": {"api_key": os.getenv("DEEPSEEK_API_KEY"), "env_var": "DEEPSEEK_API_KEY", "llm_class": CrewAILLM, "temperature": 0.7 }
}
"""
 
QUERY_TYPES = {
    "CREATIVE_WRITING": { 
        "criteria": ["creativity", "originality", "coherence"],
        "test_query" : "Напишите короткое стихотворение о печальном голубе, который полюбил робота."
    },
    "SOFTWARE_PROGRAMMING": {
        "criteria":["accuracy", "efficiency", "readability"],
        "test_query": "Provide a Python code snippet to calculate the Levenshtein distance between two strings. Include a brief explanation."
    },
    "MATH": {
        "criteria": ["precision", "clarity", "logic"],
        "test_query": ""
    },
    "TRANSLATION": {
        "criteria": ["accuracy", "fluency", "clarity", "grammar"],
        "test_query": "Translate the English idiom 'barking up the wrong tree' into natural-sounding Hebrew and Russian, and briefly explain the meaning conveyed in each language"
    },
    "SUMMARIZATION": {
        "creteria" : ["coverage_of_key_points", "conciseness", "clarity"],
        "test_query": ""
    },
    "REAL_TIME_DATA_QUERY": {
        "criteria": ["accuracy", "factual_correctness"],
        "test_query": ""
    },
    "OTHER": {
        "criteria": ["quality"],
        "test_query": [
            "Plan a weekend trip for me next month, considering weather, budget, and some fun activities, but I’m not sure where to go yet.",
            "I want to learn a new skill online that could help my career, but I'm overwhelmed by options. Suggest 2-3 potential skills relevant for 'future-proofing' a career in marketing, and outline a basic first step for learning one of them."
            ]
    }
}



QUERY_TYPES_TEMPERATURE = {
    "CREATIVE_WRITING": {
        "openai": 0.9,       # high creativity recommended 
        "gemini": 0.9,       # higher end (0.7-1.0) for creative tasks 
        "anthropic": 1.0,    # closer to 1.0 for generative tasks 
        "deepseek": 1.5,     # official recommendation for creative writing
        "xai": 1.0,          # no official value; high (range 0-1) for creativity
        "meta": 1.0,         # up to 1.0 (default 0.9) for more diverse output 
        "mistral": 0.8       # higher end (default 0.7) for creative output 
    },
    "SOFTWARE_PROGRAMMING": {
        "openai": 0.0,       # use 0 for deterministic code generation
        "gemini": 0.0,       # low (0-0.3) for deterministic outputs like code 
        "anthropic": 0.0,    # closer to 0 for analytical tasks  
        "deepseek": 0.0,     # coding/math recommended at 0.0 
        "xai": 0.0,          # no official data; assume near 0 for code accuracy 
        "meta": 0.0,         # no official; community uses ~0 for coding 
        "mistral": 0.0       # no official; use near 0 for correct code (Nemo uses 0.3)  
    },
    "MATH": {
        "openai": 0.0,       # 0-0.2 for focused tasks like math""
        "gemini": 0.0,       # treat like classification  
        "anthropic": 0.0,    # analytical task  
        "deepseek": 0.0,     # math grouped with coding at 0.0 
        "xai": 0.0,          # no official; likely 0 for precise calculation 
        "meta": 0.0,         # no official; low temp for deterministic output 
        "mistral": 0.0       # no official; low temp to avoid errors 
    },
    "TRANSLATION": {
        "openai": 0.0,       # no explicit value; ~0-0.2 for faithful translation 
        "gemini": 0.0,       # no explicit; use low (0-0.3) for deterministic tasks 
        "anthropic": 0.0,    # no explicit; treat as analytical -> low temp 
        "deepseek": 1.3,     # recommended higher temperature for translation 
        "xai": 0.0,          # no official; presumably low to stick to source text 
        "meta": 0.0,         # no official; low temp to avoid mistranslation (default 0.9) 
        "mistral": 0.0       # no official; low temp for literal accuracy 
    },
    "SUMMARIZATION": {
        "openai": 0.2,       # generally keep low for factual summary (0-0.3) 
        "gemini": 0.2,       # start low (~0.2), increase if summary too generic 
        "anthropic": 0.2,    # no explicit; low for accurate summaries (Claude default 1.0) 
        "deepseek": 1.0,     # not given; ~1.0 used for data analysis tasks (proxy) 
        "xai": 0.0,          # no official; assume low for coherence (decisive mode) 
        "meta": 0.0,         # no official; likely low to avoid hallucination 
        "mistral": 0.3       # Mistral Nemo-Instruct recommends 0.3 for guided tasks 
    },
    "REAL_TIME_DATA_QUERY": {
        "openai": 0.0,       # use 0 to avoid creativity when using external data 
        "gemini": 0.0,       # no direct quote; best to use 0 for RAG to ensure accuracy 
        "anthropic": 0.0,    # no explicit; assumed 0 to stick strictly to retrieved info 
        "deepseek": None,    # not specified by DeepSeek (likely would be low)
        "xai": None,         # not specified by xAI (likely low for accuracy)
        "meta": None,        # not specified by Meta (community uses 0 for RAG)
        "mistral": None      # not specified by Mistral (usually set ~0 for RAG)
    },
    "CLASSIFICATION": {
        "openai": 0.0,       # 0 for deterministic classification 
        "gemini": 0.0,       # Google recommends temp=0 for classification tasks 
        "anthropic": 0.0,    # analytical/multiple-choice -> temp ~0 
        "deepseek": 0,       # no specific guidance (would default to 0)
        "xai": 0,            # no specific guidance (likely 0)
        "meta": 0,           # no specific guidance (likely 0)
        "mistral": 0.1       # no specific guidance (likely low)
    },
    "OTHER": {
        "openai": 1.0,       # default API temperature is 1.0 
        "gemini": 0.7,       # default ~1.0 for models (but best practice start ~0.2) 
        "anthropic": 1.0,    # default is 1.0 (range 0-1) 
        "deepseek": 1.0,     # default 1.0, but general conversation mode uses 1.3 
        "xai": 0.7,          # default not explicitly mentioned (range 0-1) 
        "meta": 0.9,         # default temperature for LLaMA models is 0.9 
        "mistral": 0.3       # default 0.7; Nemo-Instruct model recommends 0.3 
    }
}

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()], force=True)

VERBOSE = False

# Function to get current function name for logging
my_name = lambda: inspect.currentframe().f_back.f_code.co_name
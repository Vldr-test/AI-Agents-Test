
import os
import json
from dotenv import load_dotenv
import logging
from crewai.flow import Flow, listen, start
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase
from langchain_openai import ChatOpenAI
from crewai.llm import LLM as CrewAILLM
from pydantic import BaseModel, ValidationError as VALIDATIONError  # Renamed ValidationError to VALIDATIONError
import asyncio


# Global logging
VERBOSE = False
LOG_LEVEL=logging.INFO
os.environ["CREWAI_SUPPRESS_TELEMETRY"] = "true"  # CrewAI telemetry off
os.environ["LITELLM_SUPPRESS_LOGGING"] = "true"   # LiteLLM debug logs off

logging.basicConfig(
    level=LOG_LEVEL,
    handlers=[logging.StreamHandler()],  force=True  #Forces reconfiguration
  )
llm_logger = logging.getLogger("LiteLLM")
llm_logger.setLevel(LOG_LEVEL)  # Set to WARNING to reduce noise

crew_logger = logging.getLogger("crewai")
crew_logger.setLevel(LOG_LEVEL)

# Global definition of a leader
LEADER_MODEL = "gpt-3.5-turbo"  # Can be changed to any model

# These are the names of models we would like to use for generating responses (and later, peer reviews)
AGENT_MODELS = [
    # "gpt-4",
    "gpt-3.5-turbo",
    "anthropic/claude-3-haiku-20240307",  # Updated model name
    "gemini/gemini-2.0-flash",
    "xai/grok"
]

MODEL_PROVIDER_MAP = {
    "gpt-": "openai",
    "gemini/": "google",
    "anthropic/": "anthropic",
    "xai/": "xai"
}

LLM_PROVIDERS = {
    "openai": {"api_key": os.getenv("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY", "llm_class": ChatOpenAI, "temperature": 0.7},
    "google": {"api_key": os.getenv("GOOGLE_API_KEY"), "env_var": "GOOGLE_API_KEY", "llm_class": CrewAILLM, "temperature": 0.7},
    "anthropic": {"api_key": os.getenv("ANTHROPIC_API_KEY"), "env_var": "ANTHROPIC_API_KEY", "llm_class": CrewAILLM, "temperature": 0.7},
    "xai": {"api_key": os.getenv("XAI_API_KEY"), "env_var": "XAI_API_KEY", "llm_class": CrewAILLM, "temperature": 1.0}
}


# PROMPTS FOR ALL TASKS: 
def get_analysis_prompt(query):
    """Generate the prompt for classifying the query type and criteria."""
    prompt = (
        f"Analyze the query '{query}' and output a JSON object with: "
        "'query_type' (e.g., 'programming', 'math', 'creative writing', 'translation') and "
        "'criteria' (e.g., ['accuracy', 'efficiency']). "
        "Example: {'query_type': 'programming', 'criteria': ['accuracy', 'efficiency', 'readability']}. "
        "Return ONLY the JSON—no extra text or markers like ```json```."
    )
    return prompt

def get_generation_prompt(query, query_type, criteria):
    """Generate the prompt for creating responses based on query type and criteria."""
    prompt = (
        f"Generate a {query_type} response for '{query}'. "
        f"Focus on {', '.join(criteria)}. "
        "Return ONLY the response as a string—no extra text or markers like ```json```."
    )
    return prompt

def get_peer_review_prompt(agent_names, query, responses, criteria):
    """Generate the prompt for peer reviewing responses."""
    # Handle example agents safely
    agent1 = agent_names[0] if len(agent_names) > 0 else "Agent1"
    agent2 = agent_names[1] if len(agent_names) > 1 else "Agent2"
    example = (
        f"'{agent1}': {{'improvement_points': ['Improve {criteria[0]}', 'Enhance {criteria[1]}'], "
        f"'{criteria[0]}': 7, '{criteria[1]}': 8, '{criteria[2]}': 9}}, "
        f"'{agent2}': {{'improvement_points': ['Refine {criteria[0]}', 'Boost {criteria[2]}'], "
        f"'{criteria[0]}': 6, '{criteria[1]}': 7, '{criteria[2]}': 8}}"
    )
    prompt = (
        f"Output ONLY a single JSON dictionary—no prose, no summaries, no extra text. "
        f"Review these responses from your peers {agent_names} to '{query}': {json.dumps(responses)}. "
        f"Evaluate each based on {criteria}. "
        f"Provide 2-5 specific actionable improvement points and score each criterion from 1-10 (1=poor, 10=excellent). "
        f"Return a JSON dictionary like {{{example}}}. "
        "Output ONLY this JSON—no markers like ```json```."
    )
    return prompt

def get_final_response_prompt(agent_names, query, peer_reviews, criteria):
    """Generate the prompt for calculating the winner from peer reviews."""
    agent1 = agent_names[0] if len(agent_names) > 0 else "Agent1"
    prompt = (
        f"Output ONLY a single JSON dictionary—no prose, no summaries, no extra text. "
        f"These are peer reviews by your team agents {agent_names} for '{query}': {json.dumps(peer_reviews)}. "
        f"Calculate the average score across all valid JSON reviews for each agent based on {criteria} (skip non-JSON reviews). "
        f"Pick the winner with the highest average score. "
        f"Return a JSON object like {{'Winner': '{agent1}', 'Final_score': 7.8}}. "
        "Output ONLY this JSON—no markers like ```json```."
    )
    return prompt


# Global Helper function: accepts model_name. Returns an initialized agent + a unique name
def create_agent_from_name(model_name, base_name=""):
    # find a provider:
    provider = next((prov for prefix, prov in MODEL_PROVIDER_MAP.items() if model_name.startswith(prefix)), None)
    if not provider:
        logging.error(f"Unknown provider for model: {model_name}")
        return None, None

    config = LLM_PROVIDERS[provider]
    if not config["api_key"]:
        logging.error(f"Missing API key ({config['env_var']}) in .env for {provider}")
        return None, None
    try:
        llm = config["llm_class"](model=model_name, api_key=config["api_key"], temperature=config["temperature"])

        agent = Agent(
            role="agent",
            goal="Generate responses and review others' responses.",
            backstory="Experienced thinker known for fair opinions.",
            verbose=VERBOSE,
            llm=llm,
            allow_delegation=False,
            tools=[]
        )

        agent_name = base_name + "-" + provider.upper() + "-" + model_name.replace('/', '-')

        # Crew expects one more attribure, apparently:
        agent.config = {}  # Add at the start of the class if not initialized
        agent.config[agent_name] =  {}

        logging.info(f"Initialized agent {agent_name} with model: {model_name}")
        return agent, agent_name

    except Exception as e:
        logging.error(f"Failed to initialize agent for {provider}/{model_name}: {str(e)} - Skipping")
        return None, None


# Key Workflow definition:
class TeamworkFlow(Flow):

    def __init__(self, query):
        super().__init__()
        self.state["query"] = query
        self.leader_agent = None
        self.leader_agent_name = None
        self.agents = []
        self.agent_names = []

        # create a leader: 
        leader_name = LEADER_MODEL
        self.leader_agent, self.leader_agent_name = create_agent_from_name(leader_name, "Leader")
        if self.leader_agent == None:
            logging.error(f"Can't create Leader for leader_name: {leader_name}")
            raise RuntimeError(f"Can't create Leader for leader_name: {leader_name}. Execution will be stopped")

        # create agents:  
        agent_models = AGENT_MODELS
        for model_name in agent_models:
            agent, agent_name = create_agent_from_name(model_name, "Agent")
            if agent is None:  
                logging.warning(f"Skipping {model_name} because the agent could not be created.")
                continue  # Skip to the next model, so that all agents are valid. If not, CrewAI fails if agent is None

            self.agents.append(agent)
            self.agent_names.append(agent_name)
            logging.info(f"Created agent: {agent_name}")    


    def kickoff(self):
        logging.info("Flow kickoff started.")
        super().kickoff()
        logging.info("Flow kickoff completed.")

   
    @start()
    def analyze_query(self):
        """Analyze the query and output a Pydantic object in state["analysis"]"""
        logging.info("Starting analyze_query")
        self.state["analysis_state"] = {}
        
        query = self.state["query"]

        query_analysis_task = Task(                 # create a task for the pseudo-crew 
            description=get_analysis_prompt(query),  
            agent=self.leader_agent,
            expected_output="A JSON string matching the schema: {query_type: str, criteria: list[str]}."
        )
        logging.info(f"task created for {self.leader_agent_name}")
        analysis_pseudo_crew = Crew(
            agents=[self.leader_agent],
            tasks=[query_analysis_task],
            process=Process.sequential,
            verbose=VERBOSE
        )
  
        if analysis_pseudo_crew is None:
            logging.error(f"Can't create Analysis Pseudo Crew Leader")
            raise RuntimeError(f"Can't create Crew leader. Execution will be stopped")
        
        logging.info(f"Crew created for {self.leader_agent_name}")
        analysis_result = analysis_pseudo_crew.kickoff()  # Synchronous kickoff

        # Parse and validate JSON response
        response = analysis_result.tasks_output[0].raw
        try:
            analysis_dict = json.loads(response)       # we expect {'query_type' : ['criteria', ...]}
            self.state["analysis_state"]["query_type"] = analysis_dict["query_type"]
            self.state["analysis_state"]["criteria"] = analysis_dict["criteria"]
            logging.info(f"Leader instruction (parsed): {self.state['analysis_state']}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from analysis state: {e}")
            raise RuntimeError(f"Can't parse query analysis. Execution will be stopped")
       
        return self.state  # Return state to ensure flow continuation

       
    @listen("analyze_query")
    async def generate_responses(self):
        """ Instantiates agents using globabl list, runs them, and adds their responses to state['responses'] """
        
        logging.info(f"Starting generate_responses - Agents available: {len(self.agents)}")

        # initializing variables
        tasks = [] # will store async tasks 
        # pseudo_crews = []  # these are the pseudo-crews each consisting of one "agent" for concurrent execution
           
        task_prompt = get_generation_prompt(
           query=self.state["query"],
           query_type=self.state["analysis_state"]["query_type"],
            criteria=self.state["analysis_state"]["criteria"]
)

        self.state["responses_state"] = {} 

        for agent, agent_name in zip(self.agents, self.agent_names):

            task = Task(
                description= task_prompt, 
                agent=agent,
                expected_output="A detailed response based on the description."
            )

            # create a new pseudo-crew:
            pseudo_crew=Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=VERBOSE
            )
            tasks.append(asyncio.create_task(pseudo_crew.kickoff_async()))
            logging.info(f"pseudo-crew launched for {agent_name}") 
     
        # "for" loop ends
        results = await asyncio.gather(*tasks)
        logging.info(f"Aysnc generation finished. Agent_names: {self.agent_names}")

        self.state["responses_state"] = {
            agent_name: result.tasks_output[0].raw for agent_name, result in zip(self.agent_names, results)
        }

        return self.state  # Return state to ensure flow continuation"


    @listen("generate_responses")
    async def peer_review(self):
        
        self.state["responses_state"] = {}
        criteria = self.state["analysis_state"]["criteria"]
         

        peer_review_prompt = get_peer_review_prompt(
            agent_names=self.agent_names,
            query=self.state["query"],
            responses=self.state["responses_state"],
            criteria=criteria
        )

        tasks = [] # will store tasks 

        logging.info(f"Starting peer review. Prompt: {peer_review_prompt}")
     
        for agent, agent_name in zip(self.agents, self.agent_names):
            # creating a new task:
            task = Task(
                description= peer_review_prompt, 
                agent=agent,
                expected_output="a single JSON object with improvement point and scores"
            )
                    # create a new pseudo-crew:
            pseudo_crew=Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=VERBOSE
            )
            tasks.append(asyncio.create_task(pseudo_crew.kickoff_async()))
            logging.info(f"pseudo-crew launched for {agent_name}") 
     
        # "for" loop ends
        results = await asyncio.gather(*tasks)
        logging.info(f"Aysnc peer review finished")

        self.state["peer_review_state"] = {
            agent_name: result.tasks_output[0].raw for agent_name, result in zip(self.agent_names, results)
        }
            
        logging.info(f"State after peer_review: {self.state["peer_review_state"]}")
        return self.state  # Return state to ensure flow continuation"


    @listen("peer_review")
    def final_response(self):
        # the Leader goes over the peer reviews, collects the stats and picks up the winner 
        self.state["final_response_state"] = {} 

        expected_output_prompt = "a single JSON object with a winner and its average score"

        final_response_prompt = get_final_response_prompt(
            agent_names=self.agent_names,
            query=self.state["query"],
            peer_reviews=self.state["peer_review_state"],
            criteria=self.state["analysis_state"]["criteria"]
        )

        logging.info(f"Starting final analysis. Prompt: {final_response_prompt}")
     
        task = Task(
            description= final_response_prompt, 
            agent=self.leader_team.leader_agent,
            expected_output=expected_output_prompt
        )
            
        # run Task sequentially:
        self.leader_team.crew.tasks = [task]  # Replace tasks
        results = self.leader_team.kickoff()  # Synchronous kickoff
        self.state["final_response"] = results.tasks_output[0].raw
        
        logging.info(f"State after final_response: {self.state["final_response_state"]}")

        return self.state  # Return state to ensure flow continuation"


if __name__ == "__main__":

    # set logging level globally:
    logging.getLogger().setLevel(LOG_LEVEL)

    # flow = TeamworkFlow("Write a Python function to find the longest palindromic substring...")
    flow = TeamworkFlow("Напишите короткое стихотворение о судьбе крота, который женился на кошке")
    flow.kickoff()
   
    # *************************************************************
    json_str = json.dumps(flow.state, indent=2, ensure_ascii=False)
    formatted_output = json_str.replace("\\n", "\n")  # Replace escaped newlines with real ones
    print(f"\nFinal state: {formatted_output}")


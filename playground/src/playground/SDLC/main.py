# ======================================================================
#           File design.py: high-level design in pseudo-code  
# ======================================================================

from config import (
    USE_REAL_THREADS,           # which threading model to chose 
    MAX_ITERATIONS,             # max iterations for one phase improvement loop 
    QUERY_TYPES,                # different types of queries might require different teams' setup 
    EXECUTION_SEQUENCE,         # actual execution sequence to include / exclude phases 
    AGENT_CONFIG,               # defining one agent / team config 
    PHASE_TEAM_ALLOCATION,      # defining teams per phase 
    SELF_REVIEWS_ALLOWED,       # flag to enable/disable self-reviews in peer_reviews 
    my_name
)

from prompts import (
    get_response_generation_prompt, 
    get_improvement_prompt,  
    get_peer_review_prompt, 
    get_harmonize_improvement_points_prompt,
    get_phase_review_with_human_prompt
)

from utils import dict_from_str, artifacts_to_str, extract_llm_response, to_string 

from schemas import (
            Step, Phase, 
            PhaseExecutionResult, PhaseMetadata,
            NextSteps, ContextWrapper, 
            AllResponses,
            PeerReviewAnalysis,  
            ReviewItem, PeerReviewItem, AllPeerReviews, 
            PhaseExecutionResult
            ) 

# from peer_review.leader import LeaderAgent

import time
from dotenv import load_dotenv
import logging
from langchain.chat_models import init_chat_model
from langchain.agents import create_tool_calling_agent, AgentExecutor
 
import asyncio
if USE_REAL_THREADS: from concurrent.futures import ThreadPoolExecutor 

from langchain.tools import BaseTool

from pydantic import BaseModel, ValidationError
import asyncio
from typing import Type, List, Dict, Optional, Tuple, Union, Any, Literal, Set



from langchain_core.runnables import Runnable, RunnableConfig 
from langchain_core.runnables.utils import Input, Output


class HITL_tool(BaseTool):
    name: str = "HITL_tool" # Give it a clear name
    description: str = ( # Crucial for the LLM to know when to use it
        "Use this tool when you need clarification, confirmation, or additional input "
        "directly from the human user. Ask a clear question for the human based on your current task. "
    )

    # Synchronous version
    def _run(self, query: str) -> str:
        """Synchronously ask the human for input."""
        print(f"\n Agent asks: {query}") # Display the question from the LLM
        user_response = input("Your Input: ")    # Get input from console
        return user_response

class Agent:
    """
    A simple wrapper to associate a logical name with any LangChain Runnable
    and delegate calls to it.
    """
    def __init__(self, name: str, runnable: Runnable):
        if not isinstance(name, str) or not name:
            raise ValueError("Agent name must be a non-empty string")
        if not isinstance(runnable, Runnable):
            # Check if it's a LangChain object that *should* be runnable
            # This check might need refinement based on specific LangChain versions/types
            logging.warning(f"Agent '{name}' was passed a non-Runnable object: {type(runnable)}. "
                            "Ensure it supports invoke/ainvoke.")
        self.name = name
        self.llm_name = self._get_llm_name(runnable)  # Make it a public attribute directly

        self.runnable = runnable
        logging.debug(f"NamedAgent created: {self.name} (Runnable Type: {runnable.__class__.__name__})")

    # --- Core Runnable Methods (Delegation) ---
    def _get_llm_name(self, runnable: Runnable) -> Optional[str]:
        """
            Attempts to extract the LLM name and provider from a LangChain Runnable.
        """
        try:
            # Common case: Runnable is an LLM
            if hasattr(runnable, "model_name") and hasattr(runnable, "_llm_type"):
                return f"{runnable._llm_type}:{runnable.model_name}"
            elif hasattr(runnable, "llm") and hasattr(runnable.llm, "model_name") and hasattr(runnable.llm, "_llm_type"):
                return f"{runnable.llm._llm_type}:{runnable.llm.model_name}"
            # Handle Chat Models
            elif hasattr(runnable, "model_name") and hasattr(runnable, "_model_type"):
                return f"{runnable._model_type}:{runnable.model_name}"
            elif hasattr(runnable, "llm") and hasattr(runnable.llm, "model_name") and hasattr(runnable.llm, "_model_type"):
                return f"{runnable.llm._model_type}:{runnable.llm.model_name}"
            # Handle OpenAI client directly
            elif hasattr(runnable, "client") and hasattr(runnable, "model"):
                return f"openai:{runnable.model}"
            elif hasattr(runnable, "llm") and hasattr(runnable.llm, "client") and hasattr(runnable.llm, "model"):
                return f"openai:{runnable.llm.model}"
            # Add more cases for other LLM integrations if needed
        except Exception as e:
            logging.warning(f"Could not extract LLM name from {runnable.__class__.__name__}: {e}")
        return None

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """ Synchronously invoke the underlying runnable. """
        logging.debug(f"NamedAgent '{self.name}': invoking synchronously...")
        return self.runnable.invoke(input, config=config)

    async def ainvoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """ Asynchronously invoke the underlying runnable. """
        logging.debug(f"NamedAgent '{self.name}': invoking asynchronously...")
        return await self.runnable.ainvoke(input, config=config)
    
    def __repr__(self):
        return f"NamedAgent(name='{self.name}', runnable={self.runnable.__class__.__name__})"

    # --- Optional: Delegate other common Runnable methods if needed ---
    """
    def stream(self, input: Input, config: Optional[RunnableConfig] = None) -> Iterator[Output]:
         logging.debug(f"NamedAgent '{self.name}': streaming synchronously...")
         return self.runnable.stream(input, config=config)

    async def astream(self, input: Input, config: Optional[RunnableConfig] = None) -> AsyncIterator[Output]:
         logging.debug(f"NamedAgent '{self.name}': streaming asynchronously...")
         # Ensure the underlying runnable supports astream, otherwise this might error
         async for chunk in self.runnable.astream(input, config=config):
             yield chunk

    def batch(self, inputs: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None, *, return_exceptions: bool = False, **kwargs: Any) -> List[Output]:
         logging.debug(f"NamedAgent '{self.name}': processing batch synchronously...")
         return self.runnable.batch(inputs, config=config, return_exceptions=return_exceptions, **kwargs)

    async def abatch(self, inputs: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None, *, return_exceptions: bool = False, **kwargs: Any) -> List[Output]:
         logging.debug(f"NamedAgent '{self.name}': processing batch asynchronously...")
         return await self.runnable.abatch(inputs, config=config, return_exceptions=return_exceptions, **kwargs)
    """


# ======================================================================
#         [SDLC] LIFECYCLE CLASS - encapsulates high-level logic 
# ======================================================================
class Lifecycle(): 
    agent_pool: Dict[str, Any] 
    """
    Stores initialized agents/teams {role_name: Agent or List[Agent]}: 
    {
        "product_manager": <Agent object for product_manager>,  # Instance of your Agent class
        "lead_architect":  <Agent object for lead_architect>,   # Instance of your Agent class
        "developers": [                                         # List of Agent objects
            <Agent object for dev1>,
            <Agent object for dev2>
        ],
        "architects": []                                        # Empty list
        # Note: Individual members like "dev1", "dev2" are *not* keys here unless
        # explicitly added or required as individual roles elsewhere.
    }
    """
        
    phase_allocations: Dict[Phase, Dict[str, Any]] 
    """
    Stores allocation of agents to teams: 
    {
        Phase.DEFINE: {
            "orchestrator": "product_manager",            # Value is str
            "working_team": ["product_manager"],          # Value is List[str]
            "reviewing_team": ["lead_product_owner", "lead_architect"] # Value is List[str]
            # Missing leads assumed unused or derived otherwise
        },
        Phase.DESIGN: {
            "orchestrator": "lead_architect",             # Value is str
            "working_team": ["architects"],               # Value is List[str] (referring to team name)
            "reviewing_team": ["lead_product_owner", "lead_developer", "architects"], # Value is List[str] (mix of roles and team)
    },
    """
    execution_sequence: List[Phase]                 # marks  active phases within a lifecycle 
    current_phase: Phase
    state: ContextWrapper                           # holds history / context 

    def __init__(self, agent_config: Dict = AGENT_CONFIG, 
                phase_alloc: Dict = PHASE_TEAM_ALLOCATION, 
                exec_seq: List = EXECUTION_SEQUENCE):
        """
        Initialize ContextWrapper, store configs, and initialize ONLY the agents
        required by the execution_sequence and phase_allocations.
        """
        self.state = ContextWrapper()
        self.agent_config = agent_config
        self.phase_allocations = phase_alloc
        self.execution_sequence = exec_seq
        self.current_phase = None
        self.agent_pool = {}
        logging.info(f"{my_name()}: Lifecycle initialization started")  

        # --- Determine Required Roles/Teams ---
        required_roles: Set[str] = set()
        
        for phase in self.execution_sequence:
            alloc = self.phase_allocations.get(phase)
            if not alloc:
                # Optional: Warn if a phase in sequence has no allocation
                logging.error(f"\n{my_name()} Warning: No allocation found for phase {phase.value} in sequence.")
                continue

            # Add orchestrator
            if orchestrator_name := alloc.get("orchestrator"):
                 required_roles.add(orchestrator_name)

            # Add working team members/teams
            working_cfg = alloc.get("working_team", [])
            if isinstance(working_cfg, str): # e.g., "developers"
                 required_roles.add(working_cfg)
            elif isinstance(working_cfg, list): # e.g., ["product_manager"]
                 required_roles.update(working_cfg)

            # Add reviewing team members/teams
            reviewing_cfg = alloc.get("reviewing_team", [])
            if isinstance(reviewing_cfg, str):
                 required_roles.add(reviewing_cfg)
            elif isinstance(reviewing_cfg, list):
                 required_roles.update(reviewing_cfg)

            # Add any explicitly named leads if defined differently
            # (Example using keys like "working_team_lead")
            if lead := alloc.get("working_team_lead"): required_roles.add(lead)
            if lead := alloc.get("reviewing_team_lead"): required_roles.add(lead)

        # --- Initialize Only Required Agents/Teams ---
        for role_or_team_name in required_roles:
            if role_or_team_name not in self.agent_config:
                 logging.error(f"\n{my_name()} Required role/team '{role_or_team_name}' not found in AGENT_CONFIG. Skipping.")
                 continue

            if role_or_team_name in self.agent_pool: # Avoid re-initializing if already done (e.g., lead listed separately)
                 continue

            config_data = self.agent_config[role_or_team_name]

            if config_data is None or config_data == {}: # Handle architects: {} or None
                logging.error(f"\n{my_name()} No specific config for '{role_or_team_name}'. Initializing as empty list/None.")
                # Store empty list for teams, maybe None for single roles? Consistent empty list is safer.
                self.agent_pool[role_or_team_name] = []
                continue

            if "model" in config_data: # Single agent role
                try:
                    agent = self._initialize_agent(role_or_team_name, config_data["model"])
                    self.agent_pool[role_or_team_name] = agent
                except Exception as e:
                    logging.error(f"\n{my_name()} ERROR initializing single agent '{role_or_team_name}': config: {config_data["model"]}: {e}")
            elif isinstance(config_data, dict): # Team definition
                try:
                    team_agents = self._initialize_team(config_data)
                    self.agent_pool[role_or_team_name] = team_agents
                except Exception as e:
                    logging.error(f"\n{my_name()} ERROR initializing '{role_or_team_name}' config:{config_data}: {e}")
            else:
                logging.error(f"\n{my_name()} Unrecognized config format for '{role_or_team_name}'. Skipping.")

        logging.info(f"{my_name()}: Agent pool initialization complete: {list(self.agent_pool.keys())}")


    # --------------------------------------------------------------------
    #                       INITIALIZE_AGENT()  
    # --------------------------------------------------------------------
    def _initialize_agent(self, agent_name: str, model_name: str) -> Optional[Agent]:
        """
            Initializes a single LangChain runnable and wraps it in an Agent object.
            Returns None on failure.
        """
        try:
            # Assume init_chat_model handles string IDs or more complex configs
            runnable = init_chat_model(model_name)
            # Optional: Add check if runnable is actually Runnable
            if not isinstance(runnable, Runnable):
                raise TypeError(f"\n{my_name()} init_chat_model did not return a Runnable for {agent_name}")
            
            agent = Agent(name=agent_name, runnable=runnable) # Use the Agent wrapper
            logging.info(f"{my_name()} Successfully created agent: {agent_name} using {model_name}") # Logging removed
            return agent
        except Exception as e:
            logging.error(f"\n{my_name()} Can't create an agent for: {agent_name} using {model_name}: {e}") # Logging removed
            return None  

    # --------------------------------------------------------------------
    #                       INITIALIZE_TEAM()  
    # --------------------------------------------------------------------
    def _initialize_team(self, agent_cfg: Dict[str, str]) -> List[Agent]:
        """
            Initialize actual LLM instances with Langchain. 
            agent_cfg maps logical_agent_name -> { 'model': model_string, ... }

            Args: { "agent_name" : "model_name", ...} 
            Return: list of agents
        """  
        agents: List[Agent] = []
        for agent_name, cfg in agent_cfg.items():
            # handle both {"model": "..."} and plain strings
            if isinstance(cfg, dict) and "model" in cfg:
                model_name = cfg["model"]
            elif isinstance(cfg, str):
                model_name = cfg
            else:
                logging.error(f"\n{my_name()}: Bad config for '{agent_name}' : {cfg}")
                continue

            agent = self._initialize_agent(agent_name, model_name)
            if agent:
                agents.append(agent)

        if not agents:
            logging.error(f"\n{my_name()}: Can't create even one agent for team.")
            raise RuntimeError(f"No agents in team ")

        return agents

    #---------------------------------------------------------------------------
    #                            _get_agent()
    #---------------------------------------------------------------------------
    def _get_agent(self, role_name: Optional[str]) -> Optional[Agent]:
        """
            Given a single role_name, return either:
            - the Agent instance
            - None (if role_name is None or missing)
        """
        if not role_name:
            return None
        agent = self.agent_pool.get(role_name)
        if isinstance(agent, list):
            logging.error(f"\n{my_name()}: Expected single Agent for role '{role_name}', got list.")
            return None
        return agent

    #---------------------------------------------------------------------------
    #                            _get_agents()
    #---------------------------------------------------------------------------
    def _get_agents(self, team_cfg: Optional[Union[str, List[str]]]) -> List[Agent]:
        """
            Given either a str or a list of str, look up each in self.agent_pool and
            return a flat List[Agent].
        """
        if not team_cfg:
            return []
        names = [team_cfg] if isinstance(team_cfg, str) else team_cfg
        team = []
        for name in names:
            member = self.agent_pool.get(name)
            if member is None:
                logging.error(f"\n{my_name()}: No entry in agent_pool for '{name}'")
                continue
            if isinstance(member, list):
                team.extend(member)
            else:
                team.append(member)
        return team

    # ----------------------------------------------------------------------
    #                    _GET_TEAM_DESCRIPTION() 
    # ----------------------------------------------------------------------
    def _get_team_description(self, orchestrator:Agent, 
                              working_team:List[Agent], 
                              reviewing_team: List[Agent]) -> str:
        """
            Returns a string with the team breakdown
        """
        description = ""
        if orchestrator:
            description += f"\nThe orchestrator for this task is: '{orchestrator.name}: {orchestrator.llm_name}'.\n"

        if working_team:
            description += "\nThe working team is:\n"
            for agent in working_team:
                description += f"- '{agent.name}: {orchestrator.llm_name}' "

        if reviewing_team:
            description += "\nThe reviewing team is:\n"
            for agent in reviewing_team:
                description += f"- '{agent.name}: {orchestrator.llm_name}'.\n"

        return description
 
    # ----------------------------------------------------------------------
    #                    EXECUTE_LIFECYCLE(): MAIN LOOP 
    # ----------------------------------------------------------------------
    def execute_lifecycle(self, query: str, initial_artifacts: Optional[Any] = None):
        """
        Drive the lifecycle through each Phase in self.execution_sequence.

        For each phase:
          1) Pull teams from self.phase_allocations → self.agent_pool
          2) Stringify the current artifact
          3) Call execute_phase(...) → PhaseExecutionResult(analysis, decision, meta)
          4) Persist analysis, decision, meta into ContextWrapper
          5) Advance current_artifact ← analysis.response
          6) On PLAN_SPRINT bump sprint counter
          7) next_phase ← decision.next_phase
          8) Exit on Phase.STOP or unknown phase
        """
        sprint = 0
        self.state.initialize_sprint(sprint)
        current_artifact = initial_artifacts

        # pick up the very first phase (or STOP if none)
        next_phase = self.execution_sequence[0] if self.execution_sequence else Phase.STOP

        while True:
            self.current_phase = next_phase
            if next_phase == Phase.STOP:
                logging.info(f"\n{my_name()}: Reached STOP phase. Exiting lifecycle.")
                break

            logging.info(f"\n{my_name()}: Starting phase '{next_phase.value}' (Sprint {sprint})")

            # 1) resolve teams for this phase
            alloc = self.phase_allocations.get(next_phase, {})
            orchestrator   = self._get_agent(alloc.get("orchestrator"))
            working_team   = self._get_agents(alloc.get("working_team"))
            reviewing_team = self._get_agents(alloc.get("reviewing_team"))

            # 2) stringify whatever the artifact is
            # artifacts_str = artifacts_to_str(current_artifact)

            # 3) run the core loop
            phase_result: PhaseExecutionResult = self.execute_phase(
                phase          = next_phase,
                query          = query,
                working_team   = working_team,
                reviewing_team = reviewing_team,
                orchestrator   = orchestrator,
                artifacts      = current_artifact,
                max_iterations = MAX_ITERATIONS
            )

 
            # 4) persist into the context wrapper
            self.state.save_phase_results(phase_result)

            # 5) !!! set up for the next phase
            # current_artifact = analysis.response

            # 6) bump sprint if we’ve just done planning
            if next_phase == Phase.PLAN_SPRINT:
                sprint += 1
                self.state.initialize_sprint(sprint)
                logging.info(f"\n{my_name()}: Starting Sprint {sprint}.")

            # 7) follow the human’s routing
            next_phase = phase_result.decision.next_phase

            # 8) guard against bad routing
            if next_phase not in self.execution_sequence and next_phase != Phase.STOP:
                logging.error(
                    f"\n{my_name()}: Human routed to unknown phase '{next_phase.value}'. Exiting."
                )
                break

        return self.state


    # ----------------------------------------------------------------------
    #                           EXECUTE_PHASE()
    # ----------------------------------------------------------------------

    def execute_phase(  self,
                        phase: Phase,
                        query: str,
                        working_team: List[Agent],
                        reviewing_team: List[Agent],
                        orchestrator: Agent,
                        artifacts: Optional[Any] = None,
                        tools: Optional[List[BaseTool]] = None,
                        max_iterations: int = MAX_ITERATIONS
                    ) -> PhaseExecutionResult:
        """
        Execute any project phase.  The hard‑coded flow is:

        1) GENERATE_RESPONSES (or on iteration >1, IMPROVE_BEST_RESPONSE)
        2) PEER_REVIEW
        3) ANALYZE_PEER_REVIEWS
        4) Loop back to step 1 until orchestrator.breaks() or max_iterations
        5) HUMAN_DECISION

        Returns:
            PhaseExecutionResult, a Pydantic model with three fields:

            • analysis: PeerReviewAnalysis
                - response: str                 # the final winning response
                - winner_name: str              # which agent won
                - avg_score: int                # average peer‑review score
                - improvement_points: List[str] # actionable suggestions
                - scores_table: Dict[str,int]   # all agents’ avg scores

            • decision: NextSteps
                - next_phase: Phase             # human’s routing choice
                - feedback: str                 # human’s freeform remarks
                - result                        # SUCCESS - move on. REPEAT - same phase with feedback, 
                                                # GO_TO - move back to a specified phase, STOP - exit program    

            • metadata: PhaseMetadata
                - query: str                    # the original query
                - team_description: str         # text describing the team setup   
                - steps_run: List[Step]         # e.g. [GENERATE_RESPONSES, PEER_REVIEW, …, HUMAN_DECISION]
                - iterations: int               # how many loops we performed
                - phase_duration: float         # wall‑clock seconds for this phase
                - phase_costs: Dict[str,float]  # optional cost breakdown per agent
                - phase_comments: Optional[str] # reserved for any extra notes
            """
        t0 = time.time()
        logging.info(f"\n{my_name()} Phase {phase.value} started.")

        steps_run: List[Step] = []
        final_analysis: PeerReviewAnalysis = None   # type: ignore
        human_decision: NextSteps = None            # type: ignore

        for i in range(1, max_iterations):
            logging.info(f"\n{my_name()} Phase {phase.value}: Iteration {i}")

            # --------------  GENERATE or IMPROVE  --------------
            if i == 1:
                steps_run.append(Step.GENERATE_RESPONSES)
                all_responses = asyncio.run(
                    self._generate_responses(
                        query               = query, 
                        working_team        = working_team, 
                        artifacts           = artifacts, 
                        tools               = tools
                    )
                )
            else:
                steps_run.append(Step.IMPROVE_BEST_RESPONSE)
                all_responses = asyncio.run(
                    self._generate_improvement(
                        query               = query,
                        working_team        = working_team,
                        response            = final_analysis.response,
                        improvement_points  = final_analysis.improvement_points,
                        artifacts           = artifacts,
                        tools               = tools
                    )
                )
            logging.info(f"{my_name()} Generated responses from: {list(all_responses.keys())}")

            # -------------- PEER-REVIEW --------------
            if reviewing_team:
                steps_run.append(Step.PEER_REVIEW)
                peer_reviews = asyncio.run(
                    self._peer_review(
                        query               = query, 
                        reviewing_team      = reviewing_team, 
                        responses           = all_responses)
                ) 
            else:
                peer_reviews = {}
            logging.info(f"{my_name()} Collected peer reviews from: {list(peer_reviews.keys())}")

            # --------------  ANALYZE_PEER_REVIEWS --------------

            steps_run.append(Step.ANALYZE_PEER_REVIEWS)
            final_analysis = self._analyze_peer_reviews(
                        query               = query, 
                        peer_reviews        = peer_reviews,
                        responses           = all_responses,
                        orchestrator        = orchestrator)
            
            if final_analysis is None:
                logging.error(f"{my_name()} no analysis for iteration {i}")
                break
                
            logging.info(f"{my_name()} Winner: {final_analysis.winner_name}")

            # --------------  BREAK LOOP? --------------
            if orchestrator and getattr(orchestrator, "decides_to_break", lambda *_: False)(final_analysis):
                logging.info(f"{my_name()} Orchestrator broke after iteration {i}")
                break

 
        duration = time.time() - t0
        # stub cost mapping (wire real usage if available)
        phase_costs: Dict[str, float] = {agent.name: 0.0 for agent in working_team}

        metadata = PhaseMetadata(
            query           = query, 
            steps_run       = steps_run, # ", ".join(step.name for step in steps_run),
            team_description= self._get_team_description(orchestrator, working_team, reviewing_team),
            iterations      = i,
            phase_duration  = duration,
            phase_costs     = phase_costs,
            phase_comments  = None
        )
        
        logging.info(f"\n{my_name()} Completed phase '{phase.value}'. Metadata: {metadata}")

        # --------------  HUMAN FEEDBACK --------------
        steps_run.append(Step.HUMAN_DECISION)
        phase_result = self.phase_review_with_human(
            final_analysis  = final_analysis, 
            metadata        = metadata, 
            orchestrator    = orchestrator
            )
        
        if phase_result is None:
            logging.error(f"\n{my_name()}: no human decision, trying to recover")
            next_steps = NextSteps(
                        result = Phase.STOP,
                        next_phase = self.current_phase,
                        feedback = "No human decision error"
                    )
            phase_result = PhaseExecutionResult(
                analysis    = final_analysis,
                decision    = next_steps,
                metadata    = metadata
            )

        else: 
            logging.info(f"{my_name()} Human decided: {phase_result.decision.next_phase.value}")

        return phase_result

    # ====================================================================
    #                        _GENERATE_RESPONSES()  
    # ====================================================================
    async def _generate_responses(self, 
                        query: str, 
                        working_team: List[Agent],
                        artifacts: Optional[str], 
                        tools: Optional[List[BaseTool]]) -> Dict[str, str]:
        """
            First-time response to the query. 
            Tools are currently not used. 
            Returns a dict {"agent_name": response_string}   
        """

        # !!! Tools and artifacts are NOT handled properly
        prompt = get_response_generation_prompt(
                    query               = query, 
                    artifacts           = artifacts_to_str(artifacts) if artifacts is not None else "", 
                    tools_description   = artifacts_to_str(tools) if tools is not None else ""
                )
        
        # Log the query details for debugging
        logging.info(f"{my_name()}: generating responses with the prompt {prompt}")

        # responses = asyncio.run(self._async_responses(working_team, prompt))
        responses = await self._async_responses(working_team, prompt)

        # Check if all agents provided a response
        missing_responses = [agent.name for agent in working_team if agent.name not in responses]
        if missing_responses:
            logging.warning(f"{my_name()}: Missing responses from agents: {missing_responses}")

        # we have a dictionary of {agent_name: AgentResponse}. Could validate it in future 
        logging.info(f"\n{my_name()}completed. prompt: {prompt} \n Responses: {responses} ")

        return responses

    #====================================================================
    #                   _GENERATE_IMPROVEMENT()  
    #====================================================================    
    async def _generate_improvement(self, 
            query: str, 
            working_team: List[Agent],
            response: str, 
            improvement_points:list[str], 
            artifacts: Optional[Any] = None, 
            tools: Optional[List[BaseTool]] = None) -> Dict[str, str]:
        """
            Improving the best response from previous iteration. 
            Tools are currently not used. 
            Returns a dict {"agent_name": response_str}   
        """
        # Log the query details for debugging
        logging.info(f"{my_name()}: starting")

        # !!! Tools and artifacts are NOT handled properly
        prompt = get_improvement_prompt(
                    query               = query, 
                    response            = response, 
                    improvement_points  = improvement_points, 
                    artifacts           = artifacts_to_str(artifacts) if artifacts is not None else "", 
                    tools_description   = artifacts_to_str(tools) if tools is not None else ""
                )

        responses = await self._async_responses(working_team, prompt)

        # we have a dictionary of {agent_name: agent_response}. Could validate it in future 
        logging.info(f"\n{my_name()} completed. prompt: {prompt} \n Responses: {responses} ")

        return responses

    #====================================================================
    #                           _PEER_REVIEW()  
    #====================================================================
    async def _peer_review(self, 
        query: str, 
        reviewing_team: List[Agent],
        responses: dict[str, str])-> Optional[Dict[str, Dict[str, ReviewItem]]]:  

        """
        Performs peer review asynchronously using the reviewing team.
        Each reviewer provides assessments for multiple responses. The results
        are parsed, validated, and aggregated into a dict [similar to AllPeerReviews model].
        Args:
            query: The original query context.
            reviewing_team: List of NamedAgent objects performing the review.
            responses: Dictionary mapping agent names to their response content
                       which needs to be reviewed.
        Returns:
            A dict object containing the structured reviews,
            mapping reviewer name to the list of reviews they provided or 
            None if no valid reviews are parsed.
            Othersise returns a dict: Dict[str, Dict[str, ReviewItem]], where:
            - outer 'str' is the reviewing agent; 
            - inner 'str' is the reviewed agent's name:  
            {   { 
                    'reviewing_agent_A': 
                    { 'reviewed_agent_1': {'score': int, 'improvement_points': List[str]}...},
                    { 'reviewed_agent_2': {'score': int, 'improvement_points': List[str]}...}, ....
                },
                { 
                    'reviewing_agent_B': 
                    { 'reviewed_agent_1': {'score': int, 'improvement_points': List[str]}...},
                    { 'reviewed_agent_2': {'score': int, 'improvement_points': List[str]}...}, ....
            }   }
            Skipps self-reviews if global flag SELF_REVIEWS_ALLOWED is set to False.
        """

        logging.info(f"{my_name()}: Starting peer review for query '{query[:50]}...'")
        if not reviewing_team:
             logging.warning(f"{my_name()}: Peer review skipped, no reviewing team provided.")
             return None  

        # tried serializing before passing responses. Does not help :) 
        # peer_review_prompt = get_peer_review_prompt(query=query, 
        # responses_str = to_string(responses, strong_formatting = True))
        peer_review_prompt = get_peer_review_prompt(query = query, responses = responses)
        logging.info(f"{my_name()}: Peer review prompt: {peer_review_prompt}")

        try:
            reviews = await self._async_responses(
                    reviewing_team,  
                    peer_review_prompt 
                )
        except Exception as e:
            logging.error(f"{my_name()}: Error in _async_responses: {e}")
            return None
                
        # note that in _async_responses() the reviewer agent name is 
        # already stitched to the response: Dict[str(reviewer_name), str(review_output)]
        # {
        #   reviewer_agent_name: 
        #       {reviewed_agent_1: {'score' : int, 'improvement_points': list[str]}} }, 
        #       {reviewed_agent_2: {'score' : int, 'improvement_points': list[str]}} },
        # } 
        parsed_reviews : Dict[str, Dict[str, ReviewItem]] = {}  # same as AllPeerReviews 
        
        for reviewer_name, review_str in reviews.items():          
            
            # sanity check: 
            if not isinstance(review_str, str) or not review_str or review_str == "[Agent Failed]":
                logging.warning(f"{my_name()}: Skipping invalid/empty/failed review output from {reviewer_name}.")
                continue

            # this returns the inner dict { reviewed_agent: { score:…, improvement_points:[…] } }
            logging.info(f"{my_name()}: Parsing review for {reviewer_name}: {review_str}...")
            parsed_peer_review = dict_from_str(review_str, PeerReviewItem)
            
            if parsed_peer_review is None:
                logging.error(f"{my_name()}: Failed to parse review for {reviewer_name}.")
                continue

            stitched_reviews = parsed_peer_review

            # optionally filter out self‑reviews
            if not SELF_REVIEWS_ALLOWED and reviewer_name in stitched_reviews:
                stitched_reviews.pop(reviewer_name, None)

            parsed_reviews[reviewer_name] = stitched_reviews
        
        if not parsed_reviews:
            logging.warning(f"{my_name()}: No valid peer reviews parsed.")
        else:
            logging.info(f"{my_name()}: Parsed reviews from {list(parsed_reviews.keys())}")

        logging.info(f"\n{my_name()}: Parsed peer reviews: \n {parsed_reviews}")

        return parsed_reviews
        

    # --------------------------------------------------------------------
    #                       _ANALYZE_PEER_REVIEWS()  
    # --------------------------------------------------------------------
    def _analyze_peer_reviews(self, 
                    query: str,      
                    peer_reviews: AllPeerReviews, 
                    responses: AllResponses,
                    orchestrator: Optional[Agent]=None )-> Optional[PeerReviewAnalysis]:
        """
            Accepts AllPeerReviews object: 
                {reviewing_agent: {'reviewed_agent': {improvement_points, score}}
            Calculates the averages and finds the winner. 
            Returns: PeerReviewAnalysis or None if error          
        """
        logging.info(f"{my_name()} starting")

        score_table: dict[str : List[int]] = {}               # agent_name : list[received_scores] 
        improvement_points_table: dict[str, List[str]] = {}   # agent_name : list[improvement_points]
        avg_scores: dict[str: int] = {}                       # agent_name : avg_received_score

        # --- Iterate through the dictionary structure ---
        for reviewer_name, review_data in peer_reviews.items():
            
            # 'review_data' is the inner dict like 
            # { reviewed_agent_name: {'score': ..., 'improvement_points': ...} }
            for reviewed_agent_name, inner_review_dict in review_data.items():
                
                # Ensure reviewed_agent_name exists in tables before appending
                if reviewed_agent_name not in score_table:
                    score_table[reviewed_agent_name] = []

                # --- Use dictionary key access ---
                score_value = inner_review_dict.get('score') 
                if score_value is not None and isinstance(score_value, int):
                    score_table[reviewed_agent_name].append(score_value)
                else:
                    logging.error(f"{my_name()}: Invalid or missing 'score' for {reviewed_agent_name} from {reviewer_name}")

                if reviewed_agent_name not in improvement_points_table:
                    improvement_points_table[reviewed_agent_name] = []

                # --- Use dictionary key access ---
                points_list = inner_review_dict.get('improvement_points', [])  
                if isinstance(points_list, list):
                    improvement_points_table[reviewed_agent_name].extend(points_list)
                else:
                     logging.error(f"{my_name()}: Invalid 'improvement_points' format for {reviewed_agent_name} from {reviewer_name}")
                # --- End dictionary key access ---
                    
        for name, scores in score_table.items():
            if scores: # Avoid division by zero
                avg_scores[name] = int(sum(scores) / len(scores))
            else:
                avg_scores[name] = 0 # Assign 0 if no scores
                logging.error(f"{my_name()}:scores are null")


        logging.info(f"\n{my_name()}: avg_scores: {avg_scores}")
        if not avg_scores:
            logging.error(f"{my_name()}: No average scores calculated, cannot determine a winner.")
            return None  

        # find the winner with the highest avg:  
        winner_name = max(avg_scores, key=avg_scores.get)
        winner_avg_score = avg_scores[winner_name]
        # fill in the winning response text
        winner_response = responses.get(winner_name, "")

        # harmonize improvement points list:
        improvement_points = self._harmonize_improvement_points(
            query = query, 
            improvement_points = improvement_points_table[winner_name],
            orchestrator = orchestrator
            )
        
        result = PeerReviewAnalysis(
            response = winner_response,
            winner_name = winner_name,
            avg_score = winner_avg_score,
            improvement_points = improvement_points,
            scores_table = avg_scores
        )

        logging.info(f"\n{my_name()} completed: {result}")

        return result 

    # --------------------------------------------------------------------
    #                    _HARMONIZE_IMPROVEMENT_POINTS()  
    # --------------------------------------------------------------------
    def _harmonize_improvement_points(self, 
                query: str, 
                improvement_points: List[str], 
                orchestrator: Optional [Agent] = None)-> List[str]:
        """
            An orchestrator LLM harmonize multiple improvement points. 
            Returns a single harmonized list of improvements from multiple reviews, to remove repetitions 
        """
        logging.info(f"{my_name()}: starting")
        if not orchestrator or not improvement_points: return improvement_points 

        prompt = get_harmonize_improvement_points_prompt(query = query, 
                                                         improvement_points = improvement_points)

        response = ""

        try:
            response = orchestrator.invoke(input=prompt)
            
            response_str = extract_llm_response(response)
            
            # Parse the string response into a list  
            parsed_response = dict_from_str(response_str)
            if parsed_response is None:
                logging.error(f"{my_name()}: Failed to parse improvement points: {response.content}")
                return improvement_points  # Fallback to original points
        
            return parsed_response
            
        except ValidationError as e:
            logging.error(f"{my_name()}: error: {e}")
            return improvement_points  # Fallback to original points        


    #--------------------------------------------------------------------
    #                       PHASE_REVIEW_WITH_HUMAN()   
    #--------------------------------------------------------------------
    def phase_review_with_human(self, 
        metadata: PhaseMetadata,                # metadata from the current phase, like execution time, etc  
        final_analysis: PeerReviewAnalysis,       # peer_review analysis of the current phase 
        orchestrator: Agent,                    # the one who is going to talk :) 
        ) -> PhaseExecutionResult:
        
        """
            Generic method to interact with the user. 
            Analyzes the query, questions the user if required, and gathers all necessary information 
            Returns: a parsed dict. The caller will deal with the structure!  
        """

        prompt = get_phase_review_with_human_prompt(
                                phase = self.current_phase, 
                                all_phases = self.execution_sequence,
                                query = query
                                # metadata = metadata, 
                                # final_analysis = final_analysis 
                            )

        # Create the agent with the system prompt and the tools. 
        # Requires memory & usage of tools (HITL_tool mandatory), so should be run by the Executor: 
        try:
            
            hitl = HITL_tool()

            agent = create_tool_calling_agent(
                orchestrator.runnable, tools = [hitl], prompt = prompt)
            
            executor = AgentExecutor(agent=agent, 
                                        tools= [hitl], verbose = True)
            
            logging.info(f"{my_name()}: Orchestrator executor created.")
            # logging.info(f"{my_name()}: prompt: {query}")

            # The Leader analyzes the query, asks the user for clarifications, 
            # and runs the tools if needed. The dialog with the user happens here: 
            # llm_response = executor.invoke({"input":""})
            
            llm_response = executor.invoke({ 
                "input": metadata.query,
                "phase": self.current_phase.value,
                "all_phases": ", ".join(phase.value for phase in self.execution_sequence),
            })


            logging.info(f"{my_name()}: Leader response: {llm_response}")
            
            # parse response; validate with Pydantic, but return a dict to keep it generic 
            response = dict_from_str(llm_response["output"], NextSteps)
            
            if response is None: 
                logging.error(f"\n{my_name()} Failed to parse response {llm_response["output"]}")
                return None 
            
            phase_result = PhaseExecutionResult(
                analysis    = final_analysis,
                decision    = response,
                metadata    = metadata
                )
            
            return phase_result

        except Exception as e:
            logging.error(f"{my_name()}: Error running the orchestrator: {e}")
            return None 


    #--------------------------------------------------------------------------------------------
    #                                   _ASYNC_RESPONSES()  
    #--------------------------------------------------------------------------------------------
    async def _async_responses(self, agents:List[Agent], prompt = str)-> Dict[str, Any]:
        """
            Helper function to run responses assynchronosly with LangChain. 
            Accepts a list of agents aka models and a generic prompt (one for all models)
            Returns a dict of:
            {  {'agent_name' : response_any},... }  # we assume that response is a string 
            Does NOT validate json response. 
        """
        # Initialize an empty dictionary to store model responses
        responses = {}
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=len(agents)) if USE_REAL_THREADS else None

        # Log the query details for debugging
        logging.info(f"{my_name()}: Starting async responses for {len(agents)} agents.")

        #--------------------------------------------------------------------------------------------
        # A helper function to call a model asynchronously and return its name and response.
        # Required to bind responses to model names 
        #--------------------------------------------------------------------------------------------
        async def _call_model(agent: Agent, retries: int = 3) -> Tuple[str, Any]:
            name = agent.name
            for attempt in range(1, retries + 1):
                start_time = time.time()
                logging.info(f"{my_name()}: Calling agent {name} (Attempt {attempt})")
                try:
                    if USE_REAL_THREADS:
                        response = await loop.run_in_executor(executor, lambda: agent.invoke(prompt))
                    else:
                        response = await asyncio.wait_for(agent.ainvoke(prompt), timeout=60)
                    execution_time = time.time() - start_time
                    logging.info(f"{my_name()}: Agent {name} responded in {execution_time:.2f} seconds")
                    return name, response
                
                except asyncio.TimeoutError:
                    logging.error(f"{my_name()}: Agent {name} timed out after 60 seconds (Attempt {attempt})")

                except Exception as e:
                    logging.error(f"{my_name()}: Error in agent {name} (Attempt {attempt}): {e}")

                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
            return name, None
            
        #------------------------ call_model ends ------------------------ 

        # Create a list of tasks, one for each agent model
        tasks = [asyncio.create_task(_call_model(agent)) for agent in agents]
        start_time = time.time()
        
        # Process tasks as they complete
        for finished_task in asyncio.as_completed(tasks):
            # Await the task to get the model name and response
            name, response = await finished_task
            
            # Check if we got a valid response
            if response is not None:
                # logging.info(f"{my_name()}: Received response from {name}. It took {time.time() - start_time} seconds.")
                responses[name] = extract_llm_response(response)  # safe method to extract content! 
            else:
                # Error already logged in _call_model
                responses[name] = "[Agent Failed]" # Store error placeholder string

        if USE_REAL_THREADS and executor:
           # Consider using try/finally for shutdown if exceptions can occur after executor creation but before shutdown
            try:
                executor.shutdown(wait=True)
            except Exception as e_shutdown:
                logging.error(f"{my_name()}: Error shutting down ThreadPoolExecutor: {e_shutdown}")
        
        total_processing_time = time.time() - start_time
        logging.info(f"\n{my_name()} completed in {total_processing_time:.2f} seconds with {len(responses)} responses.")

        # Return dictionary mapping agent name to extracted string content
        return responses
    

# ====================================================================
#                           __MAIN__ 
# ====================================================================
if __name__ == "__main__":

    query = QUERY_TYPES["CREATIVE_WRITING"]["test_query"]

    project = Lifecycle()
    project.current_phase = Phase.DESIGN

    # project.execute_lifecycle(query) 

    metadata = PhaseMetadata(
        query           = query, 
        steps_run       = [Step.GENERATE_RESPONSES, Step.PEER_REVIEW, Step.ANALYZE_PEER_REVIEWS], # Example steps
        team_description= "Test Team: Orchestrator, Worker A, Worker B",
        iterations      = 1,
        phase_duration  = 10.0,
        phase_costs     = {"worker_A": 0.01, "worker_B": 0.02}, # Example costs
        phase_comments  = "This is a test run of the human review phase."
    )



    final_analysis = PeerReviewAnalysis(
        response = "This is the winning response from the previous phase.",
        winner_name = "worker_A",
        avg_score = 88,
        improvement_points = ["Make it clearer.", "Add more detail.", "Check spelling."],
        scores_table = {"worker_A": 88, "worker_B": 75}
    )
        
    orchestrator_role_name = "scrum_master" # Or choose another role used as orchestrator
    orchestrator = project._get_agent(orchestrator_role_name)

    # 4. Call the method if the orchestrator was found
    if orchestrator:
        logging.info(f"Running phase_review_with_human using orchestrator: {orchestrator.name}")
        try:
            # Note: phase_review_with_human returns a PhaseExecutionResult
            phase_result = project.phase_review_with_human(
                metadata=metadata,
                final_analysis=final_analysis,
                orchestrator=orchestrator,
            )

            if phase_result:
                logging.info("--- phase_review_with_human completed ---")
                logging.info(f"Human Decision: {phase_result.decision.result}")
                logging.info(f"Next Phase: {phase_result.decision.next_phase.value}")
                logging.info(f"Feedback: {phase_result.decision.feedback}")
            else:
                logging.error("phase_review_with_human returned None (likely an error during execution).")

        except Exception as e:
            logging.error(f"An error occurred during phase_review_with_human: {e}", exc_info=True)
    else:
        logging.error(f"Could not find or initialize the orchestrator agent '{orchestrator_role_name}' needed for phase_review_with_human.")

    logging.info("--- Main Execution Finished ---")
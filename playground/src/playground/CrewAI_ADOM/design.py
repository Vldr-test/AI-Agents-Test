# ======================================================================
#           File design.py: high-level design in pseudo-code  
# ======================================================================

from config import (
    USE_REAL_THREADS, MAX_ITERATIONS, QUERY_TYPES, 
    AllPeerReviews,   
    AGENT_NAMES_TO_MODELS,
    my_name
)

from prompts import (
    get_response_generation_prompt, 
    get_improvement_prompt, # was get_iteration_prompt 
    get_peer_review_prompt, 
    get_review_improvement_points_prompt
)

from utils import dict_from_str, artifacts_to_str, extract_llm_response 

from schemas import (
            Step, Phase, 
            PhaseIterationLog, PhaseResults, NextSteps, ContextWrapper, 
            PeerReviewAnalysis, AgentResponse
            ) 

# from peer_review.leader import LeaderAgent

import time
from dotenv import load_dotenv
import logging
from langchain.chat_models import init_chat_model


# from analyze_query import LeaderAgent

import asyncio
if USE_REAL_THREADS: from concurrent.futures import ThreadPoolExecutor 
from langchain_core.language_models import BaseChatModel

from langchain.tools import BaseTool

from pydantic import BaseModel, ValidationError
import asyncio
from typing import Type, List, Dict, Optional, Tuple, Union, Any, Literal



from langchain_core.runnables import Runnable
from typing import List, Dict, Any, Optional, Tuple

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
        self.runnable = runnable
        logging.debug(f"NamedAgent created: {self.name} (Runnable Type: {runnable.__class__.__name__})")

    # --- Core Runnable Methods (Delegation) ---

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
    # define agents:
    product_manager: Agent 
    lead_product_owner: Agent
    product_owners: List[Agent]
    lead_architect: Agent 
    architects: List[Agent]
    lead_developer: Agent
    developers: List[Agent]
    state: ContextWrapper         # holds history / context 

    def __init__(self): 
        """ Initialize ContextWrapper and Agents """
        self.state = ContextWrapper()
        # TODO: Initialize all agent instances (self.product_manager = Agent(...), etc.) 
        pass


    # ====================================================================
    #                       INITIALIZE_AGENTS()  
    # ====================================================================
    
    def initialize_agents(self, agent_cfg: Dict[str, str]) -> List[Agent]
        """
            Initialize actual LLM instances with Langchain. 
            Args: { "agent_name" : "model_name", ...} 
            Return: list of agents
        """  
        agent_name: str = None
        agents: List[Agent] = []

        # create agents 
        for agent_name, model_name in agent_cfg.items():
            try:
                runnable = init_chat_model(model_name)
                agents.append(Agent(agent_name, runnable))
                logging.info(f"{my_name()}: Successfully created model {model_name} for agent: {agent_name}")
            except Exception as e:
                logging.error(f"{my_name()}: Can't create a model {model_name} for agent: {agent_name}: {e}")
                continue
        
        if agents is None:
            logging.error(f"{my_name()}: Can't create even one agent. Exiting")
            raise RuntimeError("Can't create even one agent. Exiting")
        
        return agents


    """
    # ----------------------------------------------------------------------
    #                    PROJECT_LIFECYCLE(): MAIN LOOP 
    # ----------------------------------------------------------------------
    def project_lifecycle(self, initial_artifacts: Any):

        sprint  = 0                                                 # we start from Sprint Zero
        self.state.initialize_sprint(sprint) # Use wrapper to init log for Sprint 0

        # Determine starting phase and artifact
        current_phase = Phase.DEFINE 
        current_artifact = initial_artifacts 
        last_phase_results = None # To hold results of the executed phase

        while True: # loop till all sprints are done. First sprint starts from reqirements; others can end up anywhere
        
            # ------------------ REQUIREMENTS PHASE ------------------
            if next_phase == Phase.DEFINE: 
                refined_spec, decision = self.execute_phase(
                        phase = Phase.DEFINE, 
                        artifacts = initial_artifacts,  
                        working_team = [self.product_manager], 
                        reviewing_team = [self.product_manager, self.product_owners, self.lead_architect], 
                        orchestrator = self.product_manager 
                    )

            # ---------------------- DESIGN PHASE ---------------------
            elif next_phase == Phase.DESIGN:
                high_level_design, decision = self.execute_phase(
                        phase = Phase.DESIGN, 
                        artifacts = refined_spec,  
                        working_team = [self.architects], 
                        reviewing_team = [self.architects, self.lead_product_owner, self.lead_architect], 
                        orchestrator = self.lead_architect 
                    )

            #--------------- BACKLOG REFINEMENT PHASE -----------------
            elif next_phase == Phase.REFINE_BACKLOG: 
                backlog, decision = self.execute_phase(
                        phase = Phase.DESIGN, 
                        artifacts = [refined_spec, high_level_design]
                        working_team = [self.product_owners], 
                        reviewing_team = [self.lead_product_owner, self.product_owners], 
                        orchestrator = self.lead_product_owner 
                    )
                
            #----------------- SPRINT PLANNING PHASE --------------------
            # although sprint is planned only at this point, technically, we start it from the beginning of the loop. 
            # So all phases incl sprint planning are a part of a spint 
            elif next_phase == Phase.PLAN_SPRINT: 
                sprint_plan, decision = self.execute_phase(
                        phase = Phase.DESIGN, 
                        artifacts = backlog
                        working_team = [self.product_owners], 
                        reviewing_team = [self.lead_product_owner, self.product_owners], 
                        orchestrator = self.lead_product_owner 
                    )
            
            # ----------------- CODE GENERATION PHASE -------------------
            # Combine refined_spec and hld_design as necessary.
            elif next_phase == Phase.CODE: 
                code, decision = self.improvement_loop(
                            phase = Phase.CODE, 
                            artifacts = [sprint_plan, high_level_design]  
                            working_team = [self.developers], 
                            reviewing_team = [self.architect, self.developers], 
                            orchestrator = self.scrum_master
                        )

            
            # ---------------- FUTURE WORKFLOW (TBD) ----------------
            # QA_TEAM: run_tests(), SECURITY: check_compliance(), DEVOPS_TEAM: deploy()
            # these should go BEFORE sprint review 
            else:
                pass # TBD 

            # ! we have to add a new phase to the log somehow

            # ---------------- END_OF_SPRINT MARKED  ----------------   
            if self.end_of_sprint():                  # who sets it up? Most probably at the phase_review?
                sprint_results = self.scrum_master.sprint_review_with_human()
                history.log_sprint_results   # log the decision at the entry to the new phase 
                sprint += 1
         
         
    """   

    # ======================================================================
    #                           EXECUTE_PHASE()  
    # ======================================================================
    def execute_phase(self, 
                    phase: Phase, 
                    query: str, 
                    working_team: List[Agent], 
                    reviewing_team: List[Agent], 
                    orchestrator: Agent, 
                    artifacts: Optional[Any] = None, 
                    max_iterations: int = MAX_ITERATIONS 
                ) -> Tuple[PhaseResults, NextSteps, List[PhaseIterationLog]]: 
        """
        """
        
        logging.info(f"\n{my_name()} Phase {phase.value} started.")
        
        # --- Data collection for logging ---
        iteration_logs: List[PhaseIterationLog] = []  
        input_artifacts = artifacts # Start with initial artifacts
        final_results = None # Store the results from the last successful iteration
        improvement_points = []
        peer_review_results: PeerReviewAnalysis

        artifacts_str = None
        # serialize artifacts: 
        if artifacts: 
            artifacts_str = artifacts_to_str(input_artifacts)   

        for iteration in range(1, max_iterations + 1):
            logging.info(f"{Phase.value}: --- Iteration {iteration} ---")
            iter_steps_run = []
            iter_peer_reviews = {}
            all_responses = {}
            
            # --- Generation/Improvement Step ---
            if iteration == 1:
                iter_steps_run.append(Step.GENERATE_RESPONSES)      # all agents generate their own responses 
                # TODO: Handle artifact dictionary if 'artifacts' is dict
                # We need a generic function, to be decoupled from frameworks (like CrewAI)
                all_responses = self._generate_responses(query, working_team, artifacts_str)  
            else:
                iter_steps_run.append(Step.IMPROVE_BEST_RESPONSE)   
                # Improve the 'best_response' from the *previous* iteration's results
                # all agents improve the best response from the previous iteration  
                all_responses = self._generate_improvement(query, working_team, 
                                peer_review_results.improvement_points, artifacts_str)  
            
            logging.info(f"{my_name()} Phase {phase.value} iteration {iteration} Generated responses from {list(all_responses.keys())}")


            # --- Peer Review Step ---
            if reviewing_team: # Only run if there are reviewers
                iter_steps_run.append(Step.PEER_REVIEW)
                # Pass the all output AND the original responses map for self-exclusion
                # TODO: Fix self-exclusion logic in agent.peer_review
                # agent.peer_review should accept (output_to_review, all_responses_map), 
                # This has to run async! 
                all_peer_reviews = self._peer_review(query, reviewing_team, all_responses)  
                logging.info(f"{my_name()} Phase {phase.value} iteration {iteration} Generated responses from {list(all_responses.keys())}")


            # --- Analysis Step ---
            iter_steps_run.append(Step.ANALYZE_PEER_REVIEWS)
            
            # --- orchestrator is optional, if the analysis is done by the function: 
            peer_review_results = self._analyze_peer_reviews(all_peer_reviews, orchestrator)  
            logging.info(f"{my_name()} Phase {phase.value} iteration {iteration}: best agent: {peer_review_results.best_agent}")


            # Update artifact for the *next* iteration
            input_artifacts = peer_review_results.best_response 
            final_results = peer_review_results # Keep track of the latest results
            final_results.iterations = iteration # Update final iteration count


            # --- Log Iteration Data (intermediate) ---
            # Note: Decision is logged *after* the loop finishes
            temp_iter_log = PhaseIterationLog(
                 steps_run=iter_steps_run.copy(), # Ensure copy
                 responses=all_responses.copy(),
                 peer_reviews=iter_peer_reviews.copy(),
                 results=peer_review_results.copy(), # Store result of *this* iteration
                 decision=None # Decision comes after the loop
            )
            # We don't store this intermediate log directly in the state here, 
            # but collect it to return at the end.

            # --- Orchestrator break condition ---
            if orchestrator.decides_to_break(peer_review_results):
                print(f"Orchestrator decided to break after iteration {iteration}.")
                break 
        
        # --- User Feedback Step (End of Phase Decision) ---
        if final_results is None:
             # Handle case where loop didn't run or failed early
             logging.error(f"{my_name()}: Phase {phase.value} iteration {iteration} Error: No phase results generated.")
             final_results = PhaseResults(best_agent="None", best_response=artifacts, iterations=0)
             # fail gracefully 
             human_decision = NextSteps(next_phase=Phase.STOP, feedback="Phase failed to produce results.")

        else:
            iter_steps_run.append(Step.HUMAN_DECISION) # Add this step marker
            human_decision = self.phase_review_with_human(final_results, orchestrator)
            logging.info(f"{my_name()} Phase {phase.value} iteration {iteration} Human decided: Go to {human_decision.next_phase.value}")
            
        # --- Finalize and Prepare Return ---
        # Add the final decision to the log entry for the *last* iteration run
        # Create the final log entry for the last iteration including the decision
        final_iter_log_entry = PhaseIterationLog(
            steps_run = iter_steps_run, # Steps from the last iteration
            responses = all_responses, # Responses from the last iteration
            peer_reviews = iter_peer_reviews, # Reviews from the last iteration
            results = final_results, # Final results object
            decision = human_decision # The decision made *after* the loop
        )

        iteration_logs.append(final_iter_log_entry) # Add the final log entry

        # ! shall we stitch the logs here? 
        # Return the final results, the human decision, and all iteration logs
        return final_results, human_decision, iteration_logs
    

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
        # tools are not handled properly now! 
        prompt = get_response_generation_prompt(query, artifacts, tools)
        
        # Log the query details for debugging
        logging.info(f"{my_name()}: generating responses for query {query}")

        # responses = asyncio.run(self._async_responses(working_team, prompt))
        responses = await self._async_responses(working_team, prompt)

        # Check if all agents provided a response
        missing_responses = [agent.name for agent in working_team if agent.name not in responses]
        if missing_responses:
            logging.warning(f"{my_name()}: Missing responses from agents: {missing_responses}")

        # we have a dictionary of {agent_name: AgentResponse}. Could validate it in future 
        logging.info(f"{my_name()}: Completed with responses from {list(responses.keys())}")

        return responses

    #====================================================================
    #                   _GENERATE_IMPROVEMENT()  
    #====================================================================    
    async def _generate_improvement(self, 
            query: str, 
            working_team: List[Agent],
            response: str, 
            improvement_points:list[str], 
            tools: Optional[List[BaseTool]]) -> Dict[str, str]:
        """
            Improving the best response from previous iteration. 
            Tools are currently not used. 
            Returns a dict {"agent_name": response_str}   
        """
        # Log the query details for debugging
        logging.info(f"{my_name()}: starting")

        prompt = get_improvement_prompt(query=query, response=response, 
                    improvement_points = improvement_points, tools=tools)

        # responses = asyncio.run(self.async_responses(working_team, prompt))
        responses = await self._async_responses(working_team, prompt)

        # we have a dictionary of {agent_name: agent_response}. Could validate it in future 
        logging.info(f"{my_name()}\n completed. Responses: {responses} ")

        return responses

    #====================================================================
    #                           _PEER_REVIEW()  
    #====================================================================
    def _peer_review(self, 
        query: str, 
        reviewing_team: List[Agent],
        responses: dict[str, str])-> AllPeerReviews:
        """
            Perform peer review and return a AllPeerReviews dict:
            { 
                { 'agent_name': {'score': int, 'improvement_points': List[str] }, 
                ...
                { 'agent_name': {'score': int, 'improvement_points': List[str] }
            }
        """
        logging.info(f"{my_name()} starting")

        peer_review_prompt = get_peer_review_prompt(query=query, responses = responses)

        try:
            reviews = asyncio.run( self._async_responses(
                    reviewing_team,  
                    peer_review_prompt 
                )
            )
            # logging.info(f"{my_name()}: Async peer review generated {len(responses)} results")
        except Exception as e:
            logging.error(f"{my_name()}: Error: {e}")
            raise e
        
        parsed_reviews = {}   # will store validated dictionaries
        
        # note that in async_generate_responses() the reviewer agent name is 
        # stitched to the response. "Reviews" is like: 
        # {
        #   reviewer_agent_name: 
        #       {reviewed_agent_name1: {'score' : int, 'improvement_points': list[str]}} }, 
        #       {reviewed_agent_name2: {'score' : int, 'improvement_points': list[str]}} },
        # } 
        for reviewer_name, review in reviews.items():
            # logging.info(f"{my_name()}: parsing reviewer_name: {reviewer_name}, review: {review}")
            parsed_review = dict_from_str(review, AllPeerReviews)
            
            if parsed_review is None:
                logging.error(f"{my_name()}: failed to parse JSON for {reviewer_name}.")
                continue # Skip to the next agent
            else:
                try:
                    parsed_reviews[reviewer_name] = parsed_review 
                    logging.info(f"{my_name()}: parsed_review for {reviewer_name}: {parsed_review}") 
                    # remove self-reviews if required: 
                    # ... 
                except Exception as e:
                    logging.error(f"{my_name()}: Error: {e}")
                    raise e

        if not parsed_reviews:
            logging.error(f"{my_name()}: No peer reviews available; cannot determine winner")
            raise RuntimeError("No peer reviews available") 
        else:
            return parsed_reviews 
        
    # --------------------------------------------------------------------
    #                    _REVIEW_IMPROVEMENT_POINTS()  
    # --------------------------------------------------------------------
    def _review_improvement_points(self, 
                query: str, 
                improvement_points: List[str], 
                orchestrator: Optional [Agent] = None)-> List[str]:
        """
            Harmonize multiple improvement points. 
            Returns a single harmonized list of improvements from multiple reviews, to remove repetitions 
        """
        logging.info(f"{my_name()}: starting")
        if not orchestrator: return improvement_points 

        prompt = get_review_improvement_points_prompt(query, improvement_points)

        response = ""

        try:
            response = orchestrator.invoke(input=prompt)
            # logging.info(f"{my_name()}: prompt {prompt}. \n response from orchestrator: {response.content}")
            
            # Parse the string response into a list  
            parsed_response = dict_from_str(response.content)
            if parsed_response is None:
                logging.error(f"{my_name()}: Failed to parse improvement points: {response.content}")
                return improvement_points  # Fallback to original points
        
            return parsed_response
            
        except ValidationError as e:
            logging.error(f"{my_name()}: error: {e}")
            return ""        

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

    query = QUERY_TYPES["SOFTWARE_PROGRAMMING"]["test_query"]

    project = Lifecycle()

    working_team_cfg = AGENT_NAMES_TO_MODELS["developers"]
    orchestrator_cfg = AGENT_NAMES_TO_MODELS["scrum_master"]

    working_team =  project.initialize_agents(working_team_cfg)  
    orchestrator = project.initialize_agents(orchestrator_cfg)  

    phase_results, next_steps, phase_iteration_log = project.execute_phase(
                    Phase.CODING, 
                    query, 
                    working_team = working_team,  
                    reviewing_team = working_team + [orchestrator], 
                    orchestrator = orchestrator 
                    # no artifacts yet 
                ) 
    
    logging.info(f"{my_name()}\n Execution results:")
    logging.info(f"{my_name()}\n phase_results: {phase_results}, \n next_steps: {next_steps}, \n phase_iteration_log: {phase_iteration_log}." )


"""

    "Sprint" : 0, "sprint_flow": 
        { 
        "Phase" : name, "phase_flow": 
            { 
                "iteration" : 0, "iteration_flow":
                    { 
                        "step: "GENERATE_RESPONSE" : step_flow: 
                            {
                                "duration": 12.33
                                "success" : bool 
                                ... 
                            }
                        "step: "PEER_REVIEW" : step_flow: 
                            {
                                "duration": 12.33
                                "success" : bool 
                                ... 
                            }
                        "step: "WHATEVER" : step_flow: 
                            {
                                "duration": 12.33
                                "success" : bool 
                                ... 
                            }
                        "iteration_results" : { TBD }
                    } 
            "phase_results": {TBD}
        }, 
    "sprint_results": {TBD} 
}



"""
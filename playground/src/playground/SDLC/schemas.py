
from pydantic import BaseModel, Field, RootModel 
from typing import Tuple, List, Dict, Any, Optional, Literal # Import standard types from typing
from enum import Enum  


"""
    ========================================================================================================
    TERMINONOLY: a full project 'lifecycle' consists of 'sprints'. Sprints consist of sequential 'phases'. 
    Each phase could have multiple iterations over 'steps'. At the end of each phase there is a decision, 
    which step to go next. Eg. if there are imlpications on design, we could go back to design at the end of 
    any phase in the lifecycle. Transitions are logged.  

    SPRINT LOGIC CLARIFICATION: a sprint ideally runs from the first phase to the last one. Sprint zero 
    always starts from the very beginning of the sequence: State.DEFINE. However, at the end of any phase, 
    if the next_phase is non-consequential, it means that the sprint breaks / fails. Then the new sprint is 
    started from whichever phase is mentioned as next_phase. 
    ========================================================================================================
"""

# ================================================================================
#                     LIFECYCLE: PHASES & STEPS WITHIN PHASES 
# ================================================================================

# ------------  LIFECYCLE PHASES  --------------
class Phase(str, Enum):
    DEFINE = "DEFINE"                               # leader / PM
    DESIGN = "DESIGN"                               # designers / archiects 
    REFINE_BACKLOG = "REFINE_BACKLOG"               # managers / POs 
    #---------------------------------              # STARTS THE NEXT SPRINT  
    PLAN_SPRINT = "PLAN_SPRINT"                     # managers / POs. 
    CODE = "CODE"                                   # workers / developers 
    TEST = "TEST"                                   # workers / testers 
    DEPLOY = "DEPLOY"                               # workers / devops 
    SPRINT_REVIEW = "SPRINT_REVIEW"                 # not sure it is required as a formal phase 
    STOP = "STOP"                                   # pseudo-state to break the outer loop 

# an ideal sequence. PHASE_REVIEW is NOT included by design 
PHASE_SEQUENCE = [Phase.DEFINE, Phase.DESIGN, Phase.REFINE_BACKLOG, 
                      Phase.PLAN_SPRINT, Phase.CODE, Phase.TEST, Phase.DEPLOY]

# -------- INNER STEPS INSIDE ANY PHASE  --------------
class Step(str, Enum): 
    GENERATE_RESPONSES = "gen_responses"            # workers: only first iteration 
    IMPROVE_BEST_RESPONSE = "gen_improvements"      # workers: second iteration onwards 
    STITCH_RESPONSES = "stitch_responses"           # orchestrator or function awaiting async exec per agent. Might be redudntant 
    PEER_REVIEW = "peer_review"                     # reviewers 
    ANALYZE_PEER_REVIEWS = "analyze_peer_reviews"   # orchestrator or function
    HUMAN_DECISION = "human_decision"               # orchestraror with HITL 

# an ideal sequence of steps in a phase:  
STEP_SEQUENCE = [Step.GENERATE_RESPONSES, Step.IMPROVE_BEST_RESPONSE, # Step.STITCH_RESPONSES, 
                  Step.PEER_REVIEW, Step.ANALYZE_PEER_REVIEWS, Step.HUMAN_DECISION]


# -------------------------------------------------------------------------------
#                              STEPS RESULTS  
# -------------------------------------------------------------------------------

# In case we do initial query analysis: 
class QueryAnalysis(BaseModel):
    query_type: str = Field(..., description="Type of the query")
    query_class: Literal['SIMPLE', 'COMPLEX']
    improved_query: str = Field(..., description="Enriched and improved version of the query")

# NOT USED, COULD BE DELETED 
class PromptForAgents(BaseModel):
    prompt_for_agents: str = Field(..., description="Prompt for agents")
    # added default factory for safety: 
    recommended_tools: List[str] = Field(default_factory=list, description="Names of tools recommended for agents to use for this prompt")


# -------------------------------------------------------------------------------
#                              AGENT RESPONSES  
# -------------------------------------------------------------------------------
class AllResponses(RootModel[Dict[str, Any]]):
    """
        Represents a collection of agent responses as a dictionary
        mapping agent_name (str) to response_content (Any).
        The object itself IS this dictionary.

        Example Access:
            responses_obj['agent_A'] -> response_content_A
            for agent_name, content in responses_obj.items(): ...
    """
    root: RootModel[Dict[str, Any]] = Field(..., description="Dict of all responses generated ")

    # --- Optional: Add convenience methods for dict-like behavior ---
    def __getitem__(self, key: str) -> Any: return self.root[key]
    def __iter__(self): return iter(self.root)
    def items(self): return self.root.items()
    def keys(self): return self.root.keys()
    def values(self): return self.root.values()
    def __len__(self): return len(self.root)
         

   
# -------------------------------------------------------------------------------
#                               PEER REVIEW  
# -------------------------------------------------------------------------------

class ReviewItem(BaseModel):
    """ The actual content/payload of a single review assessment. """
    # comments: str = Field(..., description="Narrative feedback from the reviewer regarding the reviewed response")
    score: int = Field(..., description="An integer score [1-100] assessing the content's quality", ge=1, le=100)
    improvement_points: List[str] = Field(default_factory=list, description="Specific, actionable suggestions for improvement")

class PeerReviewItem(RootModel[Dict[str, ReviewItem]]):
    """
    A single peer review item as returned by one agent and stitched with its name
    - str: reviewed_agent_name 
    """
    root: Dict[str, ReviewItem] = Field(..., description="Dict of one peer review.")
    # Optional convenience methods
    def __iter__(self): return iter(self.root)
    def __getitem__(self, item): return self.root[item]
    def items(self): return self.root.items()
    def keys(self): return self.root.keys()
    def values(self): return self.root.values()   
    
# {reviewed_agent: {"score": int, "improvement_points" : [improvement_point]} }
class AllPeerReviews(RootModel[Dict[str, PeerReviewItem]]):
    """
    Collection of all reviews from a peer review step.
    The object itself IS the dictionary mapping reviewer_name to reviews
    - first str: reviwing_agent_name
    - second str: reviewed_agent_name
    """
    root: Dict[str, PeerReviewItem] = Field(..., description="Dict of all peer reviews.")
    # Optional convenience methods
    def __iter__(self): return iter(self.root)
    def __getitem__(self, item): return self.root[item]
    def items(self): return self.root.items()
    def keys(self): return self.root.keys()
    def values(self): return self.root.values()

# -------------------------------------------------------------------------------
#                              PEER REVIEW ANALYSIS   
# -------------------------------------------------------------------------------  
class ScoreTable(BaseModel):
    agent: str = Field(..., description="Name of the agent")
    avg_score: int = Field(..., description = "Avg score of the winner agent, 1 to 100.", ge=1, le=100)

class PeerReviewAnalysis(BaseModel):
    response: Any = Field(..., description="Response from the winning agent")
    winner_name: str = Field(..., description="Name of the winner agent")
    avg_score: int = Field(..., description = "Avg score of the winner agent, 1 to 100.", ge=1, le=100)
    improvement_points: List[str] = Field(...,default_factory=list, description="Winner's agent improvement points")
    scores_table: Dict[str, int] = Field(
        ...,
        description="A dictionary mapping each reviewed agent's name to their calculated average score (1-100).",
        examples=[{"agent_writer": 85, "agent_researcher": 92}] # Example helps LLM
    )

# -------------------------------------------------------------------------------
#                         PHASE REVIEW WITH HUMAN   
# -------------------------------------------------------------------------------   

# --- END OF EVERY PHASE: HUMAN REVIEWS PHASE OUTCOMES AND DECIDES: 
class NextSteps(BaseModel):                         # human decision on what phase in the lifecycle to do next 
    """
        Result of discussing phase outcomes with the human. 
        Could be:
        - SUCCESS: continue the workflow, no extra comments. next_phase is set up automatically 
        - REPEAT: the phase has to be repeated, taking new user "feedback" into account 
        - GO_TO: the user has decided to go (or to return) to a particular case 
        - STOP: stop program execution completely, all phases abandoned 

    """
    next_phase: Optional[Phase] = Field(None, description="Which phase to execute next (can be the same phase)")
    feedback: Optional[str] = Field(None, description="Human feedback accompanying the decision") 
    result: Literal["SUCCESS","REPEAT","GO_TO","STOP"] = Field(..., 
                description="move to the next phase if success, repeat the same phase, go back to a specific phase, or stop ")
 

# ================================================================================
#                      STATE / HISTORY / CONTEXT 
# ================================================================================


# --- reporting the metadata for each phase: --- 
class PhaseMetadata(BaseModel):
    """Auxiliary data about how this phase ran."""
    query: str = Field(
        ..., description="Initial query"
    )
    steps_run: List[Step] = Field(
        ..., description="Ordered list of the Step enums actually executed"
    )
    iterations: int = Field(
        ..., ge=1, description="Number of generate→review loops completed"
    )
    phase_duration: Optional[float] = Field(
        None,
        description="Total time spent in this phase (seconds)."
    )
    phase_costs: Optional[Dict[str, float]] = Field(
        None,
        description="Estimated cost per agent/LLM call, keyed by agent name."
    )
    team_description: Optional[str] = Field(
        None,
        description="A string describing the agents / llms working on the phase."
    )
    phase_comments: Optional[str] = Field(
        None,
        description="Any free‑form notes or warnings captured during execution."
    )

# --- reporting the complete result for each phase execution: --- 
class PhaseExecutionResult(BaseModel):
    """Encapsulates everything returned by execute_phase()."""
    analysis: PeerReviewAnalysis = Field(
        ..., description="Final peer‑review analysis result for this phase"
    )
    decision: NextSteps = Field(
        ..., description="Human’s routing decision & feedback at end of phase"
    )
    metadata: PhaseMetadata = Field(
        ..., description="Meta‑information about how the phase ran"
    )

 
# -------------------------------------------------------------------------------
#                           CONTEXT WRAPPER 
# -------------------------------------------------------------------------------
class ContextWrapper():
    context: Any

    def __init__(self):
        # self.context = LifecycleLog()
        pass 

    def initialize_sprint(self, sprint:int):
        pass 

    def save_phase_results(self, Any):
        pass
        

    """   
    def save_iteration_results(): 
        pass

    def save_phase_results():
        pass

    def save_sprint_results():
        pass 

    def save_next_phase():  # do we need it? 
        pass

    # --- access functions: --- 
    def get_last_iteration_results():
        pass
    def get_last_phase_results():
        pass
    def get_last_sprint_results():
        pass
    def get_lifecycle_context(): 
        pass
    """

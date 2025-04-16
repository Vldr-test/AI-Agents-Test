
from pydantic import BaseModel, Enum, Tuple, List, Dict, Any, Optional, Field, Literal, RootModel 


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
    DEFINE = "define"                               # leader / PM
    DESIGN = "design"                               # designers / archiects 
    REFINE_BACKLOG = "refine_backlog"               # managers / POs 
    #---------------------------------              # STARTS THE NEXT SPRINT  
    PLAN_SPRINT = "plan_sprint"                     # managers / POs. 
    CODE = "code"                                   # workers / developers 
    TEST = "test"                                   # workers / testers 
    DEPLOY = "deploy"                               # workers / devops 
    SPRINT_REVIEW = "sprint_review"                 # not sure it is required as a formal phase 
    STOP = "stop"                                   # pseudo-state to break the outer loop 

# an ideal sequence. PHASE_REVIEW is NOT included by design 
PHASE_SEQUENCE = [Phase.DEFINE, Phase.DESIGN, Phase.REFINE_BACKLOG, 
                      Phase.PLAN_SPRINT, Phase.CODE, Phase.TEST, Phase.DEPLOY]

# -------- INNER STEPS INSIDE ANY PHASE  --------------
class Step(str, Enum): 
    GENERATE_RESPONSES = "gen_responses"            # workers: only first iteration 
    IMPROVE_BEST_RESPONSE = "gen_improvements"      # workers: second iteration onwards 
    STITCH_RESPONSES = "stitch_responses"           # orchestrator or function awaiting async exec per agent. Might be redudntant 
    PEER_REVIEW = "peer_review"                     # reviewers 
    # STITCH_PEER_REVIEWS = "stitch_peer_reviews"     # orchestrator or function awaiting async exec per agent
    ANALYZE_PEER_REVIEWS = "analyze_peer_reviews"   # orchestrator or function
    HUMAN_DECISION = "human_decision"               # orchestraror with HITL 

# an ideal sequence of steps in a phase:  
STEP_SEQUENCE = [Step.GENERATE_RESPONSES, Step.IMPROVE_BEST_RESPONSE, # Step.STITCH_RESPONSES, 
                  Step.PEER_REVIEW, Step.ANALYZE_PEER_REVIEWS, Step.HUMAN_DECISION]

# ------------  PHASE OUTCOMES --------------
class PhaseResults(BaseModel):
    best_agent: Optional[str] = Field(None, description="Name of the winning agent, if applicable")                                 # name of the winning agent 
    best_response: Any = Field(..., description="The final artifact content (dict, str, object, file path, etc.)")
    iterations: int = Field(..., description="Number of iterations actually run in the phase")

# --- END OF EVERY PHASE: HUMAN REVIEWS PHASE OUTCOMES AND DECIDES: 
class NextSteps(BaseModel):                         # human decision on what phase in the lifecycle to do next 
    next_phase: Phase = Field(..., description="Which phase to execute next (can be the same phase)")
    feedback: Optional[str] = Field(None, description="Human feedback accompanying the decision") 

# --- END OF EVERY SPRINT: A HUMAN REVIEWS SPRINT OUTCOMES AND DECIDES: 
class SprintReview(BaseModel):
    results: PhaseResults = Field(..., description="Results from the last completed phase of the sprint")
    success: bool = Field(..., description="Whether the sprint goals were considered met")
    decision: NextSteps = Field(..., description="The decision on where to go after this sprint review")
    # Added default_factory=list for safety.
    improvement_points: List[str] = Field(default_factory=list, description="List of improvement points noted during the review")
    

# -------------------------------------------------------------------------------
#                              STEPS RESULTS  
# -------------------------------------------------------------------------------

# In case we do initial query analysis: 
class QueryAnalysis(BaseModel):
    query_type: str = Field(..., description="Type of the query")
    query_class: Literal['SIMPLE', 'COMPLEX']
    improved_query: str = Field(..., description="Enriched and improved version of the query")

# returned by 
class PromptForAgents(BaseModel):
    prompt_for_agents: str = Field(..., description="Prompt for agents")
    # added default factory for safety: 
    recommended_tools: List[str] = Field(default_factory=list, description="Names of tools recommended for agents to use for this prompt")


# -------------------------------------------------------------------------------
#                              AGENT RESPONSE  
# -------------------------------------------------------------------------------

# ATTENTION: this is not used anywhere :( 
class AgentResponse(BaseModel):
    """ Contains the response generated by a single agent during a step. """
    agent_name: str = Field(..., description="Name of the agent providing the response")
    # Using Any allows flexibility for different response types (text, code, structured data)
    response_content: Any = Field(..., description="The actual response content produced by the agent")


class AllResponses(RootModel[List[AgentResponse]]):
    """ 
    A collection of responses from multiple agents for a single step/iteration.
    The object itself IS the list of AgentResponse objects.
    Example Access: all_responses_obj[0] -> AgentResponse(...)
                    for response in all_responses_obj: -> response is AgentResponse(...)
    """
    root: List[AgentResponse] = Field(..., description="List of all responses generated ")

    # --- Optional: Add convenience methods for list-like behavior ---
    def __iter__(self):  return iter(self.root)                     # Allows iterating directly over the responses.       
    def __getitem__(self, item): return self.root[item]             # Allows accessing responses by index.  
    def append(self, item: AgentResponse): self.root.append(item)   #  Allows appending a new AgentResponse.     
    def __len__(self): return len(self.root)                        # Allows getting the number of responses using len().  
         

# -------------------------------------------------------------------------------
#                               PEER REVIEW  
# -------------------------------------------------------------------------------

class ReviewContent(BaseModel):
    """ The actual content/payload of a single review assessment. """
    text: str = Field(..., description="Narrative feedback from the reviewer regarding the reviewed response")
    score: int = Field(..., description="An integer score [1-100] assessing the content's quality", ge=1, le=100)
    improvement_points: List[str] = Field(default_factory=list, description="Specific, actionable suggestions for improvement")

class PeerReviewItem(BaseModel):
    """
    Represents one agent's assessment (review) of another agent's work.
    Stored typically under the reviewer's context.
    """
    reviewed_agent: str = Field(..., description="Name of the agent whose work is being reviewed in this item")
    review: ReviewContent = Field(..., description="The review content (text, score, improvements)")

# {reviewed_agent: {"score": int, "improvement_points" : [improvement_point]} }
class AllPeerReviews(RootModel[Dict[str, PeerReviewItem]]):
    """
    Collection of all reviews from a peer review step.
    The object itself IS the dictionary mapping reviewer_name -> List[PeerReviewItem].
    """
    root: Dict[str, List[PeerReviewItem]] = Field(..., description="Dict of all peer reviews.")
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
    
 
# ================================================================================
#                      STATE / HISTORY / CONTEXT 
# ================================================================================

# -------- PhaseIteration: a single iteration of one phase during a sprint -------
class PhaseIterationLog(BaseModel):
    steps_run: List[Step] = Field(..., description="Steps executed in this iteration (ordered)")
    responses: AllResponses = Field(..., description="Agent outputs collected in this iteration")
    peer_reviews: Optional[AllPeerReviews] = Field(None, description="Peer reviews collected in this iteration")
    results: PhaseResults = Field(..., description="Outcome (best agent/response) determined *at the end* of this iteration")
    decision: Optional[NextSteps] = Field(None, description="HITL decision made AFTER this iteration (usually only on the last log)")

# -------- SprintPhaseLog: all iterations for a given phase in a sprint ----------
class PhaseLog(BaseModel):
    iterations: Dict[int, PhaseIterationLog]        # phase_iteration: PhaseIteration

# -------- SprintLog: full lifecycle for a sprint, across all phases -------------
class SprintLog(BaseModel):
    phases: Dict[Phase, PhaseLog]                   # {Phase.CODE: PhaseLog, ...}
    summary: Optional[SprintReview]                 # Final decision from last phase in the sprint

# -------- LifecycleLog: full history of all sprints and transitions ---------
class LifecycleLog(BaseModel):
    history: Dict[int, SprintLog]                   # sprint_number: SprintLog
    routing_trace: Optional[List[Dict[str, str]]]   # [{from: "design", to: "plan_sprint", reason: "..."}]
    global_notes: Dict[str, Any]                    # e.g. {"started": "2025-04-16T12:00:00Z", "owner": "ADOM"}

# -------------------------------------------------------------------------------
#                           CONTEXT WRAPPER 
# -------------------------------------------------------------------------------
class ContextWrapper():
    context: LifecycleLog

    def __init__(self):
        self.context = LifecycleLog()

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



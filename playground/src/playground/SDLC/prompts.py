# prompts.py

from config import my_name
from schemas import Phase, PeerReviewAnalysis, PhaseMetadata
import logging
from typing import List, Optional
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage

# ---------------------------------------------------------------------------
#                        GET_REFINE_QUERY_PROMPT()
# ---------------------------------------------------------------------------
def get_refine_query_prompt(
    tools_description_str: str,
    query_types_str: str
) -> ChatPromptTemplate:
    """
    Build a prompt that:
      1) classifies the user’s raw query into one of query_types_str,
      2) labels it SIMPLE or COMPLEX,
      3) (optionally) uses HITL to clarify,
      4) uses any of the listed tools,
      5) returns an improved JSON with exactly:
         { "query_type": str,
           "query_class": "SIMPLE"|"COMPLEX",
           "improved_query": str }
    """
    system = SystemMessage(
        content=(
            "You are an expert in analyzing user queries and running tools for context gathering.\n"
            "Use HITL_tool only if you need clarification, otherwise do not call it."
        )
    )
    human = HumanMessage(
        content=f"""Your task is to collect all relevant information from the user’s query ***{{input}}***.
Steps:
  1. Choose exactly one query_type from: {query_types_str}
  2. Set query_class to SIMPLE or COMPLEX.
  3. Call HITL_tool if you need any disambiguation from the user.
  4. Call any of these tools as needed: {tools_description_str}
  5. Produce an improved version of the original query.

Return a JSON _only_ with three fields:
  • "query_type": string from the list
  • "query_class": "SIMPLE" or "COMPLEX"
  • "improved_query": the improved query string

Do NOT output anything else."""
    )
    return ChatPromptTemplate.from_messages([
        system,
        human,
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

# ---------------------------------------------------------------------------
#               GET_CONSTRUCT_PROMPT_WITH_TOOLS()
# ---------------------------------------------------------------------------
def get_construct_prompt_with_tools(
    improved_query: str,
    tools_description_str: str
) -> str:
    """
    Given an improved_query, decide which tools the AI agents should use and
    wrap them into a new prompt. Return a raw string (to be passed directly
    into LLM.invoke).
    """
    return (
        f"Your task is to construct the best detailed prompt for AI agents "
        f"based on this improved query:\n\n   {improved_query}\n\n"
        f"Also recommend and list any of these tools they should invoke:\n"
        f"{tools_description_str}\n\n"
        "Return JSON _only_:\n"
        '  { "prompt_for_agents": <string>, "recommended_tools": [<tool names>] }\n'
        "No extra text."
    )

# ---------------------------------------------------------------------------
#                  GET_RESPONSE_GENERATION_PROMPT()
# ---------------------------------------------------------------------------
def get_response_generation_prompt(
    query: str,
    artifacts: Optional[str] = "",
    tools_description: str = ""
    ) -> List[BaseMessage]:
    """
    Build a message sequence (List[BaseMessage]) asking an AI agent to respond
    to `query`. You may prepend `artifacts` or `tools_description_str` into the
    system message if provided.
    """
    system_content = (
        "You are a response generator. Produce a direct answer to the user’s query."
        "Match the query’s language (e.g., Russian query = Russian response) unless told otherwise. "
        "Return a plain string — no Unicode escapes (use 'В' not '\\u0412'). "
        "Do NOT include markers or extra text — just the response."
    )

    human_content = (
        "Respond to this query: ***{query}***"
    )

    if artifacts:
        system_content += f"\n\nContext:\n{artifacts}"
    if tools_description:
        system_content += f"\n\nTools available:\n{tools_description}"

    template = ChatPromptTemplate.from_messages([
        ( "system", system_content ),
        ( "human", human_content )
    ])
    return template.format_messages(query=query, 
                                artifacts=artifacts, tools_description = tools_description)

# ---------------------------------------------------------------------------
#                        GET_PEER_REVIEW_PROMPT()
# ---------------------------------------------------------------------------
def get_peer_review_prompt(
    query: str,
    responses: any
    ) -> List[BaseMessage]:
    """
    Build a message sequence asking a reviewer to score & critique each
    response. `responses_str` must be a pre-serialized string, for example:
        "dev1: <answer1>\n\ndev2: <answer2>\n\n…"
    Returns a List[BaseMessage].
    """
    logging.info(f"\n{my_name()}: query {query}, responses_str {responses}")

    system_content = (
        "You are a reviewer, evaluating quality of each response fairly but critically."
    )
    
    human_content = (
        "Review these responses to the query ***{query}***:\n"
        "***{responses}***. For each response, produce a JSON object with: \n"
        "- 'improvement_points': a list of 2-5 clear and actionable suggestions. \n"
        "- 'score': an integer with your assessment of the responses' quality, " 
        "from 1 (lowest) to 100 (highest). Be fair but harsh with your assesments. \n"
        "Return a JSON object where each key is an actual agent's name (which you take from the response), "
        "and each value is an object with 'improvement_points' and 'score' keys. \n"
        # "Only use currency symbols inside quoted strings."
        "Do NOT include any extra text — just the JSON."
    )

    template = ChatPromptTemplate.from_messages([
        ( "system", system_content ),
        ( "human", human_content )
    ])

    return template.format_messages(query = query, responses = responses)

# ---------------------------------------------------------------------------
#                       GET_IMPROVEMENT_PROMPT()
# ---------------------------------------------------------------------------
def get_improvement_prompt(
    query: str,
    response: str,
    improvement_points: List[str],
    artifacts: Optional[str] = "",
    tools_description: str = ""
    ) -> List[BaseMessage]:
    """
    Build a message sequence asking an AI agent to refine `response` to
    `query`, addressing each bullet in `improvement_points`. If `tools`
    is given, append a tools list. Returns List[BaseMessage].
    """
    system_content = (
        "You are an expert editor. Improve the given response incrementally."
    )
    human_content = (
            "Refine this response ***{response}*** to the query ***{query}*** :\n"
            "Address these  improvement points:\n"
            "***{improvement_points}***\n\n"
            "You MUST:\n"
            "- Reuse the original response as the base, modifying it to incorporate each improvement point.\n"
            "- Ensure the new response is noticeably different and improved, not a copy.\n"
            "- Verify before returning that all improvement points are addressed and the result is better.\n"
            "- Keep what’s already strong (like clarity or key points) and only tweak what requires improvement."
            "Return only the improved response as a plain string — no Unicode escapes (use 'В' not '\\u0412'), no extra text."
    )

    if artifacts:
        system_content += f"\n\nContext:\n{artifacts}"
    if tools_description:
        system_content += f"\n\nTools available:\n{tools_description}"

    template = ChatPromptTemplate.from_messages([
        ( "system", system_content ),
        ( "human", human_content )
    ])

    return template.format_messages(query=query, response=response, 
                                improvement_points = improvement_points, 
                                artifacts = artifacts, 
                                tools_description = tools_description)

# ---------------------------------------------------------------------------
#             GET_HARMONIZE_IMPROVEMENT_POINTS_PROMPT()
# ---------------------------------------------------------------------------
def get_harmonize_improvement_points_prompt(
    query: str,
    improvement_points: List[str]
    )  -> List[BaseMessage]:
    """
    Shorten and de-duplicate a list of improvement points. Return a JSON list
    (as a Python string) 
    """
    system_content = (
        "You are an experienced editor that refines lists of improvement points."
    )
    
    human_content = (
        "Analize and rationalize this list of improvement points: ***{improvement_points}***. "
        "To give you some context, these list is based on the response to the following query: ***{query}***. "
        "Combine similar improvement points into one. "
        "Make sure that all your improvement suggestions are clear and actionable. Do not miss anything important. " 
        "Return a plain JSON list of refined improvement suggestions. "
        "Do NOT include any extra text; return ONLY the JSON list — no Unicode escapes (use 'В' not '\\u0412'). "
        )
    
    template = ChatPromptTemplate.from_messages([
        ( "system", system_content ),
        ( "human", human_content )
    ])
    return template.format_messages(query=query, improvement_points = improvement_points)

# ---------------------------------------------------------------------------
#                    GET_PHASE_REVIEW_WITH_HUMAN_PROMPT()
# ---------------------------------------------------------------------------
def get_phase_review_with_human_prompt(
        query: str, 
        phase: Phase, 
        all_phases: List[Phase], 
    #    final_analysis: PeerReviewAnalysis,
    #    metadata: PhaseMetadata
    ) -> ChatPromptTemplate:
    """
        Input parameters: 
            - {phase}:  current project phase  
            - {all_phases}:  all phases of the project (execution path)
        
        # --- PeerReviewAnalysis --- 
            - {response}: Response from the winning agent
            - {winner_name}: name of the winner agent
            - {avg_score}: avg score of the winner agent, 1 to 100.", ge=1, le=100
            - {improvement_points}: list of winner's  improvement points 
            - {scores_table}: Dict[str, int]: dictionary mapping each reviewed agent's name to their calculated average score (1-100)

        # --- PhaseMetadata --- 
            - {query}: initial query"
            - {steps_run}: List[Step] = ordered list of the Step enums actually executed in this phase 
            - {iterations}: number of generate → review loops completed
            - {phase_duration}: total time spent in this phase (seconds)
            - {phase_costs}: estimated cost per agent/LLM call, TBD (stubbed for now)
            - {team_description}: a string (!) describing the agents / llms working on the phase
            - {phase_comments}: optional[str] = any free‑form notes or warnings captured during execution.

        Returns: see the prompt below. 
        
        ATTENTION! This prompt is required for the LangChain Executor. So it has to return 
        ChatPromptTemplate, rather than a list of BaseLineMessages. 
    """
    system_content = (
            "You are an expert in human interaction, using a HITL tool .\n"
            "You talk to the user until you are clear about what they want to do next. "
            "Present the result of the previous phase (such as the query) only __once__"
            "Return a JSON with fields: result, next_phase, feedback."
    )

    human_content = (
            "A team of agents is working on your initial query ***{query}***. Phase ***{phase}*** of the project lifecycle (consisting of phases: ***{all_phases}***) has just been completed."
        #    "The results of this phase are: ***{final_analysis}***. This is additional info about the phase: \n"
        #    "***{metadata}***.\n"
            "Your task is to present these results to the human and get their feedback and decision on the next step.\n\n"
            "1. Present the phase results, acknowledging potential improvements. You can also remind about the process\n"
            
            "2. Ask the human what they want to do next:\n"
            "   - If satisfied, mark 'result' as 'SUCCESS' and inform the human about moving to the next phase.\n"
            "   - If not satisfied and wants to repeat the current phase with new feedback, mark 'result' as 'REPEAT', record the new feedback in 'feedback', and inform the human about the re-run.\n"
            "   - If wants to go back to a previous phase (from: ***{all_phases}***), mark 'result' as 'GO_TO', specify the 'next_phase', and summarize the rationale in 'feedback'.\n"
            "   - If wants to stop the program, mark 'result' as 'STOP'.\n\n"
            "You can talk to the human untill you are clear on what they want to do next.\n"
            "ALWAYS be polite: thank the human for their feedback and inform what you are doing next. No extra questions."
            
            "3. Return ONLY the following JSON object:\n"
            '{{\n'
            '  "result": "<SUCCESS|REPEAT|GO_TO|STOP>",\n'
            '  "next_phase": "<next phase or empty>",\n'
            '  "feedback": "<user’s feedback or empty>"\n'
            '}}\n\n'
            "Specifically regarding 'feedback':\n"
            "  - If 'result' is 'REPEAT', 'feedback' contains the new inputs for the re-run.\n"
            "  - If 'result' is 'GO_TO', 'feedback' contains the reason for going back.\n"
            "  - If 'result' is 'SUCCESS' or 'STOP', 'feedback' should be an empty string.\n\n"
            "Do NOT include any extra text. Return ONLY the JSON object — ensure no Unicode escapes are used."
    )

    
    # We include a placeholder so the agent can track conversation history/tool calls
    return ChatPromptTemplate.from_messages([
        SystemMessage(content               = system_content),
        HumanMessage(content                = human_content),
        MessagesPlaceholder(variable_name   = "agent_scratchpad"),
    ])
    
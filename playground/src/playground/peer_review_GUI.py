# peer_review_GUI.py

from peer_review_config import QUERY_TYPES, AVAILABLE_LEADER_LLMS, AVAILABLE_AGENT_LLMS, my_name
from peer_review_team import AgentTeam

import streamlit as st
import logging 
from typing import Type, List, Dict, Optional, Tuple, Union, Any

#----------------------------------------------------------------------------------------
# this is required to prevent conflicts between async methods in AgentTeam and StreamLit
#----------------------------------------------------------------------------------------
import nest_asyncio     
nest_asyncio.apply()


#====================================================================
#--------------------------- INITIALIZE() ---------------------------
#====================================================================
def initialize(query: Optional[str] = None):
    """Initialize session state parameters for a fresh run."""
    
    ss = st.session_state

    if "state" not in ss:
        logging.info(f"{my_name()}: Starting session state from scratch")
    else:
        logging.info(f"{my_name()}: Reinitializing session state {ss.state}")
    
    # Check if state is uninitialized or reset to INIT
    if "state" not in ss or ("state" in ss and ss.state == "INIT"): 
        logging.info(f"{my_name()}: Initializing session state")
        ss.state = "QUERY_INPUT"
        ss.available_leader_llm_names = AVAILABLE_LEADER_LLMS
        ss.available_agent_llm_names = AVAILABLE_AGENT_LLMS  
        ss.leader_llm_name = ""             # as selected in the GUI
        ss.agent_llm_names = []             # as selected in the GUI. Later, reflects instantiated agents 
        ss.query = query or ""              # initial query text is set in the 'main'
        ss.query_type = ""                  # returned by analyze_query() 
        ss.criteria = []                    # returned by analyze_query() 
        ss.improved_query = ""              # if an llm decides to improve the initial user query 
        ss.recommended_tools = []           # returned by analyze_query() 
        ss.responses = None                 # SimpleResponseFormat: {'agent_name':'response'}
        ss.peer_reviews = None              # PeerReviewResponseFormat: {'agent_name': {'score':int, 'improvement_points': list[str]}
        ss.failed_agents = []               # Agents that failed to generate response 
        ss.winner_name = ""
        ss.winner_response = ""
        ss.winner_avg_score = 0
        ss.winner_improvement_points = []
        ss.avg_scores = {}
        ss.error_message = ""
        ss.team = None              # IMPORTANT! Holds the AgentTeam object
        ss.iteration = 0
        ss.user_feedback = ""       # this is for future, when human-in-the-loop will be allowed
    

#====================================================================
#------------------------ RESET_STATE() -----------------------------
#====================================================================
def reset_state():
    """Clear session state after query completion, keeping model lists."""
    ss = st.session_state

    logging.info(f"{my_name()}: Resetting state")
    keys_to_keep = ["available_leader_llm_names", "available_agent_llm_names"]
    for key in list(ss.keys()):
        if key not in keys_to_keep:
            del ss[key]
    ss.state = "INIT"  # Fixed: Set state key, not entire session
    initialize()  # Reinitialize

#====================================================================
#------------------------ RENDER_SIDEBAR() --------------------------
#====================================================================
def render_sidebar(leader_llms: List[str], agent_llms: List[str], editable: bool = True):
    """Render sidebar with model selection or read-only display with buttons."""
    
    ss = st.session_state

    with st.sidebar:
        st.title("Leader LLM:")
        if editable:
            ss.leader_llm_name = st.selectbox(
                label="Select Leader LLM",
                options=leader_llms,
                index=(leader_llms.index(ss.leader_llm_name) 
                       if ss.leader_llm_name in leader_llms else 0),
                key="leader_selectbox"
            )
            st.title("Team (2–6 LLMs):")
            selected_team = [opt for opt in agent_llms 
                            if st.checkbox(label=opt, value=opt in ss.agent_llm_names, key=f"cb_{opt}")]
            ss.agent_llm_names = selected_team
        else:
            st.write(f"Selected Leader: {ss.leader_llm_name}")
            st.write("Selected Team:")
            for agent in ss.agent_llm_names:
                if st.button(agent, key=f"btn_{agent}", disabled=agent in ss.failed_agents):
                    st.write(f"Response from {agent}: {ss.responses.get(agent, 'No response')}")  # Stub

#====================================================================
#------------------------ QUERY_INPUT() -----------------------------
#====================================================================
def query_input():
    """Handle query input state with user selections."""
    ss = st.session_state

    if ss.state != "QUERY_INPUT":
        raise RuntimeError("Unexpected state in query_input")
    
    logging.info(f"{my_name()}: Starting")
    render_sidebar(ss.available_leader_llm_names, ss.available_agent_llm_names)
    ss.query = st.text_area("Query Input", ss.query, height=100, key="query_input")
    
    if st.button("Start", key="start_button"):
        if not ss.query.strip():
            ss.error_message = "Please enter a query."
        elif not (2 <= len(ss.agent_llm_names) <= min(6, len(ss.available_agent_llm_names))):
            ss.error_message = "Select 2–6 team members."
        else:
            ss.error_message = ""
            ss.team = AgentTeam()
            ss.leader_llm_name, ss.agent_llm_names = ss.team.initialize_team(ss.leader_llm_name, ss.agent_llm_names)
            
            # later, check those agents that failed to initialize and grey them out 
            ss.state = "QUERY_ANALYSIS_IN_PROGRESS"
            # logging.info(f"{my_name()}: AgentTeam instantiated with {ss.leader_llm_name} and {ss.agent_llm_names}")
            st.rerun()  # Force rerun to reflect new state
    if ss.error_message:
        st.error(ss.error_message)

#====================================================================
#-------------------- QUERY_ANALYSIS_IN_PROGRESS() ------------------
#====================================================================
def query_analysis_in_progress():
    """Analyze query using the leader LLM."""
    
    ss = st.session_state
    if ss.state != "QUERY_ANALYSIS_IN_PROGRESS":
        raise RuntimeError("Unexpected state in query_analysis_in_progress")
    
    logging.info(f"{my_name()}: Starting")
    render_sidebar([], [], editable=False)
    with st.spinner("Analyzing query..."):
        try:
            ss.query_type, ss.recommended_tools, ss.improved_query = ss.team.analyze_query(
                query=ss.query
            )
            ss.query_type = ss.query_type or "OTHER"
            ss.criteria = QUERY_TYPES.get(ss.query_type, [])
            ss.query = ss.improved_query or ss.query
            logging.info(f"{my_name()}: Query type: {ss.query_type}, criteria: {ss.criteria} {ss.improved_query}")
            ss.state = "QUERY_ANALYSIS_DONE"
            st.rerun()
        except Exception as e:
            st.error(f"Query analysis failed: {e}")
            ss.error_message = f"Query analysis failed: {e}"
            logging.error(f"{my_name()}: Query analysis failed: {e}")
            ss.state = "INIT"
            st.rerun()

#====================================================================
#------------------------- QUERY_ANALYSIS_DONE() --------------------
#====================================================================
def query_analysis_done():
    """Display analysis results and transition to response generation."""
    ss = st.session_state
    if ss.state != "QUERY_ANALYSIS_DONE":
        raise RuntimeError("Unexpected state in query_analysis_done")
    
    logging.info(f"{my_name()}: Starting")
    render_sidebar([], [], editable=False)
    st.text_area("Query", ss.query, height=100, disabled=True, key="query_display")
    st.write(f"**Query Type:** {ss.query_type}")
    st.write(f"**Criteria:** {', '.join(ss.criteria) if ss.criteria else 'None'}")
    st.write(f"**Improved Query:** {ss.improved_query if ss.improved_query else 'None'}") 
    st.write(f"**Recommended Tools:** {', '.join(ss.recommended_tools) if ss.recommended_tools else 'None'}")
    
    ss.state = "GENERATING_RESPONSES_IN_PROGRESS"
    st.rerun()

#====================================================================
#-------------------- GENERATING_RESPONSES_IN_PROGRESS() ------------
#====================================================================
def generating_responses_in_progress():
    """Generate responses from agent team."""
    ss = st.session_state
    if ss.state != "GENERATING_RESPONSES_IN_PROGRESS":
        raise RuntimeError("Unexpected state in generating_responses_in_progress")
    
    logging.info(f"{my_name()}: Starting")
    render_sidebar([], [], editable=False)
    with st.spinner("Generating responses..."):
        try:
            if ss.iteration == 0:
                logging.info(f"{my_name()}: First iteration")
                ss.responses = ss.team.generate_responses(
                    query=ss.query,
                    query_type=ss.query_type,
                    criteria=ss.criteria
                )
                
            else:  # we are in the next iteration already: 
                logging.info(f"{my_name()}: Iteration {ss.iteration}")
                ss.team.generate_iterative_improvement( 
                    query = ss.query, 
                    criteria = ss.criteria, 
                    improvement_points = ss.winner_improvement_points, 
                    response = ss.winner_response, 
                    user_feedback = ss.user_feedback 
                )  

            ss.failed_agents = [agent for agent in ss.agent_llm_names if not ss.responses.get(agent)]
            logging.info(f"{my_name()}: Responses generated, Failed agents: {ss.failed_agents}")
            ss.state = "GENERATING_RESPONSES_DONE"
            st.rerun()
        except Exception as e:
            logging.error(f"{my_name()}: Response generation failed: {e}")
            st.error(f"Response generation failed: {e}")
            ss.state = "QUERY_INPUT"
            st.rerun()

#====================================================================
#-------------------- GENERATING_RESPONSES_DONE() -------------------
#====================================================================
def generating_responses_done():
    """Display response generation completion."""
    ss = st.session_state
    if ss.state != "GENERATING_RESPONSES_DONE":
        raise RuntimeError("Unexpected state in generating_responses_done")
    
    logging.info(f"{my_name()}: Starting")


    render_sidebar([], [], editable=False)
  
    st.write("Responses ready.")
    ss.state = "PEER_REVIEW_IN_PROGRESS"
    st.rerun()

#====================================================================
#--------------------- PEER_REVIEW_IN_PROGRESS() --------------------
#====================================================================
def peer_review_in_progress():
    """Perform peer review of responses."""
    ss = st.session_state
    if ss.state != "PEER_REVIEW_IN_PROGRESS":
        raise RuntimeError("Unexpected state in peer_review_in_progress")
    
    logging.info(f"{my_name()}: Starting")
    render_sidebar([], [], editable=False)
    with st.spinner("Peer review in progress..."):
        ss.peer_reviews = ss.team.generate_peer_reviews(
                query=ss.query,
                criteria=ss.criteria,
                responses=ss.responses)   
        logging.info(f"{my_name()}: Peer review generated: {ss.peer_reviews}")      
        
    ss.state = "PEER_REVIEW_DONE"
    st.rerun()

#====================================================================
#---------------------------- PEER_REVIEW_DONE() -------------------
#====================================================================
def peer_review_done():
    """Handle peer review completion with iteration or finish options."""
    ss = st.session_state
    if ss.state != "PEER_REVIEW_DONE":
        raise RuntimeError("Unexpected state in peer_review_done")
    
    logging.info(f"{my_name()}: Starting")
    render_sidebar([], [], editable=False)
    st.write("Peer review completed.")
    
    (ss.avg_scores, 
    ss.winner_name, 
    ss.winner_avg_score, 
    ss.winner_improvement_points) = ss.team.analize_peer_reviews(ss.peer_reviews)
    
    logging.info(f"{my_name()}: Winner: {ss.winner_name} {ss.winner_avg_score} {ss.winner_improvement_points}")
    
    ss.winner_response = ss.responses.get(ss.winner_name, "ERROR")
    
    st.write(f"**Winner:** {ss.winner_name}")
    st.write(f"**Winner Average Score:** {ss.winner_avg_score}")
    st.write(f"**Winner Response:** {ss.winner_response}")
    improvement_points_str = '\n'.join(f"- {point}" for point in ss.winner_improvement_points)
    st.write(f"**Improvement Points:**\n{improvement_points_str}")

    if st.button("Iterate"):
        ss.iteration += 1
        logging.info(f"{my_name()}: Iteration {ss.iteration}")
        ss.state = "GENERATING_RESPONSES_IN_PROGRESS"
        st.rerun()
    if st.button("Finish"):
        reset_state()
        st.rerun()



#====================================================================
#------------------------------- MAIN() -----------------------------
#====================================================================
def main(query: Optional[str] = None):
    """Dispatch states for the app."""
    ss = st.session_state
    logging.info(f"{my_name()}: Starting")
    st.title("LLM Teamwork")
    initialize(query)
    state = ss.state
   
    if state == "QUERY_INPUT":
        query_input()
    elif state == "QUERY_ANALYSIS_IN_PROGRESS":
        query_analysis_in_progress()
    elif state == "QUERY_ANALYSIS_DONE":
        query_analysis_done()
    elif state == "GENERATING_RESPONSES_IN_PROGRESS":
        generating_responses_in_progress()
    elif state == "GENERATING_RESPONSES_DONE":
        generating_responses_done()
    elif state == "PEER_REVIEW_IN_PROGRESS":
        peer_review_in_progress()
    elif state == "PEER_REVIEW_DONE":
        peer_review_done()
    else:
        st.error(f"Unknown state: {state}")

#====================================================================
#---------------------------- __MAIN__ -----------------------------
#====================================================================
if __name__ == "__main__":
    # Test queries for debugging
    #query = "Напишите смешную историю о молодом крокодиле, который спасает принцессу из замка (не более 300 слов)"
    #query = "What time is it now?"
    #query = "Резюмируйте содержание видео: https://www.youtube.com/watch?v=JGwWNGJdvx8"
    #query = "summarize the content of the website: https://www.epam.com as of today (March 2025) and present the EPAM latest stock price"
    #query = "please find the latest information on Leo Lozner. Hint - he is related to EPAM"
    #query = (f"write a python program that calculates the factorial of 'n'. Run it in a Python interpreter to make sure it works properly.")
    #query = (f"пожалуйста, переведите на русский: \n"
    #         f"მ ი ნ დ ო ბ ი ლ ო ბ ა"
    #         f"რუპო ზალმანი პ/ნ 01391057335 და ულიანა შმიდ 01591057979 როგორც   ს/კ 01.16.01.007.003.01.01.007 უძრავი ქონების (ბინა) თანამესაკუთრეები, ოთარი ნარჩემაშვილს პ/ნ 01007015722 და გიორგი ჭიჭინაძეს პ/ნ 01001067259 ვანიჭებ უფლებამოსილებას მასზედ, რომ დაიცვან და წარმოადგინონ ჩვენი ინტერესები ქალაქ თბილისის მუნიციპალიტეტის მერიაში და მის სტრუქტურულ ერთეულებში, ქალაქ თბილისის მუნიციპალიტეტის სსიპ თბილისის მუნიციპალიტეტის არქიტექტურის სამსახურში და სხვა ადმინისტრაციულ ორგანოებში, ჩვენ თანასაკუთრებაში არსებული ბინის გამიჯვნის/გაყოფის მიზნით. მინდობილი პირები ერთად ან               ცალ-ცალკე უფლებამოსილნი არიან წინამდებარე მიზნის მისაღწევად წარადგინონ განცხადება, წერილობითი შეტყობინება, შესაბამისი დოკუმენტაცია, მონაწილეობა მიიღონ სხდომებში, მოითხოვონ შესაბამისი ნებართვების, თანხმობების მოპოვება."
    #         f" მინდობილი პირები წინამდებარე მიზნის მისაღწევად უფლებამოსილნი არიან განახორციელონ კანონმდებლობით აუკრძალავი ნებისმიერი მოქმედებები.\n"
    #         f"მინდობილობა გაცემულია წინამდებარე დავალების შესრულებამდე.")"
    query = "Напишите короткий стих о пьяном роботе"

    main(query)
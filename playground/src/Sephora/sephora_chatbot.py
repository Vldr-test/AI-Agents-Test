# sephora_chatbot.py

import logging
from typing import List, Dict, Optional, Union, Tuple, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

 
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import TavilySearchResults


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig


from langchain.agents import initialize_agent, AgentType, AgentExecutor
import time
import os
from dotenv import load_dotenv

import inspect
from langchain.tools import Tool

import streamlit as st




#==============================================================================================
#------------------------------------ CONFIGURATIONS ------------------------------------------
#==============================================================================================
# File paths
JSONL_FILE: str = r"C:\Users\Vladi_Ruppo\Downloads\skincare_parsed.jsonl"
# VECTOR_STORE_PATH: str = r"C:\Users\Vladi_Ruppo\Downloads\faiss_index"
VECTOR_STORE_PATH: str = "faiss_index"

RAG_RESULTS = 7

CHAIN_TYPE = "CHATBOT"    # "STATELESS" or "CHATBOT"

LLM_MODEL = "anthropic:claude-3-5-sonnet-latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Lambda to get current function name for logging
my_name = lambda: inspect.currentframe().f_back.f_code.co_name

#=========================================================================================
#--------------------------------------- PROMPTS -----------------------------------------
#=========================================================================================
# Create a chat prompt template for system message 
assistant_prompt = """
    You are a helpful and friendly AI Product assistant that answers generic questions about 
    healthcare products, focusing on and prioritizing Sephora products 
    You work for Sephora, providing concise, clear, and relevant information to its potential customers. 
    Your name is Marfusha  - a healthcare assistant from Sephora identifying as female
    
    - Always speak in a professional, helpful tone.
    
    - Only answer based on the provided context — never make up details. 
      If unsure, say: 'I'm sorry, I'm not sure.'
    
    - Find most relevant information about specific Sephora products using the internal search tool. 
    
    - NEVER tell or imply that the current year is "2024" - it is wrong! 
      Always use your Web search tool to check the current date! 
    
    - If the user asks  a generic question about modern trends, latest products, 
      new offerings and discoveries, etc., ALWAYS use your Web search tool.   
  
    - Use your Web search tool, when the user asks about Sephora's product, 
        but there is not enough info from the internal search tool. 
        If you are using information from the Web search tool, 
        ALWAYS mention that this information is from the Web - and ALWAYS mention the country(ies)) 
        your gathered it from! No need to mention Tavily explicitely 
   
    - Answer general question without overselling Sephora's products. Do not oversell / push, but 
      you might find subtle and gentle ways to offer Sephora products naturally. 

    - Never promote competition's products.  
    
    - Keep your responses brief and to the point.

    
    Format:
    - Always include all relevant information in your final response, even if it was mentioned earlier in the conversation.
    - Use emojis to add warmth where appropriate.
    - Use bullet points for clarity.
    - Use MARKUP to emphasize important text in your reply, such as product names
    - Whenever relevant, include image links or product URLs in markdown format:
       - Example image: ![Product Image](https://www.sephora.com/productimages/abc.jpg)
       - Example link: [View on Sephora](https://www.sephora.com/product/{{product_id}})
    """
 
#=========================================================================================
#--------------------------------------- LOCAL_SEARCH() ----------------------------------
#=========================================================================================
def local_search(query: str, retriever)->str:
    """
    Search the local FAISS vector store for Sephora skincare product information based on the user's query.
    Args:
        query (str): The user's question or search term (e.g., "best Sephora moisturizer for dry skin").
    Returns:
        str: A string containing relevant product information retrieved from the database, 
             with each document separated by a newline.
    """
    logging.info(f"{my_name()} starting")

    # Retrieve relevant documents from the FAISS vector store using its retriever
    docs = retriever.get_relevant_documents(query)
    
    # Combine the content of all retrieved documents into a single string
    # Each document's content is separated by a newline for readability
    if docs:
        result = "\n".join([doc.page_content for doc in docs])
        logging.info(f"{my_name()} returning results")
        return result
    else:
        logging.error(f"{my_name()}, no results found")
        return "No matching products found in Sephora’s database."



#=========================================================================================
#----------------------------------- SETUP_RAG_CHAIN () ----------------------------------
#=========================================================================================
def setup_rag_chain(llm_model: str, vector_store_path: str)-> AgentExecutor:
    """
    Set up the RAG chain with the specified LLM provider.
    Args:
        vector_store (FAISS): FAISS vector store for retrieval.
        llm_model (str): LLM model name.
    Returns:
        ConversationalRetrievalChain: Basic chatbot chain with memory.
    """
    start_time = time.time()
    logging.info(f"{my_name()}: Setting up RAG chain with LLM provider '{llm_model}'.")

    #-------------------------------------------------------------------------------- 
    #----------------------- CONNECT TO THE VECTOR STORE: ---------------------------
    #-------------------------------------------------------------------------------- 
    
    def explore_env():
        path_to_check = VECTOR_STORE_PATH:
        parent_dir = os.path.dirname(path_to_check) if path_to_check != "" else "."
        if os.path.exists(parent_dir):
            contents = os.listdir(parent_dir)
            logging.error(f"{my_name()} content of parent directory ('{parent_dir}')")
            st.write(f"Listing contents of the parent directory ('{parent_dir}'):")
            for item in contents:
                st.write(f"- {item}")
                logging.error(f"- {item}")
        else:
            logging.error(f"{my_name()} env: The parent directory '{parent_dir}' does not exist.")
            st.write(f"The parent directory '{parent_dir}' also does not exist.")

    #--------------------------------------------------------------------------------
    if os.path.exists(vector_store_path):
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vector_store = FAISS.load_local(vector_store_path, embeddings=embeddings, allow_dangerous_deserialization=True)
            retriever = vector_store.as_retriever()
            logging.info(f"\n Vector store loaded in {time.time() - start_time} seconds.")
        except Exception as e:
            logging.error(f"{my_name()}: Error loading vector store: {e}")
            explore_env()
            return None
    else:
        logging.error(f"{my_name()} Vector store not found. Please create the vector store first.")
        return None 
    
    #-------------------------------------------------------------------------------- 
    #-------------------------------- CREATE LLM MODEL- -----------------------------
    #-------------------------------------------------------------------------------- 
  
    try:

        #-------------------------------------------------------------------------------- 
        #------------------------------- CREATE TOOLS: ----------------------------------
        #-------------------------------------------------------------------------------- 
        # define Tavily web search tool
        load_dotenv()
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logging.error(f"{my_name()}: TAVILY_API_KEY not found in .env file")
            return None 

        web_search_tool = TavilySearchResults(
            max_results=5,  # Limit results
            search_depth="advanced",  # More comprehensive search
            include_images=True  # Include image results
        )

        # Define the local search tool: 
        local_search_tool = Tool(
            name="local_database_search",  # Unique name for the tool
            func= lambda query: local_search(query, retriever),   # The function to execute when this tool is called
            description="Search Sephora's skincare product database for specific product information, "
                    "such as ingredients, images, links to products, etc.This is your PRIMARY source of information."
                    "Please use the tool to improve your answers with Sephora - specific information. " 
                    "Use the tool to enrich your answers with links and images"
            )
        
        
        tools = [local_search_tool, web_search_tool]

        #--------------------------------------------------------------------------------
        #-------------------------------- CREATE THE AGENT ------------------------------
        #-------------------------------------------------------------------------------- 

        # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,  output_key="output")
        # agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=False,
        #    memory=memory, agent_kwargs={"prompt": assistant_prompt})

        # Create checkpointer with initial history
        checkpointer = MemorySaver()
        
        agent = create_react_agent(
            model = llm_model,   
            tools = tools,   
            checkpointer=checkpointer,
            debug = False,                  # this is a replacement for Verbose
            prompt = assistant_prompt
            )

        logging.info(f"{my_name()}: Agent initialized.")

    except Exception as e:
        logging.error(f"{my_name()}: Error in setup_rag_chain: {type(e).__name__} - {str(e)}")
        return None 
        
    logging.info(f"{my_name()}: RAG chain initialized.")
    return agent

#====================================================================================
#---------------------- HELPER: PARSE_AGENT_RESPONSE() ------------------------------
#====================================================================================
def parse_agent_response(response: Dict[str, Any]) -> str:
    """
    Parses the raw response from the agent, collecting all text from AI messages.
    Args:
        response (dict): The raw response containing a list of messages.
    Returns:
        str: A concatenated string of all text content from AI messages.
    """
    # Validate the response structure
    if not isinstance(response, dict) or 'messages' not in response or not isinstance(response['messages'], list):
        logging.error(f"{my_name()}: Unexpected response structure: {response}")
        return "Error: Received an unexpected response from the agent."

    try:
        # Strategy 1: Direct output key
        if isinstance(response, dict) and 'output' in response:
            return str(response['output'])
        
        # Strategy 2: Last AI Message
        if isinstance(response, dict) and 'messages' in response:
            ai_messages = [
                msg for msg in response['messages'] 
                if isinstance(msg, AIMessage)
            ]
            
            if ai_messages:
                return ai_messages[-1].content
        
        # Strategy 3: Fallback to string representation
        return str(response)
    
    except Exception as e:
        logging.error(f"{my_name()}: Error extracting response: {e}")
        return f"{my_name()}: Error extracting response: {e}"
 
 
#====================================================================================
#----------------------------------- MAIN() -----------------------------------------
#====================================================================================
def main(llm_model: str, vector_store_path: str) -> None:
    """
    Main function to run the Streamlit app for the Sephora chatbot.
    Args:
        llm_model (str): LLM provider name.
        vector_store_path (str): Path to the FAISS vector store.
    """
    # Short alias for session state
    ss = st.session_state
    start_time = time.time()
    logging.info("Starting!")

    # Initialize the agent with a spinner
    with st.spinner("Please allow me about ten seconds to get ready"):
        if "agent" not in ss:
            ss.agent = setup_rag_chain(llm_model=llm_model, vector_store_path=vector_store_path)
            if ss.agent is None:
                st.error("Error initializing RAG chain. Exiting.")
                return
            logging.info("RAG chain initialized!")

    st.write("I am ready for your questions!")

    # Initialize chat history if not present
    if "messages" not in ss:
        logging.info("Initializing chat history.")
        ss.messages = []

    # Container for chat history (non-scrollable, grows with content)
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for message in ss.messages:
            with st.chat_message(message["role"], avatar=None):
                st.markdown(message["content"])

    # Get user input
    user_input = st.chat_input("Please ask me...")
    logging.info(f"{my_name()}: User question: {user_input}")

    if user_input:
        # Append and display user message immediately
        ss.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user", avatar=None):
                st.markdown(user_input)

        # Process agent response with spinner
        with st.spinner("Thinking..."):
            try:
                response = ss.agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config={"configurable": {"thread_id": "sephora_chat"}}
                )
                logging.info(f"{my_name()}: raw response: {response}")
                # Parse the response
                bot_answer = parse_agent_response(response)
                logging.info(f"{my_name()}: Bot answer: {bot_answer}")
            except Exception as e:
                error_msg = f"An error occurred: {type(e).__name__} - {str(e)}"
                logging.error(error_msg)
                bot_answer = error_msg

        # Append assistant response to chat history
        ss.messages.append({"role": "assistant", "content": bot_answer})

        # Display assistant response with default character-by-character typing effect
        with chat_container:
            with st.chat_message("assistant", avatar=None):
                placeholder = st.empty()
                displayed_text = ""
                for char in bot_answer:
                    displayed_text += char
                    placeholder.markdown(displayed_text)
                    time.sleep(0.01)  # Adjust typing speed as needed

    logging.info(f"{my_name()}: Time passed: {time.time() - start_time}")

if __name__ == "__main__":
    main(llm_model=LLM_MODEL, vector_store_path= VECTOR_STORE_PATH )

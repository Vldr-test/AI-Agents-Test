# sephora_chatbot.py

import logging
from typing import List, Dict, Optional, Union, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType, AgentExecutor
import time
import os
from dotenv import load_dotenv
import inspect
from langchain.tools import Tool
import requests
import streamlit as st


#==============================================================================================
#------------------------------------ CONFIGURATIONS ------------------------------------------
#==============================================================================================
# File paths
JSONL_FILE: str = r"C:\Users\Vladi_Ruppo\Downloads\skincare_parsed.jsonl"
VECTOR_STORE_PATH: str = r"C:\Users\Vladi_Ruppo\Downloads\faiss_index"

RAG_RESULTS = 7

CHAIN_TYPE = "CHATBOT"    # "STATELESS" or "CHATBOT"

LLM_MODEL = "claude-3-5-sonnet-latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Lambda to get current function name for logging
my_name = lambda: inspect.currentframe().f_back.f_code.co_name

#=========================================================================================
#--------------------------------------- PROMPTS -----------------------------------------
#=========================================================================================
assistant_prompt = PromptTemplate.from_template(
    """
        Previous conversation:
            ***{chat_history}***
        
        You are a helpful and friendly AI Product assistant that answers generic questions about 
        healthcare products, focusing on and prioritizing Sephora products 
        You work for Sephora, making sure its customers get concise, clear, and relevant information. 

        - Always speak in a professional, helpful tone.
        - Only answer based on the provided context — never make up details.
        - If asked a generic question, you can use the Web search tool. 
        - If unsure, say: "I’m sorry, I’m not sure."
        - If available and relevant, include image links or product URLs in markdown format:
           - Example image: ![Product Image](https://www.sephora.com/productimages/abc.jpg)
           - Example link: [View on Sephora](https://www.sephora.com/product/{{product_id}})
        
        Format:
        - Use emojis to add warmth where appropriate.
        - Use bullet points for clarity.
        - Use markup to emphasize important points 
        - Keep your responses brief and to the point.

        Use the following pieces of context to answer the question:
        ***{context}***

        Do NOT explicitly refer to the provided context. Highlight product names or ingredients when useful.

        Question: {question}
    """
)


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
    logging.info(f"{my_name()}, query: {query}")

    # Retrieve relevant documents from the FAISS vector store using its retriever
    docs = retriever.get_relevant_documents(query)
    
    # Combine the content of all retrieved documents into a single string
    # Each document's content is separated by a newline for readability
    if docs:
        result = "\n".join([doc.page_content for doc in docs])
        logging.info(f"{my_name()}, result: {result}")
        return result
    else:
        logging.error(f"{my_name()}, no results found")
        return "No matching products found in Sephora’s database."


#=========================================================================================
#------------------------------------ WEB_SEARCH() ---------------------------------------
#=========================================================================================
import requests
import logging
from dotenv import load_dotenv
import os
import streamlit as st

def web_search(query: str) -> str:
    """
    Perform a web search using the Serper API to find general skincare information or real-time data.
    Args:
        query (str): The user's question or search term (e.g., "current time in New York").
    Returns:
        str: A string containing the answer or snippets from the top search results,
             or a message if no results are found or an error occurs.
    """
    logging.info(f"\n{my_name()} query: {query}")
    
    # Ensure API key is set
    if "serper_api_key" not in st.session_state:
        load_dotenv()
        st.session_state.serper_api_key = os.getenv("SERPER_API_KEY")
        if not st.session_state.serper_api_key:
            logging.error("{my_name()}: SERPER_API_KEY not found in .env file")
            return "Web search is unavailable due to missing API key."
    
    # Define the correct API endpoint
    url = "https://google.serper.dev/search"
    
    # Set headers with API key
    headers = {
        "X-API-KEY": st.session_state.serper_api_key
    }
    
    # Set parameters: query and limit to top 3 results
    params = {"q": query, "num": 4}
    
    try:
        # Send the GET request to the Serper API
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an error for non-200 status codes
        
        # Parse the JSON response
        results = response.json()
        
        # Check for answerBox first (for direct answers like time queries)
        if "answerBox" in results:
            answer = results["answerBox"].get("answer", "No direct answer found")
            return answer
        else:
            # Fallback to organic results' snippets
            snippets = [result["snippet"] for result in results.get("organic", [])]
            if snippets:
                return "\n".join(snippets)
            else:
                return "No results found."
    
    except requests.exceptions.RequestException as e:
        logging.error(f"{my_name()}, web_search, request error: {e}")
        return "Sorry, I couldn’t retrieve web search results due to a network error."
    except ValueError as e:
        logging.error(f"{my_name()}, web_search, JSONDecodeError: {e}, Response: {response.text}")
        return "Sorry, there was an issue processing the search results."
    
    # Alternative: 
    #   from langchain.utilities import GoogleSerperAPIWrapper
    #   serper = GoogleSerperAPIWrapper()  # Ensure API key is set in environment or Streamlit secrets
    #   return serper.run(query)



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
    if os.path.exists(vector_store_path):
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vector_store = FAISS.load_local(vector_store_path, embeddings=embeddings, allow_dangerous_deserialization=True)
            retriever = vector_store.as_retriever()
            logging.info(f"\n Vector store loaded in {time.time() - start_time} seconds.")
        except Exception as e:
            logging.error(f"{my_name()}: Error loading vector store: {e}")
            return None
    else:
        logging.error(f"{my_name()} Vector store not found. Please create the vector store first.")
        return None 
    
    #-------------------------------------------------------------------------------- 
    #-------------------------------- CREATE LLM MODEL- -----------------------------
    #-------------------------------------------------------------------------------- 
    try:
        llm = init_chat_model(model=llm_model)
        logging.info(f"{my_name()}: LLM created for: {llm_model}")
    except Exception as e:
        logging.error(f"{my_name()}: Can't create an LLM for: {llm_model}")
        raise e
    
    try:
        # Create memory object:
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,  # Needed by newer chat models
            output_key="output" 
        )

        #-------------------------------------------------------------------------------- 
        #------------------------------- CREATE TOOLS: ----------------------------------
        #-------------------------------------------------------------------------------- 
        SERPER_API_KEY = os.getenv("SERPER_API_KEY")  # Get the API key
        if "serper_api_key" not in st.session_state:
            load_dotenv()
            st.session_state.serper_api_key = os.getenv("SERPER_API_KEY")
            if not st.session_state.serper_api_key:
                logging.error(f"{my_name()}: SERPER_API_KEY not found in .env file")
                return None     # could be made more robust, but I will leave it for future

        # Define the local search tool: 
        local_search_tool = Tool(
            name="local_database_search",  # Unique name for the tool
            func= lambda query: local_search(query, retriever),   # The function to execute when this tool is called
            description="Search Sephora's skincare product database for specific product information, "
                    "such as ingredients, images, links to products, etc.This is your PRIMARY source of information."
                    "Please use the tool to improve your answers with Sephora - specific information. " 
                    "Use the tool to enrich your answers with links and images"
            )
        
        # Define the web search tool: 
        web_search_tool = Tool(
            name="web_search",  # Unique name for the tool
            func=web_search,    # The function to execute when this tool is called
            description="Search the web for general skincare information or questions "
                        "that are not specific to Sephora products, or require real-time data."
                        "Do NOT use the tool for searching for products other than Sephora."
                        
        )
        
        tools = [local_search_tool, web_search_tool]

        #--------------------------------------------------------------------------------
        #-------------------------------- CREATE THE AGENT ------------------------------
        #-------------------------------------------------------------------------------- 
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory
        )

        logging.info(f"{my_name()}: Agent initialized.")

    except Exception as e:
        logging.error(f"{my_name()}: Error in setup_rag_chain: {type(e).__name__} - {str(e)}")
        return None 
        
    logging.info(f"{my_name()}: RAG chain initialized.")
    return agent

#=========================================================================================
#---------------------------------------- MAIN () ----------------------------------------
#=========================================================================================
def main(llm_model: str, vector_store_path: str) -> None:
    """
    Main function to run the Streamlit app for the Sephora chatbot.
    Args:
        llm_model (str): LLM provider name.
    """
    ss = st.session_state
    start_time = time.time()
    logging.info(f"{my_name()}: Starting! ")
     
    with st.spinner("Please allow me about ten seconds to get ready"):
        # Assume the vector store is created on disk.
        # We store it in session state to avoid re-loading on every interaction.
        if "agent" not in ss: 
            ss.agent = setup_rag_chain(llm_model=llm_model, vector_store_path=vector_store_path)
            if ss.agent is None:
                st.error("Error initializing RAG chain. Exiting.")
                return      
            logging.info(f"{my_name()}: RAG chain initialized!")
    
    st.write("I am ready for your questions!")

    # Initialize or retrieve chat history from session state
    if "messages" not in st.session_state:
        logging.info(f"{my_name()}: Initializing chat history.")
        st.session_state.messages = []

    # Display the chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    user_input = st.chat_input("Please ask me...")
    logging.info(f"User question: {user_input}")

    with st.spinner("Thinking about it..."):
        if user_input:
            # Append the user's message to history
            ss.messages.append({"role": "user", "content": user_input})
            try:
                response = ss.agent.invoke({"input": user_input})
                # logging.info(f"\n bot raw response: {response}")
                bot_answer = response["output"]
                # logging.info(f"\n metadata = {[r.metadata for r in response["source_documents"]]}")
                logging.info(f"\n bot answer: {bot_answer}")
            except Exception as e:
                bot_answer = "Sorry, an error occurred while processing your request."
                logging.error(f"\n Exception: {type(e).__name__} - {str(e)}")
                st.error(str(e))
            st.session_state.messages.append({"role": "assistant", "content": bot_answer})
            logging.info(f"time passed: {time.time() - start_time}")
            # Rerun the app to update the conversation
            st.rerun()

if __name__ == "__main__":
    main(llm_model=LLM_MODEL, vector_store_path= VECTOR_STORE_PATH )

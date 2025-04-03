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
VECTOR_STORE_PATH = "faiss_index"

RAG_RESULTS = 5

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

greeting = str("Hi! I'm Amina, your Sephora beauty advisor. " 
        "I'm here to help you with any questions about skincare, makeup, or beauty products available at Sephora. " 
        "How can I assist you today?")

assistant_prompt = """
You are **Amina**, a helpful, friendly, and professional AI assistant for **Sephora** customers.  
You specialize in providing concise, accurate, and relevant information about Sephora products, beauty trends, and personalized recommendations.
---
Behavior Guidelines:

- Always respond in a **professional, warm, and helpful tone**.
- If you're **unsure** of an answer, say so — NEVER make up details.
- If the **user's question is unclear**, ask for clarification before answering.
- **Never promote competitor products**.
- If appropriate, introduce Sephora products and suggestions **without overselling**.
- When you mention a Sephora product, ALWAYS **show product's image and / or link to this product** from [internal_search] tool.
- Aim to **understand the customer's needs** — ask clarifying questions (e.g. skin type, preferences, budget, occasion) when helpful.
---

Tool Usage:

- Use the `[internal_search]` tool to retrieve product details from Sephora’s internal catalog.
- Use the `[web_search]` tool when:
  - The user asks about **trends, new launches, modern beauty topics**, or
  - The internal search does **not return results**.
- If you use `[web_search]` data, always state that **this information is from the web** and mention the **country** the result is from.
- Never refer to the tools by name when speaking to the user (e.g., don’t mention Tavily or "search tool").
---

 Response Format:

- Be **brief and to the point**.
- Include **all relevant information** in your final response, even if it came up earlier in the conversation.
   + After using any tool, ALWAYS generate a final, complete response for the user. 
   + Do not wait for further instructions — conclude the thought clearly, even for open-ended or trend-related questions.
- Use:
  - Bullet points for features, benefits, or comparisons.
  - **Bold** to emphasize product names.
  
  - When you mention a Sephora product, show it's images and / or links to this product from [internal_search_tool]
  - Markdown format for product images and links:
    - Product image: `![Product Image](https://www.sephora.com/productimages/abc.jpg)`
    - Product link: `[View on Sephora](https://www.sephora.com/product/{{product_id}})`
---
Temporal Accuracy:

- NEVER say or imply that the current year is "2024" — it is wrong!
- Use the `[web_search]` tool to verify the current date when needed.

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
    start_time = time.time()
    logging.info(f"{my_name()} starting")

    # Retrieve relevant documents from the FAISS vector store using its retriever
    docs = retriever.get_relevant_documents(query)
    
    # Combine the content of all retrieved documents into a single string
    # Each document's content is separated by a newline for readability
    if docs:
        result = "\n".join([doc.page_content for doc in docs])
        logging.info(f"{my_name()} returning results. response time: {time.time() - start_time}")
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
     
    def explore_env(vector_store_path: str):
        path_to_check = vector_store_path 
        logging.info(f"{my_name()} path_to_check: {path_to_check}")
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
            explore_env(vector_store_path)
            return None
    else:
        logging.error(f"{my_name()} Vector store not found. Please create the vector store first.")
        explore_env(vector_store_path)
        return None 
    
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
                    "such as ingredients, images, links to products, etc."
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


    # Initialize chat history if not present
    if "messages" not in ss:
        logging.info("Initializing chat history.")
        ss.messages = []
        ss.messages.append({"role": "assistant", "content": greeting})

    # Container for chat history (non-scrollable, grows with content)
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for message in ss.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Get user input
    user_input = st.chat_input("Please ask me...")
    logging.info(f"{my_name()}: User question: {user_input}")

    if user_input:
        # Append and display user message immediately
        ss.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
    
        start_time = time.time()

        # Process agent response with spinner
        with st.spinner("Thinking..."):
            try:
                response = ss.agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config={"configurable": {"thread_id": "sephora_chat"}}
                )
                # logging.info(f"{my_name()}: raw response: {response}")
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
        # with chat_container:
        #    with st.chat_message("assistant"):
        #        placeholder = st.empty()
        #        displayed_text = ""
        #        for char in bot_answer:
        #            displayed_text += char
        #            placeholder.markdown(displayed_text)
        #            time.sleep(0.01)  # Adjust typing speed as needed

        # Display assistant response with word-by-word typing effect, preserving Markdown
        with chat_container:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                displayed_text = ""
                lines = bot_answer.split("\n")  # Split into lines first
                for line in lines:
                    words = line.split()  # Split each line into words
                    for word in words:
                        displayed_text += word + " "  # Add word and a space
                        placeholder.markdown(displayed_text)
                        time.sleep(0.05)  # Adjust typing speed as needed
                    displayed_text += "\n"  # Add newline after each line
                    placeholder.markdown(displayed_text)


    logging.info(f"{my_name()}: Total time for response: {time.time() - start_time}")

if __name__ == "__main__":
    # Get the absolute path of the directory containing this script
    working_dir = os.path.dirname(os.path.abspath(__file__))
    logging.info(f"Working directory: {working_dir}")
    vector_store_path = os.path.join(working_dir, VECTOR_STORE_PATH)
    logging.info(f"Vector store path: {vector_store_path}")
    main(llm_model=LLM_MODEL, vector_store_path= vector_store_path  )

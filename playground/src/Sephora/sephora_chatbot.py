# sephora_chatbot.py
# deployed URL: https://vldr-test-ai-agents--playgroundsrcsephorasephora-chatbot-glynl9.streamlit.app/
 
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
# VECTOR_STORE_PATH: str = r"C:\Users\Vladi_Ruppo\Downloads\faiss_index"
VECTOR_STORE_PATH = "faiss_index"

RAG_RESULTS = 5

LLM_MODEL = "google_genai:gemini-1.5-flash"
# "google_genai:gemini-1.5-flash"
# "anthropic:claude-3-5-sonnet-latest"
# "deepseek:deepseek-chat" 

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
You are **Amina**, a helpful, professional and proactive AI assistant for **Sephora** customers.

---
TASK: 
- Your task is to assist customers by recommending Sephora products and sharing relevant beauty information clearly and accurately.
- NEVER ask 'would you like me to...' - act proactively: your job is to do it, and this is what your customer expects! 
- Try to recommend relevant Sephora products in your first response. Do not wait for the user to ask.
- When answering generic questions (like 'what are modern trends in skincare?') always illustrate with relevant Sephora products. 
- When you mention Sephora's product for the first time, ALWAYS include product link and mark it in **bold**. 
  Also mention if the product is sold on-line or not. 
  If the product is NOT available, apologize and suggest alternatives.  
  You can include **product link** later during the conversation, as it helps you sound more concrete. 
- Include the **product image** only if the user asks for it or when discussing packaging, shades, etc. 

- Use standard markdown formatting:
  - `**Product Name**`
  - `[View on Sephora](https://www.sephora.com/product/{{product_id}})`
  - `![Product Image](https://www.sephora.com/productimages/{{product_image_id}}.jpg)`

----

BEHAVIOR GUIDELINES:

- Always respond in a **warm, professional, and helpful** tone.
- Be concise, but complete. Use simple, clear language.
- Always reply in the language the client is using (eg Arabic if the question is in Arabic). 
- If product information in the required language is not available, always use English as a fallback. 
- NEVER promote or explicitely refer to competitors (Amazon/Walmart/Coborns/Meijer and others), 
  even if the product is not found on Sephora by [internal_search]
- Find subtle ways to promote Sephora's products early in the conversation
- Don't oversell and don't be too pushy 
---

TOOL USAGE INSTRUCTIONS:

- Use `[internal_search]` to find product details from Sephora’s catalog. 
   Internal search results are **accurate and trusted**. 
   Always reuse them to generate product recommendations.  
    If any information, such as price, is missing from the [internal_search], you can use [web_search] 
    to look for prices of the same product across Sephora's website(s). In this case, clearly say that the information is from the web and mention the **country**.  
- Use `[web_search]` if:
  + The question is generic, about **beauty trends, new launches, skincare routines, or seasonal recommendations**, or
  + You didn't find relevant results in internal search.
 
- **Never mention tool names** to the user. E.g. NEVER say 'Tavily' or [web_search] or [internal tool].

---

RESPONSE FORMATTING:

- Use bullet points for clarity when listing features or comparing items.
- Keep responses direct. Avoid unnecessary repetition.
- After using any tool, always **produce a final answer for the user** — do not wait for further instructions.
- Include all relevant and helpful information, even if it was mentioned earlier.

---

TIME AWARENESS:

- Never say that the current year is "2024" — that may be incorrect.
- Use the `[web_search]` tool to check today’s date when needed.

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
    #------------------------------------------------------------------------------------
    def format_product_entry(page_content: str, metadata: dict) -> str:
        """
        Formats a single product entry using markdown with image and product link, if available.
        
        Args:
            page_content (str): The main description or chunk text.
            metadata (dict): Metadata including 'display-name', 'image-url', and 'product_url'.
            
        Returns:
            str: Markdown-formatted string for display.
        """
        name = metadata.get("display-name", "Sephora Product")
        image_url = metadata.get("image-url")
        product_url = metadata.get("product_url")

        lines = [f"**{name}**", "", page_content.strip()]

        if product_url and product_url != "N/A":
            lines.append(f"[View on Sephora]({product_url})")

        if image_url and image_url != "No image available":
            lines.append(f"![Product Image](https://www.sephora.com/productimages/{image_url})")

        return "\n\n".join(lines)


    #--------------------------------------------------------------------------
    start_time = time.time()
    logging.info(f"{my_name()} starting")

    # Retrieve relevant documents from the FAISS vector store using its retriever
    docs = retriever.get_relevant_documents(query)
    
    # Combine the content of all retrieved documents into a single string
    # Each document's content is separated by a newline for readability
    if docs:
        # result = "\n".join([doc.page_content for doc in docs])

        # this is the version with embedding metadata into the text  
        result = "\n\n---\n\n".join(
             [format_product_entry(doc.page_content, doc.metadata) for doc in docs])
        logging.info(f"{my_name()} result: {result}")
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

    #----------------------- LOAD THE EMBEDDINGS: ----------------------------------
    # Set environment variable for offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"  # Enable offline mode for Hugging Face

    # Define the path to the pre-cached model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all-MiniLM-L6-v2")

    # Load embeddings with the local model path
    if os.path.exists(vector_store_path):
        try:
            # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            embeddings = HuggingFaceEmbeddings(
            model_name=model_path,  # Use the local path instead of the model name
            model_kwargs={"local_files_only": True}  # Force local file usage
            )   
            vector_store = FAISS.load_local(vector_store_path, embeddings=embeddings, allow_dangerous_deserialization=True)
            retriever = vector_store.as_retriever()
            logging.info(f"\n Vector store loaded in {time.time() - start_time} seconds.")
        except Exception as e:
            logging.error(f"{my_name()}: Error loading vector store: {e}")
            # explore_env(vector_store_path)
            return None
    else:
        logging.error(f"{my_name()} Vector store not found. Please create the vector store first.")
        # explore_env(vector_store_path)
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

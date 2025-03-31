# sephora_chatbot.py

import logging
from typing import List, Dict, Optional, Union
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
# from langchain_community.chat_models import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
import json
import inspect


#==============================================================================================
#------------------------------------ CONFIGURATIONS ------------------------------------------
#==============================================================================================
# File paths
JSONL_FILE: str = r"C:\Users\Vladi_Ruppo\Downloads\skincare_parsed.jsonl"
VECTOR_STORE_PATH: str = r"C:\Users\Vladi_Ruppo\Downloads\faiss_index"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100

RAG_RESULTS = 5

CHAIN_TYPE = "CHATBOT"    # "STATELESS" or "CHATBOT"


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "claude-3-5-sonnet-latest"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Lambda to get current function name for logging
my_name = lambda: inspect.currentframe().f_back.f_code.co_name

#=========================================================================================
#--------------------------------------- PROMPTS -----------------------------------------
#=========================================================================================
assistant_prompt = PromptTemplate.from_template(
    """
        You are a helpful and friendly AI Product assistant that answers questions about healthcare products,
        based on product specifications. 
        You work for Sephora, making sure its customers get concise, clear, and relevant information. 

        - Always speak in a professional, helpful tone.
        - Only answer based on the provided context — never make up details.
        - Use bullet points for clarity.
        - If unsure, say: "I’m sorry, I’m not sure."
        - Use emojis to add warmth where appropriate.
        - If available and relevant, include image links or product URLs in markdown format:
        - Example image: ![Product Image](https://www.sephora.com/productimages/abc.jpg)
        - Example link: [View on Sephora](https://www.sephora.com/product/{{product_id}})

        Use the following pieces of context to answer the question:
        ***{context}***

        Do NOT explicitly refer to the provided context. Highlight product names or ingredients when useful.

        Question: {question}
    """
    )

#=========================================================================================
#-------------------------------- LOAD_DOCUMENTS_FROM_JSON() -----------------------------
#=========================================================================================
def load_documents_from_jsonl(file_path: str) -> List[Document]:
    """
    Load documents from a JSONL file using JSONLoader, converting all fields into content.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        List[Document]: List of LangChain Document objects with all product fields embedded.

    Raises:
        FileNotFoundError: If the input file does not exist.
    """
    logging.info(f"{my_name()}: Loading documents from '{file_path}'...")
    if not os.path.exists(file_path):
        logging.error(f"{my_name()}: Input file '{file_path}' not found.")
        raise FileNotFoundError(f"Input file '{file_path}' not found.")
    
    def dict_to_text(record: Dict) -> str:
        """Helper to format all fields of a record into a single string."""
        content_lines = []
        for key, value in record.items():
            if isinstance(value, dict):
                sub_content = "; ".join(f"{k}: {v}" for k, v in value.items() if v is not None)
                content_lines.append(f"{key}: {sub_content}")
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    sub_content = "; ".join(
                        f"{item.get('view-type', 'unknown')}: {', '.join(item.get('images', []))}"
                        for item in value
                    )
                    content_lines.append(f"{key}: {sub_content}")
                else:
                    content_lines.append(f"{key}: {', '.join(str(v) for v in value if v)}")
            elif value is not None:
                content_lines.append(f"{key}: {value}")
        return "\n".join(content_lines)

    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".",
        text_content=False,
        json_lines=True
    )
    raw_docs = loader.load()
    documents = []

    for doc in raw_docs:
        try:
            parsed = json.loads(doc.page_content)
        except json.JSONDecodeError as e:
            logging.error(f"{my_name()}: Error decoding JSON: {e}")
            continue
        content = dict_to_text(parsed)
        enriched_metadata = {
            "product_id": parsed.get("product-id", "N/A"),
            "brand": parsed.get("brand", "N/A"),
            "display-name": parsed.get("display-name", "N/A"),
            "online-flag": parsed.get("online-flag", "N/A"),
            "available-flag": parsed.get("available-flag", "N/A")
        }
        
        # Try to extract the first image URL, if available.
        image_url = None
        images = parsed.get("images", [])
        if isinstance(images, list) and images:
            for view in images:
                if isinstance(view, dict):
                    image_list = view.get("images", [])
                    if image_list:
                        image_url = image_list[0]
                        break
        if image_url:
            enriched_metadata["image-url"] = image_url

        documents.append(Document(page_content=content, metadata=enriched_metadata))

    logging.info(f"{my_name()}: Loaded {len(documents)} initial documents.")
    return documents


#=========================================================================================
#------------------------------------ CHUNK_DOCUMENTS() ----------------------------------
#=========================================================================================
def chunk_documents(documents: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP) -> List[Document]:
    """
    Chunk documents using RecursiveCharacterTextSplitter.
    Args:
        documents (List[Document]): List of initial documents to chunk.
        chunk_size (int): Target character length per chunk. Defaults to 1000.
        chunk_overlap (int): Number of overlapping characters between chunks. Defaults to 50.
    Returns:
        List[Document]: List of chunked documents with preserved metadata.
    """
    logging.info(f"{my_name()}: Chunking documents with size {chunk_size} and overlap {chunk_overlap}...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )

    chunked_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        
        # Retrieve enriched metadata directly from the loaded document
        # (Assuming load_documents_from_jsonl() has stored basic fields in metadata)
        enriched_metadata = doc.metadata.copy()
        product_id = enriched_metadata.get("product_id", "N/A")
        display_name = enriched_metadata.get("display-name", "N/A")
        online_flag = enriched_metadata.get("online-flag", "N/A")
        available_flag = enriched_metadata.get("available-flag", "N/A")
        
        # Construct a product URL if product_id is available
        product_url = f"https://www.sephora.com/product/{product_id}" if product_id != "N/A" else "N/A"
        enriched_metadata["product_url"] = product_url

        # Construct image markdown, if an image URL is available in metadata.
        # If you haven't extracted it during loading, you might extract it here from raw JSON,
        # but ideally it should be present in enriched_metadata (e.g., under "image-url").
        image_url = enriched_metadata.get("image-url")
        image_md = f"![Product Image]({image_url})" if image_url else "No image available"
        enriched_metadata["image_md"] = image_md

        # Create an availability text for user-friendly display.
        availability_text = "Available" if str(available_flag).lower() == "true" else "Not available"
        enriched_metadata["availability_text"] = availability_text

        # Build the prefix text for each chunk including the enriched metadata.
        chunk_prefix = (
            f"Product: {display_name} (ID: {product_id})\n"
            f"Availability: {availability_text} | {online_flag}\n"
            f"Image: {image_md}\n"
            f"[View on Sephora]({product_url})\n\n"
        )
        
        for i, chunk in enumerate(chunks):
            chunk_with_context = chunk_prefix + chunk
            chunked_docs.append(Document(
                page_content=chunk_with_context,
                metadata={**enriched_metadata, "chunk_id": f"{product_id}_chunk{i+1}"}
            ))

    logging.info(f"{my_name()}: Created {len(chunked_docs)} chunked documents.")
    return chunked_docs




#=========================================================================================
#----------------------------------- SETUP_RAG_CHAIN () ----------------------------------
#=========================================================================================
def setup_rag_chain(vector_store: FAISS, llm_model: str, chain_type: str = "CHATBOT"):
    """
    Set up the RAG chain with the specified LLM provider.
    Args:
        vector_store (FAISS): FAISS vector store for retrieval.
        llm_provider (str): LLM model name.
        chain_type: "STATELESS" or "CHATBOT"
    Returns:
        RetrievalQA: stateless QA 
        ConversationalRetrievalChain: basic chatbot  
    """
    logging.info(f"{my_name()}: Setting up RAG chain with LLM provider '{llm_model} chain_type {chain_type}.")
    
    chain = None

    try:
        llm = init_chat_model(model=llm_model)
        logging.info(f"{my_name()} llm created for: {llm_model}")
    except Exception as e:
        logging.error(f"{my_name()}: Can't create an llm for: { llm_model}")
        raise e
    
    try:
        if chain_type == "CHATBOT":
            # create memory object:
            memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,  # needed by newer chat models
            output_key="answer" 
            )

            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(search_kwargs={"k": RAG_RESULTS}),
                memory = memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": assistant_prompt}
            )

        elif chain_type == "STATELESS":
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": RAG_RESULTS}),
                return_source_documents=True
            )
        else:
            logging.error(f"{my_name()}: unsupported chain_type: {chain_type}")
            raise RuntimeError(f"unsupported chain_type: {chain_type}")
    
    except Exception as e:
        logging.error(f"{my_name()}: Error in setup_rag_chain: {type(e).__name__} - {str(e)}")
        raise e
        
    logging.info(f"{my_name()}: RAG chain initialized.")
    return chain

#=========================================================================================
#----------------------------------- QUERY_RAG_CHAIN () ----------------------------------
#=========================================================================================
def query_stateless_chain(chain: RetrievalQA, question: str) -> None:
    """
    Query the RAG chain and log the answer and source documents.
    Args:
        chain (RetrievalQA): The RAG chain to query.
        question (str): The question to ask.
    """
    logging.info(f"{my_name()}: Querying with question: '{question}'")
    result = chain.invoke({"query": question})
    logging.info(f"{my_name()}: Answer: {result['result']}")
    logging.info(f"{my_name()}: Source Documents:")
    for doc in result["source_documents"]:
        logging.info(f"{my_name()}: - Product ID: {doc.metadata['product_id']}")
        logging.info(f"{my_name()}:   Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
        logging.info(f"{my_name()}:   Content (first 200 chars): {doc.page_content[:200]}...")

#=========================================================================================
#---------------------------------------- MAIN () ----------------------------------------
#=========================================================================================
def main(llm_model: str) -> None:
    """
    Main function to run the LangChain pipeline with dynamic chunking 
    Args:
        llm_model (str): LLM provider name.  
    """
    logging.info(f"{my_name()}: Starting pipeline with LLM model '{llm_model}'...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # Check if vector store already exists
        if os.path.exists(VECTOR_STORE_PATH):
            logging.info(f"{my_name()}: Found existing vector store. Loading...")
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings = embeddings, allow_dangerous_deserialization=True)
            # Optionally, if you expect changes in the JSONL file, add a file timestamp or hash check here
        else:
            # If not, load and process the documents
            documents = load_documents_from_jsonl(JSONL_FILE)
            chunked_documents = chunk_documents(documents)
            vector_store = FAISS.from_documents(chunked_documents, embedding = embeddings)
            vector_store.save_local(VECTOR_STORE_PATH)
            logging.info(f"{my_name()}: Built and saved new vector store.")
    
        rag_chain = setup_rag_chain(vector_store, llm_model, chain_type=CHAIN_TYPE)
        
        while True and CHAIN_TYPE == "CHATBOT":
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            response = rag_chain.invoke({"question": user_input})
            print(f"Bot: {response['answer']}")
     
    except Exception as e:
        logging.error(f"{my_name()}: Error in pipeline: {type(e).__name__} - {str(e)}")

if __name__ == "__main__":
    main(llm_model=LLM_MODEL)
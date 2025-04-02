# sephora_indexing.py

import logging
from typing import List, Dict, Optional, Union
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter

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

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Lambda to get current function name for logging
my_name = lambda: inspect.currentframe().f_back.f_code.co_name


#=========================================================================================
#-------------------------------- LOAD_DOCUMENTS_FROM_JSON() -----------------------------
#=========================================================================================
def load_documents_from_jsonl(file_path: str) -> List[Document]:
    """
    Load documents from a JSONL file, converting each JSON record into a flattened text
    and extracting enriched metadata (product_id, brand, display-name, online/available flags,
    and image URL if available).
    """
    logging.info(f"{my_name()}: Loading documents from '{file_path}'...")
    if not os.path.exists(file_path):
        logging.error(f"{my_name()}: Input file '{file_path}' not found.")
        raise FileNotFoundError(f"Input file '{file_path}' not found.")
    
    def dict_to_text(record: Dict) -> str:
        """Recursively convert a dictionary to a human-readable text string."""
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
            # Parse the JSON string
            record = json.loads(doc.page_content)
        except Exception as e:
            logging.warning(f"{my_name()}: Skipping invalid JSON line: {e}")
            continue

        text = dict_to_text(record)
        enriched_metadata = {
            "product_id": record.get("product-id", "N/A"),
            "brand": record.get("brand", "N/A"),
            "display-name": record.get("display-name", "N/A"),
            "online-flag": record.get("online-flag", "N/A"),
            "available-flag": record.get("available-flag", "N/A")
        }
        # Attempt to extract the first image URL (if available)
        image_url = None
        images = record.get("images", [])
        if isinstance(images, list) and images:
            for view in images:
                if isinstance(view, dict):
                    image_list = view.get("images", [])
                    if image_list:
                        image_url = image_list[0]
                        break
        if image_url:
            enriched_metadata["image-url"] = image_url

        documents.append(Document(page_content=text, metadata=enriched_metadata))

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
        
        # Retrieve enriched metadata from the document (extracted during loading)
        enriched_metadata = doc.metadata.copy()
        product_id = enriched_metadata.get("product_id", "N/A")
        brand = enriched_metadata.get("brand", "N/A")
        display_name = enriched_metadata.get("display-name", "N/A")
        online_flag = enriched_metadata.get("online-flag", "false")
        available_flag = enriched_metadata.get("available-flag", "false")
        
        # Build product URL from product_id
        product_url = f"https://www.sephora.com/product/{product_id}" if product_id != "N/A" else "N/A"
        enriched_metadata["product_url"] = product_url
        
        # Build availability text (general and online)
        availability_text = "Available" if str(available_flag).lower() == "true" else "Not available"
        online_status = "Online" if str(online_flag).lower() == "true" else "Not online"
        
        # Build image markdown if available
        image_url = enriched_metadata.get("image-url")
        image_md = f"![Product Image](https://www.sephora.com/productimages/{image_url})" if image_url else "No image available"
        enriched_metadata["image_md"] = image_md
        
        # Build the prefix text that will be embedded with each chunk
        chunk_prefix = (
            f"Brand: {brand}\n"
            f"Product: {display_name} (ID: {product_id})\n"
            f"[View on Sephora]({product_url})\n"
            f"Availability: {availability_text} | {online_status}\n"
            f"Image: {image_md}\n\n"
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
#---------------------------------------- MAIN () ----------------------------------------
#=========================================================================================
def main(llm_model: str) -> None:
     
    logging.info(f"{my_name()}: Starting. ")
    
    if os.path.exists(VECTOR_STORE_PATH):
        # Check if vector store already exists
        print(f"{my_name()}: Found existing vector store. If you want it to be re-written, delete it :) ")
        exit(0)

    try:     
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        documents = load_documents_from_jsonl(JSONL_FILE)
        chunked_documents = chunk_documents(documents)
        vector_store = FAISS.from_documents(chunked_documents, embedding = embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
        logging.info(f"{my_name()}: Built and saved new vector store.")

    except Exception as e:
        logging.error(f"{my_name()}: Error: {e}")

if __name__ == "__main__":
    main()
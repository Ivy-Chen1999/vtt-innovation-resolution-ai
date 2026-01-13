import os, dotenv

from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from qdrant import VectorStore 

from log_config import logger

dotenv.load_dotenv()



DEFAULT_DIMS = 3072


def get_env_azure_config() -> Dict:
    azure_config = {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", ""),

        "chat_deployment": {
            "gpt_4_1": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_gpt_4_1", ""),
            "gpt_4o_mini": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_gpt_4o_mini", ""),
        },

        "embedding_deployment": {
            "large": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_LARGE", ""),
            "small": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_SMALL", "")  
        },

        "ai_search": {
            "azure_search_endpoint": os.getenv("AZURE_SEARCH_ENDPOINT", ""),
            "azure_search_key": os.getenv("AZURE_SEARCH_KEY", ""),
            "index_name": os.getenv("AZURE_SEARCH_INDEX_NAME", "")
        }
    }

    return azure_config


def get_openai_models(dimensions = DEFAULT_DIMS, temperature = 0):
    azure_config = get_env_azure_config()

    if not (azure_config["api_key"] and azure_config["azure_endpoint"] and azure_config["chat_deployment"]):
        logger.error("Failed to load Azure configuration.")
        return None, None, None, False    

    llm = AzureChatOpenAI(
        azure_endpoint=azure_config["azure_endpoint"],
        api_key=azure_config["api_key"],
        api_version=azure_config["api_version"],
        azure_deployment=azure_config["chat_deployment"]["gpt_4_1"],
        temperature=temperature
    )

    embedding_model = AzureOpenAIEmbeddings(
        azure_endpoint=azure_config["azure_endpoint"],
        api_key=azure_config["api_key"],
        api_version=azure_config["api_version"],

        azure_deployment=azure_config["embedding_deployment"]["large"],
        dimensions=dimensions
    )

    search_endpoint = azure_config["ai_search"]["azure_search_endpoint"]
    search_key = azure_config["ai_search"]["azure_search_key"]
    index_name = azure_config["ai_search"]["index_name"]

    if search_endpoint and search_key and index_name:
        logger.info("Using Azure AI Search as vector store.")
        vector_store = AzureSearch(
            azure_search_endpoint=search_endpoint,
            azure_search_key=search_key,
            index_name=index_name,
            embedding_model=embedding_model
        )
        using_local_store = False
    else:
        logger.warning("Azure AI Search config missing, using local vector store.")
        vector_store = VectorStore(embedding_model = embedding_model)
        using_local_store = True

    return llm, embedding_model, vector_store, using_local_store


def chat_bot(query: str) -> str:
    llm, embedding_model, vector_store, using_local_store = get_openai_models()

    if llm is None:
        return "[Error] LLM not available. Please check your configuration."
    if embedding_model is None:
        return "[Error] Embedding model unavailable. Please check your configuration."
    if vector_store is None:
        return "[Error] Vector store unavailable. Please check your configuration."

    chatbot_prompt = PromptTemplate.from_template("""
        You're a smart assistant helping extract insights from VTT innovation relationships.

        Context:
        {context}

        According to the context, answer this question:
        {question}
    """)

    chatbot_llm = chatbot_prompt | llm

    try:
        if using_local_store:
            result_texts, _ = vector_store.search(query)

            if(len(result_texts) == 0):
                return f"Sorry, There isn't any related result for {query}."
            
            context = "\n".join(result_texts)
        else:
            results = vector_store.vector_search(query, k = 5)
            context = "\n".join([res.metadata['source'] for res in results])
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        return "[Error] Vector search failed."

    try:
        llm_result = chatbot_llm.invoke({"context": context, "question": query})
        return llm_result.content
    except Exception as e:
        logger.error(f"LLM failed to process your question: {str(e)}")
        return f"[Error] LLM failed to process your question. Error: {str(e)}"

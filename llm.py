import streamlit as st 
from langchain_ollama import OllamaLLM, OllamaEmbeddings


# Create the LLM
llm = OllamaLLM(model="mistral")

# Create the Embedding model 
embeddings = OllamaEmbeddings(model="mistral")

""" # Create the LLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model=st.secrets["OPENAI_MODEL"],
)
# Create the Embedding model

embeddings = OpenAIEmbeddings(
    openai_api_key=st.secrets["OPENAI_API_KEY"]
) """
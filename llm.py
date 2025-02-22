import streamlit as st

from langchain_community.llms import Ollama
import streamlit as st

# Create the LLM
llm = Ollama(model="mistral")

# Create the Embedding model 
from langchain_community.embeddings import OllamaEmbeddings

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
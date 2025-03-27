from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
#from langchain_neo4j import Neo4jChatMessageHistory
#from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import get_session_id
from tools.prompts import get_movie_agent_prompt
from tools.vector import get_movie_plot

#import time
# from openai import RateLimitError

import streamlit as st

# Create a movie chat chain
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert providing information about movies."),
        ("human", "{input}"),
    ]
)

movie_chat = chat_prompt | llm | StrOutputParser()

# Create a set of tools
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general movie chat not covered by other tools",
        func=movie_chat.invoke,
    ), 
    Tool.from_function(
        name="Movie Plot Search",  
        description="For when you need to find information about movies based on a plot",
        func=get_movie_plot, 
    )
]

# Create chat history callback
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Create the agent
agent_prompt = get_movie_agent_prompt()
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
    )

# Create a handler to call the agent
chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Handle user's input
def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    try:
        response = chat_agent.invoke(
            {"input": user_input},
            {"configurable": {"session_id": get_session_id()}},)

        return response['output']
   # except RateLimitError:
        # st.warning("Rate limit exceeded. Please wait and try again.")
        # time.sleep(60)  # Wait for 60 seconds
        # return generate_response(user_input)  # Retry
    except Exception as e:
        return str(e)
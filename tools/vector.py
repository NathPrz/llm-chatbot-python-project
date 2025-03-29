import streamlit as st
from llm import llm, embeddings
from graph import graph
from langchain_neo4j import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Création de l'index si nécessaire
graph.query("DROP INDEX moviePlots IF EXISTS")

def create_vector_index(graph, embeddings):
    # Vérifier si l'index existe déjà
    index_query = """
    SHOW INDEXES
    WHERE name = 'moviePlots' AND type = 'VECTOR'
    """
    result = graph.query(index_query)
    
    if not result:
        # Créer l'index s'il n'existe pas
        create_index_query = """
        CREATE VECTOR INDEX moviePlots IF NOT EXISTS
        FOR (m:Movie) ON (m.plotEmbedding) 
        OPTIONS {indexConfig: {
            `vector.dimensions`: 4096,
            `vector.similarity_function`: 'cosine'
        }}
        """
        graph.query(create_index_query)
        print("Index 'moviePlots' créé avec succès.")
    else:
        print("Index 'moviePlots' existe déjà.")

# Appeler cette fonction avant d'utiliser from_existing_index
create_vector_index(graph, embeddings)

# Create the Neo4jVector
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              
    graph=graph,                             
    index_name="moviePlots",                 
    node_label="Movie",                     
    text_node_property="plot",               
    embedding_node_property="plotEmbedding", 
    retrieval_query="""
RETURN
    node.plot AS text,
    score,
    {
        title: node.title,
        directors: [ (person)-[:DIRECTED]->(node) | person.name ],
        actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
        tmdbId: node.tmdbId,
        source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
    } AS metadata
"""
)

# Create the retriever
retriever = neo4jvector.as_retriever()

# Create the prompt
instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

# Create the chain 
question_answer_chain = create_stuff_documents_chain(llm, prompt)
plot_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)

# Create a function to call the chain
def get_movie_plot(input):
    return plot_retriever.invoke({"input": input})
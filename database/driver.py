#driver connected to neo4j db that will craete the actual graph for RAG

from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

#simlpy loads the databse and allows for queries
#actual "query" building will be done by constructor agent

load_dotenv()

class Neo4jHandler:
    #class for the interface with neo4j database
    def __init__(self):
        #connect to server by providing credentials
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")

        #establish a connection
        self.driver = GraphDatabase.driver(self.uri, auth = (self.user, self.password))

    def close(self):
        #terminate the connection to db
        self.driver.close()
    
    def run_query(self, query, parameters = None):
        #standard method to execute cypher commands
        with self.driver.session() as session:
            result = session.run(query, parameters)
            #return a list of dicts so the Agents can easily associate term to info
            return [record.data() for record in result]

#instantiate a global instance for the driver (so we dont have to keep making new ones)
db = Neo4jHandler()
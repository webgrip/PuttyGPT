---
parent: Decisions
nav_order: 101
title: ADR 2: Weaviate for Knowledge Graph and Vector Storage

status: proposed
date: 2023-04-14
deciders: [Developer Team which consists of the top 5% of their field]
consulted: [Top 5% of their field AI Experts, Top 5% of their field Solution Architect, Top 5% of their field Software Engineers, top 5% DevOps Engineers, hundreds of 10x individuals, Developer team, The SecOps team]
informed: [Whomever it may concern]
Weaviate for Knowledge Graph and Vector Storage
---

Weaviate for Knowledge Graph and Vector Storage
===============================================

Context and Problem Statement
-----------------------------

The solution requires an efficient and scalable way to store and query knowledge graphs and vector data. How can we enable effective storage and retrieval of knowledge graph and vector data while ensuring the maintainability and scalability of the architecture?

Decision Drivers
----------------

Overall
-------

*   Privacy oriented, we need to adhere to ISO standards
*   Free usage
*   Stability
*   Cutting edge tech and integrations

Component specific
------------------

*   Efficient storage and retrieval of knowledge graph and vector data
*   Scalability
*   Maintainability of the architecture

Considered Options
------------------

*   Weaviate
*   Azure Digital Twins
*   Dgraph
*   Neo4j
*   ArangoDB

Decision Outcome
----------------

Chosen option: "Weaviate"

It is specifically designed for knowledge graph and vector storage, providing the best combination of efficient storage and retrieval, scalability, and maintainability for the solution.

### Consequences

*   Good, because Weaviate enables efficient storage and retrieval of knowledge graph and vector data
*   Good, because Weaviate provides a scalable architecture
*   Bad, because maintaining and updating Weaviate may require significant effort and resources

### Implementation Example 1: Weaviate and Langchain

To demonstrate the integration of Weaviate for knowledge graph and vector storage, a simple proof of concept (POC) can be implemented. For example, this POC could involve connecting Weaviate to the Langchain component communication system. A short code snippet for this POC might look like:

```python
from weaviate import WeaviateClient
from langchain import Chain

# Initialize WeaviateClient and Langchain
weaviate_client = WeaviateClient("http://localhost:8080")
chain = Chain()

# Add Weaviate as a component in Langchain
chain.add_component("weaviate", weaviate_client)

# Store knowledge graph data in Weaviate
data = {
    "name": "Example entity",
    "description": "An example entity in the knowledge graph"
}
chain.components["weaviate"].create(data)

```


### Implementation Example 2: Weaviate and ChatGPT plugins

Another example of how Weaviate can be integrated with other components is through the use of ChatGPT plugins. These plugins can be built to leverage Weaviate's capabilities for knowledge graph and vector data storage. For instance, a ChatGPT plugin could query Weaviate to retrieve relevant information for a user's question:

```python
from weaviate import WeaviateClient
from chatgpt import ChatGPT

# Initialize WeaviateClient and ChatGPT
weaviate_client = WeaviateClient("http://localhost:8080")
chatgpt = ChatGPT()

# Define a ChatGPT plugin to query Weaviate
def weaviate_query_plugin(query):
    # Execute a query in Weaviate
    result = weaviate_client.query(query)

    # Process the result and return a response
    response = process_weaviate_result(result)
    return response

# Register the plugin with ChatGPT
chatgpt.register_plugin("weaviate_query", weaviate_query_plugin)

```

### Synergy with Other Solutions

Weaviate can easily integrate with Langchain, ChatGPT plugins, and other proposed components of the solution. This integration allows for efficient storage and retrieval of knowledge graph and vector data and provides a modular architecture for future extension or modification.

Validation
----------

The implementation of Weaviate in the solution will be validated by creating a proof of concept (POC) that demonstrates the efficient storage and retrieval of knowledge graph and vector data. The POC will be reviewed by the developer team, solution architect, and AI experts to ensure it meets the requirements for efficient storage and retrieval, scalability, and maintainability.

Pros and Cons of the Options
----------------------------

### Weaviate

*   Good, because it is specifically designed for knowledge graph and vector storage
*   Good, because it provides a scalable architecture
*   Good, because it enables efficient storage and retrieval of knowledge graph and vector data
*   Neutral, because it requires some setup and configuration for optimal performance
*   Bad, because maintaining and updating Weaviate may require significant effort and resources

### Azure Digital Twins

*   Good, because it is a Microsoft Azure service, which may provide seamless integration with other Azure services
*   Good, because it supports knowledge graph storage and querying
*   Neutral, because it may not be as efficient for vector storage and retrieval as Weaviate
*   Bad, because it is not specifically designed for AI and NLP component communication, which may result in additional customization and development effort
*   Bad, because it may require additional setup and configuration, as well as reliance on Microsoft Azure

### Dgraph

*   Good, because it supports GraphQL, which provides a powerful and flexible query language
*   Neutral, because it is primarily designed for graph storage, not specifically vector storage
*   Bad, because it may require additional customization and development effort for AI and NLP component communication
*   Bad, because maintaining and updating Dgraph may require significant effort and resources

### Neo4j

*   Good, because it is a popular and widely used graph database
*   Good, because it supports Cypher, a powerful and expressive graph query language
*   Neutral, because it is primarily designed for graph storage, not specifically vector storage
*   Bad, because it may require additional customization and development effort for AI and NLP component communication
*   Bad, because maintaining and updating Neo4j may require significant effort and resources

### ArangoDB

*   Good, because it is a multi-model database, supporting graph, document, and key-value data models
*   Good, because it provides flexible querying options, including AQL, a powerful query language
*   Neutral, because it is not specifically designed for vector storage and retrieval
*   Bad, because it may require additional customization and development effort for AI and NLP component communication
*   Bad, because maintaining and updating ArangoDB may require significant effort and resources

More Information
----------------

The decision outcome is based on the evaluation of the considered options against the decision drivers.

The implementation of Weaviate will be validated through a proof of concept, and the decision may be revisited if requirements change or new solutions emerge.

The development team and consultants have agreed on the choice of Weaviate for knowledge graph and vector storage.
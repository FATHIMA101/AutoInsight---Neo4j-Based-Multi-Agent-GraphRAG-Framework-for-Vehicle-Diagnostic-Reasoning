# AutoInsight---Neo4j-Based-Multi-Agent-GraphRAG-Framework-for-Vehicle-Diagnostic-Reasoning
AutoInsight is an intelligent assistant that converts raw CAN data into structured insights using a Neo4j-based knowledge graph and multi-agent system. It enables users to query vehicle behavior in natural language without deep protocol knowledge, making diagnostics faster, accurate, and accessible.
# Overview of Controller Area Network (CAN)
The Controller Area Network (CAN) is a reliable communication protocol developed by Bosch, widely used in modern, especially autonomous and electric vehicles. It enables real-time communication between Electronic Control Units (ECUs), which manage key vehicle functions like engine control, braking, and battery systems.

CAN data consists of compact, encoded messages that are efficient for machines but difficult for humans to interpret. These messages use identifiers (CAN IDs) and contain multiple signals packed into a few bytes. Decoding them requires DBC (Database CAN) files, which define how signals are extracted using bit positions, scaling factors, and offsets.

In autonomous vehicles, the volume of CAN data is significantly higher due to continuous sensor and actuator communication. This makes manual analysis complex and time-consuming, especially for junior engineers or non-experts.

ECUs operate in real time and constantly exchange critical data via CAN. As vehicle systems grow more connected and automated, understanding CAN data becomes essential for diagnostics, safety, and performance—requiring advanced tools to simplify interpretation.

# GraphRAG (Graph based Retrieval Augmented Generation)
GraphRAG (Graph based Retrieval Augmented Generation) is a method that combines graph databases with language models to improve how information is retrieved and used in responses. Instead of relying only on text search, GraphRAG organizes data as a network of connected nodes representing concepts, entities, or documents. This structure helps the model understand relationships and context more effectively. When a query is made, it retrieves relevant nodes and their connections, allowing for more accurate, contextual, and explainable answers. It is often used in applications like knowledge management, research analysis, and intelligent search systems.

# Role of Multi-Agent Systems in Intelligent Vehicle Assistants
Agents are autonomous software entities that perceive their environment, make decisions, and take actions to achieve specific goals. In a multi-agent system (MAS), several such agents work together, each handling a specialized task while collaborating to solve complex problems more efficiently than a single monolithic system.

MAS architectures offer modularity, scalability, and fault tolerance, making them ideal for complex, distributed domains like robotics, smart grids, and increasingly, intelligent vehicle systems. Each agent can function independently, ensuring that the system remains robust even if one component fails.

In our vehicle diagnostic assistant, the multi-agent approach enables efficient handling of complex user queries. Individual agents are responsible for tasks like natural language processing, semantic retrieval, graph exploration, and response generation. Together, they form a collaborative system that interprets user input and returns human-readable insights from CAN data.

This design allows even non-experts to interact with the system intuitively, without needing deep knowledge of CAN protocols or graph databases, making advanced diagnostics more accessible.

# Dataset Description
The dataset used in this project was collected from a real world electric vehicle specifically a Tesla
Model 3 and contains CAN log files in .MF4 format. These logs capture binary encoded messages
transmitted over the vehicle’s internal Controller Area Network (CAN) during various driving
sessions. Along with the logs, a DBC (Database CAN) file is provided, which serves as a decoding
schema by mapping CAN IDs to signal names, scaling factors, and physical units.
The dataset was selected due to its richness in signal diversity, realistic driving conditions, and
compatibility with diagnostic analysis. It serves as the foundational input for the entire diagnostic
pipeline from CAN decoding to knowledge graph construction and semantic query resolution.

# CAN Data Preprocessing and Decoding
The diagnostic process starts with decoding raw CAN data, typically stored in the standardized .MF4 binary format, commonly used in automotive applications. This format, defined by ASAM, contains time-stamped messages recorded during real-world driving, such as logs from a Tesla Model 3.

To interpret this data, a DBC (Database CAN) file is used. It defines how binary messages map to human-readable signals through message names, signal definitions, scaling factors, offsets, units, and valid ranges. It also outlines the byte structure of each signal within a message.

During preprocessing, signals undergo scaling to real-world values, offset correction, range validation, and contextual labeling with physical units (e.g., °C, kph). Each message is accurately timestamped to maintain the temporal sequence.

The decoded output is structured into a tabular format with fields like Timestamp, CAN ID, Message Name, Signal, Physical Value, Raw Value, and Unit. This cleaned dataset forms the foundation for building the semantic knowledge graph used in further diagnostics.
<img width="820" height="457" alt="can architecture" src="https://github.com/user-attachments/assets/97f30677-d8ed-4a0f-a43f-3ec47bf7908d" />


# Knowledge Graph Construction using Neo4j
Neo4j is a highly scalable and efficient graph database designed to store, manage, and query data
structured as interconnected nodes and relationships. Unlike traditional relational databases that
organize data into rows and tables, Neo4j uses a graph-based model where each piece of information
is stored as a node (such as a signal, message, or unit), and the connections between them are
represented as relationships (like "HAS_SIGNAL" or "OBSERVED_AT"). This model closely
mirrors the way data is naturally connected in the real world, making it ideal for applications
involving complex relationships such as vehicle communication systems.

Once the CAN data is decoded, it is structured into a semantic Knowledge Graph using Neo4j, a
highly scalable graph database. Neo4j offers a native graph processing engine and supports Cypher,
a declarative query language designed for flexible traversal of graph patterns.

In this system, each entity in the vehicle communication stream such as messages, signals,
timestamps, and units is represented as a node, while their associations are represented as edges.
This approach reflects the real world relational nature of vehicular data, enabling semantic in-
terpretation, reasoning, and efficient query execution. This structured knowledge graph enables efficient and explainable reasoning across interconnected vehicle data, allowing intelligent agents
to traverse, search, and analyze relationships with ease.
<img width="642" height="505" alt="Knowledge graph" src="https://github.com/user-attachments/assets/36aa5a38-26d1-47a0-a769-f097744d414e" />


# Multi-Agent System Design
To manage the complexity of vehicle diagnostics and ensure modular, intelligent interactions, the
system is implemented using a multi-agent architecture. Each agent is a specialized software com-
ponent designed to handle a specific role within the pipeline, coordinated by a central Orchestrator
Agent.
Agents and Their Roles:

– Orchestrator Agent: Acts as the central controller, managing the end-to-end session flow. It
invokes other agents in sequence, maintains context across user queries, and integrates their
outputs to generate coherent responses.

– Query Planner Agent: Interprets the user’s natural language input to extract intent and relevant
entities. It uses regular expressions and keyword patterns to classify query types (e.g., temporal,
relational) and extract signal names, units, or timestamps. The output is a structured "intent
profile" that guides subsequent processing.

– Semantic Retriever Agent: Performs semantic search to match user queries with the most
relevant entities in the knowledge graph. It utilizes the MiniLM-L6-v2 model from Sentence
Transformers to generate 384-dimensional sentence embeddings and applies FAISS (Facebook
AI Similarity Search) for efficient nearest-neighbor retrieval using L2 distance metrics. This
agent enables fuzzy, meaning-based matching even when the user’s query does not exactly
match the graph labels.

– Graph Explorer Agent: Constructs and executes dynamic Cypher queries to interact with the
Neo4j graph. It retrieves relevant nodes, relationships, and metadata based on the planner’s
intent and the retriever’s matched entities. It also performs temporal reasoning where needed.
– Response Synthesizer Agent: Converts structured query results into a fluent, human-readable
response using the Gemini LLM API. It integrates metadata such as values, timestamps, and
signal labels to produce coherent explanations tailored to user needs.

This agent-based system mirrors collaborative human problem-solving, where each agent con-
tributes domain-specific knowledge and works in sequence to achieve a shared diagnostic goal.
<img width="733" height="482" alt="Multi agent architecture" src="https://github.com/user-attachments/assets/d834ee5a-b9fb-46b1-925d-765ca4346009" />

# Result
# Decoded CAN Data
<img width="752" height="277" alt="result 1" src="https://github.com/user-attachments/assets/4fdc0c9d-d2f7-4fed-87b6-a8579bc26a00" />

# Knowledge Graph created using Decoded CAN Data
<img width="547" height="478" alt="result 2" src="https://github.com/user-attachments/assets/e2c979ce-ac38-4ec6-9949-6e54b933e831" />
<img width="508" height="512" alt="result 3" src="https://github.com/user-attachments/assets/18338f2c-4c0d-4857-beea-9ab38ee6a48b" />

# Queries and Response with Answer Relevence
<img width="735" height="775" alt="result 4" src="https://github.com/user-attachments/assets/29cea984-fe34-4b8f-b03e-29ed25bc02ac" />




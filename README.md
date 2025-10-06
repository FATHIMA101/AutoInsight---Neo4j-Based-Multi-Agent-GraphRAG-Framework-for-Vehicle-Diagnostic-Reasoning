# AutoInsight---Neo4j-Based-Multi-Agent-GraphRAG-Framework-for-Vehicle-Diagnostic-Reasoning
AutoInsight is an intelligent assistant that converts raw CAN data into structured insights using a Neo4j-based knowledge graph and multi-agent system. It enables users to query vehicle behavior in natural language without deep protocol knowledge, making diagnostics faster, accurate, and accessible.
# Overview of Controller Area Network (CAN)
The Controller Area Network (CAN) is a reliable communication protocol developed by Bosch, widely used in modern, especially autonomous and electric vehicles. It enables real-time communication between Electronic Control Units (ECUs), which manage key vehicle functions like engine control, braking, and battery systems.

CAN data consists of compact, encoded messages that are efficient for machines but difficult for humans to interpret. These messages use identifiers (CAN IDs) and contain multiple signals packed into a few bytes. Decoding them requires DBC (Database CAN) files, which define how signals are extracted using bit positions, scaling factors, and offsets.

In autonomous vehicles, the volume of CAN data is significantly higher due to continuous sensor and actuator communication. This makes manual analysis complex and time-consuming, especially for junior engineers or non-experts.

ECUs operate in real time and constantly exchange critical data via CAN. As vehicle systems grow more connected and automated, understanding CAN data becomes essential for diagnostics, safety, and performance—requiring advanced tools to simplify interpretation.

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

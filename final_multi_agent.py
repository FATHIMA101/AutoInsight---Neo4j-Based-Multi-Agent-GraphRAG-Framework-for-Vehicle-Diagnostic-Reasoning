#--- START OF FILE multi_agent_2.py ---

#--- START OF FILE multi_agent.py ---

import re
import json
import time
from datetime import datetime, timezone, timedelta
import pytz
from typing import List, Dict, Optional, Tuple, Any
from neo4j import GraphDatabase, graph
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai
from dateutil import parser

# ==============================================================================
# 1. SPECIALIZED AGENTS
# ==============================================================================

class QueryPlannerAgent:
    """
    Analyzes the user's query to understand its intent and extract key parameters.
    This agent replaces the `extract_query_intent` function.
    """
    def __init__(self, query_patterns: Dict):
        self.query_patterns = query_patterns
        print("✅ Query Planner Agent initialized.")

    def plan(self, query: str) -> Dict:
        """Determines the intent and parameters of a user query."""
        intent = {
            'type': 'general',
            'entities': [],
            'parameters': {},
            'modifiers': [],
            'temporal_type': None
        }
        query_lower = query.lower()

        for pattern_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower, re.IGNORECASE)
                if match:
                    intent['type'] = pattern_type
                    intent['entities'] = list(match.groups())
                    if pattern_type == 'temporal':
                        if 'before' in query_lower:
                            intent['temporal_type'] = 'before'
                        elif 'after' in query_lower:
                            intent['temporal_type'] = 'after'
                        elif 'latest' in query_lower:
                            intent['temporal_type'] = 'latest'
                        elif 'earliest' in query_lower:
                            intent['temporal_type'] = 'earliest'
                        elif 'sequence' in query_lower:
                            intent['temporal_type'] = 'sequence'
                        elif 'duration' in query_lower:
                            intent['temporal_type'] = 'duration'
                        elif 'frequency' in query_lower:
                            intent['temporal_type'] = 'frequency'
                        elif 'gap' in query_lower or 'time difference' in query_lower or 'interval' in query_lower:
                            intent['temporal_type'] = 'gap'
                        elif 'between' in query_lower or ('from' in query_lower and 'to' in query_lower):
                            intent['temporal_type'] = 'range'
                        elif 'since' in query_lower:
                            intent['temporal_type'] = 'since'
                        elif 'until' in query_lower:
                            intent['temporal_type'] = 'until'
                        elif 'exactly' in query_lower or 'precisely' in query_lower:
                            intent['temporal_type'] = 'exact'
                        elif 'ago' in query_lower:
                            intent['temporal_type'] = 'relative'
                        elif 'when' in query_lower and ('message' in query_lower or 'signal' in query_lower) and ('sent' in query_lower or 'send' in query_lower):
                            intent['temporal_type'] = 'when_sent'
                        else:
                            intent['temporal_type'] = 'when'
                    break
            if intent['type'] != 'general':
                break

        if any(word in query_lower for word in ['all', 'every', 'complete', 'full']):
            intent['modifiers'].append('comprehensive')
        if any(word in query_lower for word in ['count', 'number', 'how many']):
            intent['modifiers'].append('count')
        if any(word in query_lower for word in ['recent', 'latest', 'new']):
            intent['modifiers'].append('recent')
        if intent['type'] == 'physical_value':
            intent['modifiers'].append('physical_value')

        return intent

class SemanticRetrieverAgent:
    """
    Uses vector embeddings to find entities in the graph that are semantically
    related to the user's query.
    """
    def __init__(self, embedding_model, all_entities, entity_embeddings, faiss_index):
        self.embedding_model = embedding_model
        self.all_entities = all_entities
        self.entity_embeddings = entity_embeddings
        self.faiss_index = faiss_index
        print("✅ Semantic Retriever Agent initialized.")

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Finds relevant entities using semantic search."""
        if not self.all_entities or self.faiss_index is None:
            return []
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        if query_embedding.shape[0] == 0:
            return []
        distances, indices = self.faiss_index.search(query_embedding, k=min(k, len(self.all_entities)))
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.all_entities):
                entity = self.all_entities[idx]
                # Ensure distance is a standard float for calculations
                confidence = 1.0 / (1.0 + float(distance))
                results.append((entity, confidence))
        return results

class GraphExplorerAgent:
    """
    Interacts with the Neo4j database by constructing and executing Cypher queries.
    This agent encapsulates all graph-specific logic.
    """
    def __init__(self, driver: GraphDatabase.driver, temporal_fields: List[Dict]):
        self.driver = driver
        self.temporal_fields = temporal_fields
        print("✅ Graph Explorer Agent initialized.")

    def parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Enhanced timestamp parsing with better error handling."""
        if not timestamp_str:
            return None
        
        try:
            return datetime.fromisoformat(str(timestamp_str))
        except (ValueError, TypeError):
            pass
        
        try:
            return datetime.fromisoformat(str(timestamp_str).replace(' ', 'T'))
        except (ValueError, TypeError):
            pass
        
        try:
            ts_str = str(timestamp_str)
            return datetime.fromisoformat(ts_str.split('.')[0] + ts_str[-6:])
        except (ValueError, TypeError, IndexError):
            pass
        
        try:
            return datetime.strptime(str(timestamp_str).split('+')[0].strip(), '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            pass
        
        try:
            return parser.parse(str(timestamp_str))
        except (parser.ParserError, TypeError):
            return None

    def explore(self, intent: Dict, relevant_entities: List[Tuple[Dict, float]]) -> List[Dict]:
        """Executes a query against the graph based on intent."""
        with self.driver.session() as session:
            results = []
            try:
                if intent['type'] == 'can_id' and intent['entities']:
                    results = self._query_can_id_comprehensive(session, intent['entities'][0])
                elif intent['type'] == 'node_type' and intent['entities']:
                    results = self._query_by_node_type(session, intent['entities'][0].title(), intent['modifiers'])
                elif intent['type'] == 'relationship' and intent['entities']:
                    results = self._query_relationships(session, intent['entities'][0])
                elif intent['type'] == 'temporal' and intent['entities']:
                    results = self._execute_temporal_query(session, intent, relevant_entities)
                elif intent['type'] == 'aggregation' and intent['entities']:
                    results = self._query_aggregation(session, intent['entities'][0], intent['modifiers'])
                elif intent['type'] == 'properties' and intent['entities']:
                    results = self._query_properties(session, intent['entities'][0])
                elif intent['type'] == 'physical_value' and intent['entities']:
                    results = self._query_physical_value(session, intent['entities'][0])
                else:
                    if relevant_entities:
                        results = self._query_by_semantic_match(session, relevant_entities)
            except Exception as e:
                print(f"❌ Graph Explorer Agent error during execution: {e}")
            return results

    def _execute_query_list(self, session, queries: List[str]) -> List[Dict]:
        results = []
        for query in queries:
            try:
                result = session.run(query)
                records = [dict(record) for record in result]
                if records:
                    results.extend(records)
            except Exception:
                # This error is expected for some queries that might not match the schema, so we can ignore it.
                continue
        return results

    def _execute_temporal_query(self, session, intent: Dict, relevant_entities: List[Tuple[Dict, float]]) -> List[Dict]:
        temporal_type = intent.get('temporal_type')
        entities = intent.get('entities', [])
        if not self.temporal_fields:
            print("⚠️ No temporal fields discovered in the database. Cannot execute temporal query.")
            return []
            
        if temporal_type == 'when_sent' and entities:
            return self._query_when_sent_enhanced(session, entities[0])
        elif temporal_type == 'before' and entities:
            return self._query_before_message_enhanced(session, entities[0])
        elif temporal_type == 'after' and entities:
            return self._query_after_message_enhanced(session, entities[0])
        elif temporal_type == 'sequence' and entities:
            return self._query_message_sequence_enhanced(session, entities[0])
        elif temporal_type == 'range' and len(entities) >= 2:
            return self._query_temporal_range_enhanced(session, entities[0], entities[1])
        elif temporal_type == 'duration' and entities:
            return self._query_duration_enhanced(session, entities[0])
        elif temporal_type == 'frequency' and entities:
            return self._query_frequency_enhanced(session, entities[0])
        elif temporal_type == 'gap' and len(entities) >= 2:
            return self._query_gap_enhanced(session, entities[0], entities[1])
        elif temporal_type == 'latest' and entities:
            return self._query_latest_enhanced(session, entities[0])
        elif temporal_type == 'earliest' and entities:
            return self._query_earliest_enhanced(session, entities[0])
        elif temporal_type == 'since' and entities:
            return self._query_since_enhanced(session, entities[0])
        elif temporal_type == 'until' and entities:
            return self._query_until_enhanced(session, entities[0])
        elif temporal_type == 'exact' and entities:
            return self._query_exact_enhanced(session, entities[0])
        elif temporal_type == 'relative' and entities:
            return self._query_relative_enhanced(session, entities[0])
        return []

    def _query_can_id_comprehensive(self, session, can_id: str) -> List[Dict]:
        queries = [
            f"""
            MATCH (can)
            WHERE toString(can.id) = '{can_id}' OR toString(can.can_id) = '{can_id}' OR toString(can.canid) = '{can_id}'
            OPTIONAL MATCH (can)-[:HAS_MESSAGE]->(m)-[:HAS_SIGNAL]->(s)
            OPTIONAL MATCH (s)-[:MEASURED_AS]->(pv:PhysicalValue)
            OPTIONAL MATCH (s)-[:HAS_UNIT]->(u:Unit)
            RETURN can, m, s, pv.value AS physical_value, u.name AS unit,
                   labels(can) as can_labels, labels(m) as m_labels, labels(s) as s_labels
            LIMIT 50
            """,
            f"""
            MATCH (n)
            WHERE ANY(prop IN keys(n) WHERE toString(n[prop]) CONTAINS '{can_id}')
            OPTIONAL MATCH (n)-[:HAS_SIGNAL]->(s)-[:MEASURED_AS]->(pv:PhysicalValue)
            OPTIONAL MATCH (s)-[:HAS_UNIT]->(u:Unit)
            RETURN n, s, pv.value AS physical_value, u.name AS unit,
                   labels(n) as n_labels, labels(s) as s_labels
            LIMIT 30
            """
        ]
        return self._execute_query_list(session, queries)

    def _query_by_node_type(self, session, node_type: str, modifiers: List[str]) -> List[Dict]:
        limit = "LIMIT 100" if 'comprehensive' in modifiers else "LIMIT 20"
        queries = [
            f"""
            MATCH (n:{node_type})
            OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
            OPTIONAL MATCH (n)-[:OBSERVED_AT]->(t:Timestamp)
            OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n, pv.value AS physical_value, t.value AS timestamp,
                   toString(pv.value) AS physical_value_str, toString(t.value) AS timestamp_str,
                   u.name AS unit, r, m, labels(n) AS n_labels, labels(m) AS m_labels, type(r) AS rel_type
            {limit}
            """,
            f"""
            MATCH (n) WHERE '{node_type.lower()}' IN [label IN labels(n) | toLower(label)]
            OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
            OPTIONAL MATCH (n)-[:OBSERVED_AT]->(t:Timestamp)
            OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n, pv.value AS physical_value, t.value AS timestamp,
                   toString(pv.value) AS physical_value_str, toString(t.value) AS timestamp_str,
                   u.name AS unit, r, m, labels(n) AS n_labels, labels(m) AS m_labels, type(r) AS rel_type
            {limit}
            """
        ]
        return self._execute_query_list(session, queries)

    def _query_relationships(self, session, entity: str) -> List[Dict]:
        queries = [
            f"""
            MATCH (n)-[r]-(m)
            WHERE n.name CONTAINS '{entity}' OR n.id CONTAINS '{entity}'
            OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
            OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
            RETURN n, r, m, pv.value AS physical_value, u.name AS unit,
                   labels(n) as n_labels, labels(m) as m_labels, type(r) as rel_type
            LIMIT 30
            """,
            f"""
            MATCH (n)-[r:{entity.upper()}]-(m)
            OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
            OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
            RETURN n, r, m, pv.value AS physical_value, u.name AS unit,
                   labels(n) as n_labels, labels(m) as m_labels, type(r) as rel_type
            LIMIT 30
            """
        ]
        return self._execute_query_list(session, queries)

    def _query_aggregation(self, session, entity: str, modifiers: List[str]) -> List[Dict]:
        if 'count' in modifiers:
            queries = [
                f"""
                MATCH (n:{entity.title()})
                RETURN count(n) as total_count, '{entity}' as entity_type
                """,
                f"""
                MATCH (n) WHERE ANY(label IN labels(n) WHERE toLower(label) CONTAINS '{entity.lower()}')
                RETURN count(n) as total_count, labels(n) as entity_labels
                """
            ]
        else: # For other aggregations, we return samples for the LLM to process
            queries = [
                f"""
                MATCH (n) WHERE n.name CONTAINS '{entity}' OR ANY(prop IN keys(n) WHERE toLower(prop) CONTAINS '{entity.lower()}')
                OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
                OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                RETURN n, pv.value AS physical_value, u.name AS unit, labels(n) as labels, keys(n) as properties
                LIMIT 20
                """
            ]
        return self._execute_query_list(session, queries)

    def _query_properties(self, session, entity: str) -> List[Dict]:
        queries = [
            f"""
            MATCH (n) WHERE n.name = '{entity}' OR n.id = '{entity}'
            OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
            OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
            RETURN n, pv.value AS physical_value, u.name AS unit, labels(n) as labels, properties(n) as all_properties
            LIMIT 10
            """,
            f"""
            MATCH (n) WHERE n.name CONTAINS '{entity}' OR toString(n.id) CONTAINS '{entity}'
            OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
            OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
            RETURN n, pv.value AS physical_value, u.name AS unit, labels(n) as labels, properties(n) as all_properties
            LIMIT 10
            """
        ]
        return self._execute_query_list(session, queries)
        
    def _query_by_semantic_match(self, session, relevant_entities: List[Tuple[Dict, float]]) -> List[Dict]:
        results = []
        for entity, confidence in relevant_entities[:3]:
            identifier = entity.get('identifier', '')
            label = entity.get('label', '')
            entity_type = entity.get('type', '')

            if not identifier:
                continue

            if entity_type == 'node' and label:
                query = f"""
                MATCH (n:{label}) WHERE n.name = '{identifier}' OR toString(n.id) = '{identifier}' OR n.value = '{identifier}'
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN n, r, m, labels(n) as n_labels, labels(m) as m_labels, type(r) as rel_type
                LIMIT 15
                """
            elif entity_type == 'relationship':
                query = f"""
                MATCH (n)-[r:{identifier}]-(m)
                RETURN n, r, m, labels(n) as n_labels, labels(m) as m_labels, type(r) as rel_type
                LIMIT 15
                """
            else:
                continue
            
            try:
                result = session.run(query)
                records = [dict(record) for record in result]
                results.extend(records)
            except Exception:
                continue
        return results

    def _query_when_sent_enhanced(self, session, entity: str) -> List[Dict]:
        queries = []
        # Query for Message nodes with related Signals and PhysicalValues
        queries.append(f"""
        MATCH (m:Message)
        WHERE m.name = '{entity}' OR toLower(m.name) = toLower('{entity}') OR toString(m.id) = '{entity}'
        MATCH (m)-[:OBSERVED_AT]->(t:Timestamp)
        OPTIONAL MATCH (m)-[:HAS_SIGNAL]->(s:Signal)
        OPTIONAL MATCH (s)-[:MEASURED_AS]->(pv:PhysicalValue)
        OPTIONAL MATCH (s)-[:HAS_UNIT]->(u:Unit)
        WHERE t.value IS NOT NULL
        RETURN t.value AS send_time,
               toString(t.value) AS send_time_str,
               s.name AS signal_name,
               pv.value AS physical_value,
               toString(pv.value) AS physical_value_str,
               u.name AS unit,
               s AS signal,
               pv AS physical_value_node
        ORDER BY t.value ASC
        LIMIT 10
        """)
        # Query for Signal nodes directly with PhysicalValues
        queries.append(f"""
        MATCH (s:Signal)
        WHERE s.name = '{entity}' OR toLower(s.name) = toLower('{entity}') OR toString(s.id) = '{entity}'
        MATCH (s)-[:OBSERVED_AT]->(t:Timestamp)
        OPTIONAL MATCH (s)-[:MEASURED_AS]->(pv:PhysicalValue)
        OPTIONAL MATCH (s)-[:HAS_UNIT]->(u:Unit)
        WHERE t.value IS NOT NULL
        RETURN t.value AS send_time,
               toString(t.value) AS send_time_str,
               s.name AS signal_name,
               pv.value AS physical_value,
               toString(pv.value) AS physical_value_str,
               u.name AS unit,
               s AS signal,
               pv AS physical_value_node
        ORDER BY t.value ASC
        LIMIT 10
        """)
        # Generic query for any node with temporal properties and PhysicalValues
        queries.append(f"""
        MATCH (n)
        WHERE n.name = '{entity}' OR toLower(n.name) = toLower('{entity}') OR toString(n.id) = '{entity}'
        OPTIONAL MATCH (n)-[:OBSERVED_AT]->(t:Timestamp)
        OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
        OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
        WHERE t.value IS NOT NULL
        RETURN t.value AS send_time,
               toString(t.value) AS send_time_str,
               n.name AS name,
               pv.value AS physical_value,
               toString(pv.value) AS physical_value_str,
               u.name AS unit,
               n AS node,
               pv AS physical_value_node
        ORDER BY t.value ASC
        LIMIT 10
        """)
        return self._execute_query_list(session, queries)

    def _query_physical_value(self, session, entity: str) -> List[Dict]:
        queries = [
            f"""
            MATCH (s:Signal)
            WHERE s.name = '{entity}' OR toLower(s.name) = toLower('{entity}') OR toString(s.id) = '{entity}'
            OPTIONAL MATCH (s)-[:MEASURED_AS]->(pv:PhysicalValue)
            OPTIONAL MATCH (s)-[:OBSERVED_AT]->(t:Timestamp)
            OPTIONAL MATCH (s)-[:HAS_UNIT]->(u:Unit)
            RETURN s, pv.value AS physical_value, t.value AS timestamp,
                   toString(pv.value) AS physical_value_str, toString(t.value) AS timestamp_str,
                   u.name AS unit, labels(s) AS labels
            ORDER BY t.value DESC
            LIMIT 10
            """,
            f"""
            MATCH (s:Signal)
            WHERE s.name CONTAINS '{entity}' OR toString(s.id) CONTAINS '{entity}'
            OPTIONAL MATCH (s)-[:MEASURED_AS]->(pv:PhysicalValue)
            OPTIONAL MATCH (s)-[:OBSERVED_AT]->(t:Timestamp)
            OPTIONAL MATCH (s)-[:HAS_UNIT]->(u:Unit)
            RETURN s, pv.value AS physical_value, t.value AS timestamp,
                   toString(pv.value) AS physical_value_str, toString(t.value) AS timestamp_str,
                   u.name AS unit, labels(s) AS labels
            ORDER BY t.value DESC
            LIMIT 10
            """
        ]
        return self._execute_query_list(session, queries)

    def _query_before_message_enhanced(self, session, entity: str) -> List[Dict]:
        queries = []
        for field in self.temporal_fields:
            queries.extend([
                f"""
                MATCH (target) WHERE (target.name CONTAINS '{entity}' OR toString(target.id) CONTAINS '{entity}') AND target.{field['property']} IS NOT NULL
                WITH target.{field['property']} as target_time LIMIT 1
                MATCH (n) WHERE n.{field['property']} IS NOT NULL AND datetime(n.{field['property']}) < datetime(target_time)
                OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
                OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                RETURN n, n.{field['property']} as timestamp, target_time, labels(n) as labels,
                       toString(n.{field['property']}) as timestamp_str,
                       duration.between(datetime(n.{field['property']}), datetime(target_time)) as time_diff,
                       pv.value AS physical_value, u.name AS unit
                ORDER BY timestamp DESC LIMIT 5
                """
            ])
        return self._execute_query_list(session, queries)

    def _query_after_message_enhanced(self, session, entity: str) -> List[Dict]:
        queries = []
        for field in self.temporal_fields:
            queries.extend([
                f"""
                MATCH (target) WHERE (target.name CONTAINS '{entity}' OR toString(target.id) CONTAINS '{entity}') AND target.{field['property']} IS NOT NULL
                WITH target.{field['property']} as target_time LIMIT 1
                MATCH (n) WHERE n.{field['property']} IS NOT NULL AND datetime(n.{field['property']}) > datetime(target_time)
                OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
                OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                RETURN n, n.{field['property']} as timestamp, target_time, labels(n) as labels,
                       toString(n.{field['property']}) as timestamp_str,
                       duration.between(datetime(target_time), datetime(n.{field['property']})) as time_diff,
                       pv.value AS physical_value, u.name AS unit
                ORDER BY timestamp ASC LIMIT 5
                """
            ])
        return self._execute_query_list(session, queries)

    def _query_message_sequence_enhanced(self, session, entity: str) -> List[Dict]:
        queries = []
        for field in self.temporal_fields:
            queries.extend([
                f"""
                MATCH (target) WHERE (target.name CONTAINS '{entity}' OR toString(target.id) CONTAINS '{entity}') AND target.{field['property']} IS NOT NULL
                WITH target.{field['property']} as target_time LIMIT 1
                MATCH (n) WHERE n.{field['property']} IS NOT NULL
                  AND duration.between(datetime(target_time), datetime(n.{field['property']})).seconds <= 3600
                  AND duration.between(datetime(target_time), datetime(n.{field['property']})).seconds >= -3600
                OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
                OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                RETURN n, n.{field['property']} as timestamp, labels(n) as labels,
                       toString(n.{field['property']}) as timestamp_str,
                       duration.between(datetime(target_time), datetime(n.{field['property']})).seconds as time_diff_seconds,
                       pv.value AS physical_value, u.name AS unit
                ORDER BY timestamp ASC LIMIT 20
                """
            ])
        return self._execute_query_list(session, queries)

    def _query_temporal_range_enhanced(self, session, start_time: str, end_time: str) -> List[Dict]:
        queries = []
        parsed_start = self.parse_timestamp(start_time)
        parsed_end = self.parse_timestamp(end_time)
        if not (parsed_start and parsed_end):
            print(f"⚠️ Invalid time range: {start_time} to {end_time}")
            return []
        start_str = parsed_start.isoformat()
        end_str = parsed_end.isoformat()
        for field in self.temporal_fields:
            queries.extend([
                f"""
                MATCH (n) WHERE n.{field['property']} IS NOT NULL
                  AND datetime(n.{field['property']}) >= datetime('{start_str}')
                  AND datetime(n.{field['property']}) <= datetime('{end_str}')
                OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
                OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                RETURN n, n.{field['property']} as timestamp, labels(n) as labels,
                       toString(n.{field['property']}) as timestamp_str,
                       pv.value AS physical_value, u.name AS unit
                ORDER BY timestamp ASC LIMIT 20
                """
            ])
        return self._execute_query_list(session, queries)

    def _query_duration_enhanced(self, session, entity: str) -> List[Dict]:
        queries = []
        for field in self.temporal_fields:
            queries.extend([
                f"""
                MATCH (n) WHERE (n.name CONTAINS '{entity}' OR toString(n.id) CONTAINS '{entity}') AND n.{field['property']} IS NOT NULL
                WITH n.{field['property']} as timestamp
                ORDER BY timestamp
                WITH collect(timestamp) as timestamps
                WHERE size(timestamps) > 1
                RETURN duration.between(datetime(timestamps[0]), datetime(timestamps[-1])) as duration,
                       timestamps[0] as first_time, timestamps[-1] as last_time,
                       toString(timestamps[0]) as first_time_str, toString(timestamps[-1]) as last_time_str,
                       size(timestamps) as event_count
                LIMIT 1
                """
            ])
        return self._execute_query_list(session, queries)

    def _query_frequency_enhanced(self, session, entity: str) -> List[Dict]:
        queries = []
        for field in self.temporal_fields:
            queries.extend([
                f"""
                MATCH (n) WHERE (n.name CONTAINS '{entity}' OR toString(n.id) CONTAINS '{entity}') AND n.{field['property']} IS NOT NULL
                WITH n.{field['property']} as timestamp
                WITH collect(timestamp) as timestamps
                WITH timestamps, size(timestamps) as count
                WHERE count > 1
                RETURN count as frequency,
                       min(timestamps) as earliest, max(timestamps) as latest,
                       toString(min(timestamps)) as earliest_str, toString(max(timestamps)) as latest_str
                LIMIT 1
                """
            ])
        return self._execute_query_list(session, queries)

    def _query_gap_enhanced(self, session, entity1: str, entity2: str) -> List[Dict]:
        queries = []
        for field in self.temporal_fields:
            queries.extend([
                f"""
                MATCH (n1) WHERE (n1.name CONTAINS '{entity1}' OR toString(n1.id) CONTAINS '{entity1}') AND n1.{field['property']} IS NOT NULL
                WITH n1.{field['property']} as time1 ORDER BY time1 DESC LIMIT 1
                MATCH (n2) WHERE (n2.name CONTAINS '{entity2}' OR toString(n2.id) CONTAINS '{entity2}') AND n2.{field['property']} IS NOT NULL
                WITH n2.{field['property']} as time2, time1 ORDER BY time2 DESC LIMIT 1
                RETURN duration.between(datetime(time1), datetime(time2)) as gap,
                       time1, time2, toString(time1) as time1_str, toString(time2) as time2_str
                LIMIT 1
                """
            ])
        return self._execute_query_list(session, queries)

    def _query_latest_enhanced(self, session, entity: str) -> List[Dict]:
        queries = []
        for field in self.temporal_fields:
            queries.extend([
                f"""
                MATCH (n) WHERE (n.name CONTAINS '{entity}' OR toString(n.id) CONTAINS '{entity}') AND n.{field['property']} IS NOT NULL
                OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
                OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                RETURN n, n.{field['property']} as timestamp, labels(n) as labels,
                       toString(n.{field['property']}) as timestamp_str,
                       pv.value AS physical_value, u.name AS unit
                ORDER BY datetime(n.{field['property']}) DESC LIMIT 1
                """
            ])
        return self._execute_query_list(session, queries)

    def _query_earliest_enhanced(self, session, entity: str) -> List[Dict]:
        queries = []
        for field in self.temporal_fields:
            queries.extend([
                f"""
                MATCH (n) WHERE (n.name CONTAINS '{entity}' OR toString(n.id) CONTAINS '{entity}') AND n.{field['property']} IS NOT NULL
                OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
                OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                RETURN n, n.{field['property']} as timestamp, labels(n) as labels,
                       toString(n.{field['property']}) as timestamp_str,
                       pv.value AS physical_value, u.name AS unit
                ORDER BY datetime(n.{field['property']}) ASC LIMIT 1
                """
            ])
        return self._execute_query_list(session, queries)

    def _query_since_enhanced(self, session, timestamp: str) -> List[Dict]:
        queries = []
        parsed_time = self.parse_timestamp(timestamp)
        if not parsed_time:
            print(f"⚠️ Invalid timestamp format: {timestamp}")
            return []
        timestamp_str = parsed_time.isoformat()
        for field in self.temporal_fields:
            queries.extend([
                f"""
                MATCH (n) WHERE n.{field['property']} IS NOT NULL AND datetime(n.{field['property']}) >= datetime('{timestamp_str}')
                OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
                OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                RETURN n, n.{field['property']} as timestamp, labels(n) as labels,
                       toString(n.{field['property']}) as timestamp_str,
                       pv.value AS physical_value, u.name AS unit
                ORDER BY timestamp ASC LIMIT 20
                """
            ])
        return self._execute_query_list(session, queries)

    def _query_until_enhanced(self, session, timestamp: str) -> List[Dict]:
        queries = []
        parsed_time = self.parse_timestamp(timestamp)
        if not parsed_time:
            print(f"⚠️ Invalid timestamp format: {timestamp}")
            return []
        timestamp_str = parsed_time.isoformat()
        for field in self.temporal_fields:
            queries.extend([
                f"""
                MATCH (n) WHERE n.{field['property']} IS NOT NULL AND datetime(n.{field['property']}) <= datetime('{timestamp_str}')
                OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
                OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                RETURN n, n.{field['property']} as timestamp, labels(n) as labels,
                       toString(n.{field['property']}) as timestamp_str,
                       pv.value AS physical_value, u.name AS unit
                ORDER BY timestamp DESC LIMIT 20
                """
            ])
        return self._execute_query_list(session, queries)

    def _query_exact_enhanced(self, session, timestamp: str) -> List[Dict]:
        queries = []
        parsed_time = self.parse_timestamp(timestamp)
        if not parsed_time:
            print(f"⚠️ Invalid timestamp format: {timestamp}")
            return []
        timestamp_str = parsed_time.isoformat()
        for field in self.temporal_fields:
            queries.extend([
                f"""
                MATCH (n) WHERE n.{field['property']} IS NOT NULL AND datetime(n.{field['property']}) = datetime('{timestamp_str}')
                OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
                OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                RETURN n, n.{field['property']} as timestamp, labels(n) as labels,
                       toString(n.{field['property']}) as timestamp_str,
                       pv.value AS physical_value, u.name AS unit
                LIMIT 20
                """
            ])
        return self._execute_query_list(session, queries)

    def _query_relative_enhanced(self, session, duration: str) -> List[Dict]:
        queries = []
        now = datetime.now(timezone.utc)
        match = re.match(r'(\d+)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?) ago', duration.lower())
        if not match:
            print(f"⚠️ Invalid relative time format: {duration}")
            return []
        
        amount = int(match.group(1))
        unit = match.group(2)
        if unit.startswith('second'):
            delta = timedelta(seconds=amount)
        elif unit.startswith('minute'):
            delta = timedelta(minutes=amount)
        elif unit.startswith('hour'):
            delta = timedelta(hours=amount)
        elif unit.startswith('day'):
            delta = timedelta(days=amount)
        elif unit.startswith('week'):
            delta = timedelta(weeks=amount)
        elif unit.startswith('month'):
            delta = timedelta(days=amount * 30) # Approximation
        elif unit.startswith('year'):
            delta = timedelta(days=amount * 365) # Approximation
        else:
            return []
            
        start_time_str = (now - delta).isoformat()
        for field in self.temporal_fields:
            queries.extend([
                f"""
                MATCH (n) WHERE n.{field['property']} IS NOT NULL AND datetime(n.{field['property']}) >= datetime('{start_time_str}')
                OPTIONAL MATCH (n)-[:MEASURED_AS]->(pv:PhysicalValue)
                OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                RETURN n, n.{field['property']} as timestamp, labels(n) as labels,
                       toString(n.{field['property']}) as timestamp_str,
                       pv.value AS physical_value, u.name AS unit
                ORDER BY timestamp DESC LIMIT 20
                """
            ])
        return self._execute_query_list(session, queries)

class ResponseSynthesizerAgent:
    """
    Generates a final, user-friendly response by synthesizing information
    from all other agents using a generative AI model.
    """
    def __init__(self, gemini_model, schema_info, graph_statistics):
        self.gemini_model = gemini_model
        self.schema_info = schema_info
        self.graph_statistics = graph_statistics
        print("✅ Response Synthesizer Agent initialized.")
        
    def parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Utility for fallback response generation."""
        if not timestamp_str: return None
        try: return parser.parse(str(timestamp_str))
        except (parser.ParserError, TypeError): return None

    def _serialize_graph_elements(self, data):
        """Custom serializer to handle Neo4j graph types for JSON conversion."""
        if isinstance(data, (graph.Node, graph.Relationship, graph.Path)):
            return dict(data)
        if isinstance(data, datetime):
            return data.isoformat()
        if isinstance(data, timedelta):
            return str(data)
        # Fallback for any other un-serializable types
        return str(data)

    def synthesize(self, query: str, intent: Dict,
                   relevant_entities: List[Tuple[Dict, float]],
                   query_results: List[Dict]) -> str:
        """Generates a comprehensive response."""
        context_parts = [f"User Query: {query}\n"]
        context_parts.append(f"Query Intent: {intent['type']}")
        if intent['entities']:
            context_parts.append(f"Extracted Entities: {', '.join(map(str, intent['entities']))}")
        if intent['modifiers']:
            context_parts.append(f"Query Modifiers: {', '.join(intent['modifiers'])}")

        if relevant_entities:
            context_parts.append("\nRelevant Entities Found:")
            for entity, confidence in relevant_entities[:5]:
                context_parts.append(f"- {entity['identifier']} ({entity['type']}, confidence: {confidence:.2f})")
        
        if query_results:
            context_parts.append(f"\nData Retrieved ({len(query_results)} records):")
            try:
                # Use custom serializer for Neo4j objects
                for i, result in enumerate(query_results[:5]): # Show up to 5 records in context
                    context_parts.append(f"Record {i+1}: {json.dumps(result, default=self._serialize_graph_elements, indent=2)}")
                if len(query_results) > 5:
                    context_parts.append(f"... and {len(query_results) - 5} more records")
            except (TypeError, OverflowError) as e:
                 context_parts.append(f"Note: Some data could not be serialized for display. Error: {e}")

        context_parts.append(f"\nDatabase Schema Summary:")
        context_parts.append(f"Node types: {list(self.schema_info.get('labels', {}).keys())}")
        context_parts.append(f"Relationship types: {list(self.schema_info.get('relationships', {}).keys())}")
        
        if self.graph_statistics:
            stats = self.graph_statistics.get('totals', {})
            context_parts.append(f"Graph size: {stats.get('nodes', 0)} nodes, {stats.get('relationships', 0)} relationships")
        
        context_str = "\n".join(context_parts)

        prompt = f"""
You are an advanced graph database analyst assistant. Based on the comprehensive context below, provide a detailed and informative response to the user's query.

--- CONTEXT ---
{context_str}
--- END CONTEXT ---

Instructions:
1. Provide a direct answer to the user's question based on the retrieved data.
2. If specific data was found, explain it clearly and highlight key insights.
3. If no data was found, explain why and suggest alternative approaches.
4. For CAN ID queries, list all associated signals and their purposes.
5. For node/relationship queries, explain the graph structure and connections.
6. For aggregation queries, provide counts and summaries.
7. For temporal queries, clearly state the timestamps in IST (UTC+05:30) and associated entities.
8. For signal-related queries, always include physical values from PhysicalValue nodes (via MEASURED_AS) and their units (via HAS_UNIT) if available.
9. Include relevant technical details but keep explanations accessible.
10. If you see interesting patterns or anomalies, point them out.
11.  **Be Specific:**
    *   **Physical Values:** If the query is about a value (e.g., "what is the speed"), find the relevant signal in the data (like `DI_vehicleSpeed`), and state its `physical_value` and `unit` clearly. If multiple values exist, mention the most recent one based on the timestamp.
    *   **Temporal Queries:** When asked about time, state the full timestamp clearly. Convert it to a more readable format and mention the timezone (IST: UTC+05:30) as requested.
    *   **Node/Entity Queries:** If asked about an entity like an "inverter" or "signal", describe its properties and connections based on the retrieved records.
    """
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            # Fallback response generation
            return self._generate_fallback_response(query, intent, relevant_entities, query_results)
            
    def _generate_fallback_response(self, query: str, intent: Dict, relevant_entities: List[Tuple[Dict, float]], query_results: List[Dict]) -> str:
        """Generates a simple, structured response if the LLM fails."""
        if query_results and intent['type'] in ['temporal', 'physical_value']:
            ist = pytz.timezone('Asia/Kolkata')
            entity_name = intent['entities'][0] if intent.get('entities') else "your query"
            summary = f"Results for '{entity_name}' in IST (UTC+05:30):\n"
            for r in query_results:
                ts = r.get('send_time') or r.get('timestamp') or r.get('timestamp_str')
                pv = r.get('physical_value') or r.get('physical_value_str')
                unit = r.get('unit', 'N/A')
                signal_name = r.get('signal_name', entity_name)
                if ts or pv is not None:
                    result_str = f"- Signal: {signal_name}"
                    if ts:
                        parsed_ts = self.parse_timestamp(str(ts))
                        if parsed_ts:
                            try:
                                formatted_ts = parsed_ts.astimezone(ist).strftime('%Y-%m-%d %H:%M:%S.%f %Z')
                                result_str += f", Timestamp: {formatted_ts}"
                            except (ValueError, TypeError):
                                result_str += f", Timestamp: {str(parsed_ts)} (could not format timezone)"
                    if pv is not None:
                        result_str += f", Physical Value: {pv} ({unit})"
                    summary += result_str + "\n"
            if not any(r.get('physical_value') or r.get('physical_value_str') for r in query_results):
                summary += "No physical values found.\n"
            summary += "\nTry asking for a specific time range or check the signal name."
            return summary
        elif query_results:
            summary = f"Found {len(query_results)} records matching your query '{query}'. "
            if intent['type'] == 'can_id':
                summary += f"This includes information about CAN ID {intent['entities'][0]} and its associated signals/data."
            elif relevant_entities:
                entities_list = [e[0]['identifier'] for e in relevant_entities[:3]]
                summary += f"The most relevant entities are: {', '.join(entities_list)}."
            return summary
        else:
            entity_str = intent['entities'][0] if intent.get('entities') else 'x'
            return (f"I couldn't find specific data for your query '{query}', but I can help you explore the graph database. "
                    f"Try asking about specific CAN IDs, node types, relationships, timestamps, or physical values "
                    f"(e.g., 'What is the physical value of signal {entity_str}').")

# ==============================================================================
# 2. THE ORCHESTRATOR AGENT
# ==============================================================================

class OrchestratorAgent:
    """
    Initializes, coordinates, and manages the entire multi-agent system.
    Handles the main processing pipeline and user interaction.
    """
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, gemini_api_key: str):
        # --- Connections and Models ---
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        
        # --- Data Structures & Config ---
        self.all_entities = []
        self.entity_embeddings = None
        self.faiss_index = None
        self.schema_info = {}
        self.graph_statistics = {}
        self.temporal_fields = []
        # Enhanced query patterns from original code
        self.query_patterns = {
            'can_id': [
                r'CAN ID (\d+)', r'can id (\d+)', r'ID (\d+)',
                r'signals? (?:from|by|sent by|transmitted by) (?:CAN )?ID (\d+)',
                r'what (?:signals?|data) (?:does|are) (?:CAN )?ID (\d+)',
            ],
            'node_type': [r'(?:all|show|list|get) (\w+)s?', r'what (\w+)s? (?:are|exist)', r'find (\w+)s?'],
            'relationship': [
                r'connected to (\w+)', r'related to (\w+)', r'linked with (\w+)',
                r'relationships? (?:of|with) (\w+)',
            ],
           'temporal': [
                r'at (\d{1,2}:\d{2}(?::\d{2})?)', r'around (\d{1,2}:\d{2}(?::\d{2})?)',
                r'on (\d{4}-\d{2}-\d{2})',
                r'at (\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2}(?:\.\d{3,6})?(?:[+-]\d{2}:\d{2})?)',
                r'since (\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2}(?:\.\d{3,6})?(?:[+-]\d{2}:\d{2})?)',
                r'until (\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2}(?:\.\d{3,6})?(?:[+-]\d{2}:\d{2})?)',
                r'exactly (\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2}(?:\.\d{3,6})?(?:[+-]\d{2}:\d{2})?)',
                r'between (.+?)\s+and\s+(.+)', r'from (.+?)\s+to\s+(.+)',
                r'(\d+)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+ago',
                r'before (.+)', r'after (.+)',
                r'(latest|earliest|first|last|most recent|oldest)\s+(\w+)',
                r'(sequence|duration|frequency|gap|time difference|interval)\s+(?:of|between)?\s*(.+)',
                r'when (?:was|were) (?:the )?(?:message|signal)\s+([A-Za-z0-9_]+)\s+(?:sent|transmitted)',
                r'timestamp of (?:message|signal)\s+([A-Za-z0-9_]+)',
                r'when (?:was )?CAN ID\s+(0x[0-9A-Fa-f]+|[0-9]+)\s+(?:sent|transmitted)',
                r'today', r'yesterday', r'this week', r'last week',
            ],
            'comparison': [r'compare (\w+) (?:and|with) (\w+)', r'difference between (\w+) and (\w+)', r'(\w+) vs (\w+)'],
            'aggregation': [r'(?:count|total|sum) (?:of )?(\w+)', r'how many (\w+)', r'average (\w+)', r'maximum (\w+)', r'minimum (\w+)'],
            'properties': [r'properties of (\w+)', r'attributes of (\w+)', r'details (?:of|about) (\w+)', r'information (?:on|about) (\w+)'],
            'physical_value': [r'physical value of (\w+)', r'value of signal (\w+)', r'what is the value of (\w+)', r'signal (\w+) value']
        }

        # --- Setup and Agent Initialization ---
        self.setup_pipeline()
        
        print("\n🤝 Initializing specialized agents...")
        self.planner = QueryPlannerAgent(self.query_patterns)
        self.retriever = SemanticRetrieverAgent(self.embedding_model, self.all_entities, self.entity_embeddings, self.faiss_index)
        self.explorer = GraphExplorerAgent(self.driver, self.temporal_fields)
        self.synthesizer = ResponseSynthesizerAgent(self.gemini_model, self.schema_info, self.graph_statistics)
        print("✅ Orchestrator Agent is ready and all specialized agents are online.")


    def setup_pipeline(self):
        """Runs all setup tasks to prepare the system."""
        print("🔄 Setting up orchestrator pipeline...")
        self.discover_comprehensive_schema()
        self.discover_temporal_fields_enhanced()
        self._load_all_entities()
        self._generate_graph_statistics()
        print("✅ Orchestrator pipeline setup complete!")

    def discover_comprehensive_schema(self):
        print("🔍 Discovering comprehensive database schema...")
        with self.driver.session() as session:
            try:
                labels_info = {}
                labels_result = session.run("CALL db.labels()")
                labels = [record["label"] for record in labels_result]
                
                for label in labels:
                    try:
                        prop_query = f"MATCH (n:{label}) WITH n LIMIT 5 RETURN keys(n) as properties"
                        prop_result = session.run(prop_query)
                        properties = set()
                        for record in prop_result:
                            properties.update(record["properties"])
                        
                        count_result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                        total_count = count_result.single()["count"]
                        
                        labels_info[label] = {'count': total_count, 'properties': list(properties)}
                    except Exception:
                        labels_info[label] = {'count': 0, 'properties': []}
                
                relationships_info = {}
                rels_result = session.run("CALL db.relationshipTypes()")
                relationships = [record["relationshipType"] for record in rels_result]
                
                for rel_type in relationships:
                    try:
                        rel_query = f"""
                        MATCH (a)-[r:{rel_type}]->(b)
                        RETURN labels(a) as from_labels, labels(b) as to_labels, count(r) as count
                        LIMIT 10
                        """
                        rel_result = session.run(rel_query)
                        connections = []
                        total_count = 0
                        for record in rel_result:
                            conn = {'from': record["from_labels"], 'to': record["to_labels"]}
                            if conn not in connections:
                                connections.append(conn)
                        
                        count_res = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                        total_count = count_res.single()["count"]
                        
                        relationships_info[rel_type] = {'total_count': total_count, 'connections': connections}
                    except Exception:
                        relationships_info[rel_type] = {'total_count': 0, 'connections': []}
                
                self.schema_info = {'labels': labels_info, 'relationships': relationships_info}
                print(f"📊 Found {len(labels_info)} node types and {len(relationships_info)} relationship types")

            except Exception as e:
                print(f"❌ Schema discovery error: {e}")
                self.schema_info = {'labels': {}, 'relationships': {}}

    def _load_all_entities(self):
        print("🔄 Loading all searchable entities...")
        entities = []
        with self.driver.session() as session:
            try:
                # Load Nodes
                for label, info in self.schema_info['labels'].items():
                    properties = info.get('properties', [])
                    name_props = [prop for prop in properties if any(keyword in prop.lower() for keyword in ['name', 'id', 'title', 'value', 'physical_value'])]
                    prop_selector = f"COALESCE({', '.join([f'toString(n.{prop})' for prop in name_props])}, toString(id(n)))" if name_props else "toString(id(n))"
                    query = f"MATCH (n:{label}) WHERE {prop_selector} IS NOT NULL RETURN {prop_selector} as identifier LIMIT 1000"
                    result = session.run(query)
                    for record in result:
                        identifier = record["identifier"]
                        if identifier:
                            entities.append({'type': 'node', 'label': label, 'identifier': str(identifier), 'searchable_text': f"{label} {identifier}"})
                
                # Load Relationships
                for rel_type, info in self.schema_info['relationships'].items():
                    entities.append({'type': 'relationship', 'identifier': rel_type, 'searchable_text': f"relationship {rel_type}"})
                
                # Load Properties
                all_properties = set()
                for label_info in self.schema_info['labels'].values():
                    all_properties.update(label_info.get('properties', []))
                for prop in all_properties:
                    entities.append({'type': 'property', 'identifier': prop, 'searchable_text': f"property {prop}"})
                
                self.all_entities = entities
                print(f"✅ Loaded {len(entities)} searchable entities")
                
                if entities:
                    print("🔄 Creating embeddings for all entities...")
                    searchable_texts = [entity['searchable_text'] for entity in entities]
                    self.entity_embeddings = self.embedding_model.encode(searchable_texts, convert_to_numpy=True)
                    embedding_dim = self.entity_embeddings.shape[1]
                    self.faiss_index = faiss.IndexFlatL2(embedding_dim)
                    self.faiss_index.add(self.entity_embeddings)
                    print("✅ Entity embeddings created and indexed")
                
            except Exception as e:
                print(f"❌ Error loading entities: {e}")

    def _generate_graph_statistics(self):
        print("📊 Generating graph statistics...")
        with self.driver.session() as session:
            try:
                stats = {}
                total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                
                stats['totals'] = {'nodes': total_nodes, 'relationships': total_rels}
                
                degree_query = """
                MATCH (n)
                WITH n, size((n)--()) as degree
                ORDER BY degree DESC LIMIT 10
                RETURN labels(n) as labels, COALESCE(n.name, n.id, n.value, toString(id(n))) as identifier, degree
                """
                degree_result = session.run(degree_query)
                stats['most_connected'] = [dict(record) for record in degree_result]
                
                self.graph_statistics = stats
                print(f"📈 Graph stats: {total_nodes} nodes, {total_rels} relationships")
                
            except Exception as e:
                print(f"❌ Error generating statistics: {e}")
                self.graph_statistics = {}

    def _is_temporal_value(self, value) -> bool:
        if not value: return False
        value_str = str(value)
        # Using a general parser is more robust than regex for this
        try:
            parser.parse(value_str)
            return True
        except (parser.ParserError, TypeError, ValueError):
            return False

    def discover_temporal_fields_enhanced(self):
        print("🕒 Discovering temporal fields...")
        temporal_keywords = ['time', 'date', 'created', 'sent', 'received', 'occurred', 'start', 'end', 'when', 'at']
        with self.driver.session() as session:
            try:
                temporal_fields = []
                for label, info in self.schema_info.get('labels', {}).items():
                    for prop in info.get('properties', []):
                        if any(keyword in prop.lower() for keyword in temporal_keywords):
                            try:
                                sample_query = f"MATCH (n:{label}) WHERE n.{prop} IS NOT NULL RETURN n.{prop} as value LIMIT 1"
                                result = session.run(sample_query)
                                sample_record = result.single()
                                if sample_record and self._is_temporal_value(sample_record['value']):
                                    temporal_fields.append({'label': label, 'property': prop})
                                    print(f"   ✓ Found temporal field: {label}.{prop}")
                            except Exception:
                                continue
                self.temporal_fields = temporal_fields
                print(f"🕒 Found {len(temporal_fields)} potential temporal fields total")
            except Exception as e:
                print(f"❌ Error discovering temporal fields: {e}")
                self.temporal_fields = []

    def process_query(self, user_query: str) -> str:
        """Processes a user query by coordinating the agents in a pipeline."""
        print(f"\n🚀 Orchestrator processing query: '{user_query}'")
        
        print("  - Step 1: Engaging Query Planner Agent...")
        intent = self.planner.plan(user_query)
        print(f"  - Plan received: Intent is '{intent['type']}', Temporal type: {intent.get('temporal_type', 'N/A')}")

        print("  - Step 2: Engaging Semantic Retriever Agent...")
        relevant_entities = self.retriever.retrieve(user_query, k=8)
        print(f"  - Retrieved {len(relevant_entities)} semantically relevant entities.")

        print("  - Step 3: Engaging Graph Explorer Agent...")
        query_results = self.explorer.explore(intent, relevant_entities)
        print(f"  - Explorer returned {len(query_results)} graph results.")

        print("  - Step 4: Engaging Response Synthesizer Agent...")
        response = self.synthesizer.synthesize(user_query, intent, relevant_entities, query_results)
        print("  - Synthesis complete. Final response generated.")
        
        return response

    def start_interactive_session(self):
        """Starts the main interactive command-line interface."""
        print("\n" + "="*70)
        print("🚗 MULTI-AGENT VEHICLE DATA ASSISTANT - HYBRID GRAPH EXPLORER")
        print("="*70)
        print("An orchestrator and a team of specialized agents are at your service.")
        print("Ask anything about your graph. Type 'help', 'schema', 'stats', or 'quit'.")
        print("="*70)
        while True:
            try:
                user_input = input("\n💬 Your question: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! The agent team is signing off.")
                    break
                elif user_input.lower() == 'help':
                    self._show_enhanced_help()
                    continue
                elif user_input.lower() == 'schema':
                    self._show_comprehensive_schema()
                    continue
                elif user_input.lower() == 'stats':
                    self._show_graph_statistics()
                    continue
                elif user_input.lower() == 'entities':
                    self._show_sample_entities()
                    continue
                elif user_input.lower().startswith('debug entity '):
                    entity = user_input[12:].strip()
                    debug_info = self.debug_entity_search(entity)
                    print(json.dumps(debug_info, indent=2, default=str))
                    continue
                elif not user_input:
                    continue
                
                response = self.process_query(user_input)
                print(f"\n🤖 Assistant Team: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An unexpected error occurred in the orchestrator: {e}")
                import traceback
                traceback.print_exc()
                print("Please try a different query.")

    def _show_enhanced_help(self):
        print("""
📋 COMPREHENSIVE QUERY EXAMPLES:

🚌 CAN ID Queries:
   • "Show me everything about CAN ID 551"
   • "What signals does CAN ID 227 transmit?"
   • "Find all data related to ID 1234"

🏷️ Node Type Queries:
   • "Show me all Signals"
   • "List all CAN_IDs in the database"
   • "Find all Messages"

🔗 Relationship Queries:
   • "What is connected to Signal_ABC?"
   • "Show relationships of CAN_ID_551"
   • "How are Messages linked to Signals?"

📊 Aggregation Queries:
   • "Count all Signals"
   • "How many CAN IDs are there?"
   • "Total number of Messages"

🕒 Temporal Queries:
   • "When was message ABC123 sent?"
   • "When did the message ABC123 send?"
   • "When was signal XYZ transmitted?"
   • "Messages at 2022-08-14 18:59:11.177450+00:00"
   • "Data since 2022-08-14 18:00:00.000000+00:00"
   • "Messages until 2022-08-14 19:30:00.000000+00:00"
   • "Data between 2022-08-14 18:00 and 2022-08-14 19:00"
   • "Messages from 5 minutes ago"
   • "Latest signals"
   • "Earliest messages"

📈 Analysis Queries:
   • "What is the physical value of ID1D8RearTorque"
   • "Show me the most connected nodes"
   • "Show graph statistics"
""")

    def _show_comprehensive_schema(self):
        print(f"\n📊 COMPREHENSIVE DATABASE SCHEMA:")
        if not self.schema_info.get('labels') and not self.schema_info.get('relationships'):
            print("  Schema information is not available. Please check the database connection.")
            return

        print(f"\nNode Types ({len(self.schema_info.get('labels', {}))}):")
        for label, info in self.schema_info.get('labels', {}).items():
            props = ", ".join(info.get('properties', [])) or "N/A"
            print(f"   • {label}: {info.get('count', 0)} nodes, Properties: {props}")
            
        print(f"\nRelationship Types ({len(self.schema_info.get('relationships', {}))}):")
        for rel, info in self.schema_info.get('relationships', {}).items():
            connections = [f"{conn['from']}->{conn['to']}" for conn in info.get('connections', [])]
            conn_str = ", ".join(set(connections)) or "N/A"
            print(f"   • {rel}: {info.get('total_count', 0)} instances, Connections: {conn_str}")

    def _show_graph_statistics(self):
        print(f"\n📈 GRAPH STATISTICS:")
        if not self.graph_statistics:
            print("  Statistics are not available.")
            return

        print("\nTotal Counts:")
        print(f"   • Nodes: {self.graph_statistics.get('totals', {}).get('nodes', 0)}")
        print(f"   • Relationships: {self.graph_statistics.get('totals', {}).get('relationships', 0)}")

        print("\nMost Connected Nodes:")
        for node in self.graph_statistics.get('most_connected', []):
            labels = ", ".join(node.get('labels', []))
            print(f"   • {node.get('identifier')} ({labels}): {node.get('degree')} connections")

    def _show_sample_entities(self):
        if not self.all_entities:
            print("❌ No entities loaded. Check your database connection and schema.")
            return
        
        print(f"\n📋 SAMPLE ENTITIES ({len(self.all_entities)} total):")
        entity_types = {}
        for entity in self.all_entities:
            entity_type = entity['type']
            if len(entity_types.get(entity_type, [])) < 5:
                entity_types.setdefault(entity_type, []).append(entity)
        
        for entity_type, entities in entity_types.items():
            print(f"\n{entity_type.capitalize()}s:")
            for entity in entities:
                print(f"   • {entity['identifier']}")

    def debug_entity_search(self, entity: str) -> Dict:
        results = {
            'entity': entity, 'found_in_entities': False, 'matching_entities': [], 'semantic_search_results': []
        }
        for e in self.all_entities:
            if e['identifier'] == entity or entity.lower() in e['identifier'].lower():
                results['found_in_entities'] = True
                results['matching_entities'].append(e)
        if self.faiss_index:
            query_embedding = self.embedding_model.encode([entity], convert_to_numpy=True)
            distances, indices = self.faiss_index.search(query_embedding, k=5)
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.all_entities):
                    entity_info = self.all_entities[idx]
                    confidence = 1.0 / (1.0 + float(distance))
                    results['semantic_search_results'].append({'entity': entity_info, 'confidence': confidence})
        return results

    def close(self):
        """Cleans up resources."""
        if self.driver:
            self.driver.close()
            print("\n🔌 Neo4j connection closed by Orchestrator.")

# ==============================================================================
# 3. MAIN EXECUTION BLOCK
# ==============================================================================

def batch_process_queries(orchestrator, input_file: str, output_file: str):
    results = []

    try:
        with open(input_file, 'r') as f:
            queries = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found.")
        return

    for idx, query in enumerate(queries, 1):
        print(f"\n🔎 Processing Query {idx}/{len(queries)}: {query}")
        start_time = time.time()
        try:
            response = orchestrator.process_query(query)
        except Exception as e:
            response = f"❌ Error: {str(e)}"
        end_time = time.time()
        elapsed = round(end_time - start_time, 2)

        try:
            # SentenceTransformer expects a list of strings and returns a 2D array. Get the first vector.
            query_embedding = orchestrator.embedding_model.encode([query], convert_to_numpy=True)[0]
            response_embedding = orchestrator.embedding_model.encode([response], convert_to_numpy=True)[0]
            
            # Calculate cosine similarity: (A . B) / (||A|| * ||B||)
            dot_product = np.dot(query_embedding, response_embedding)
            norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding)

            # Avoid division by zero
            if norm_product == 0:
                cosine_sim = 0.0
            else:
                cosine_sim = float(dot_product / norm_product)

        except Exception as sim_error:
            cosine_sim = None
            print(f"⚠️ Similarity computation failed for Query {idx}: {sim_error}")

        result = {
            'query': query,
            'response': response,
            'time_taken_sec': elapsed,
            'answer_relevance': cosine_sim
        }
        results.append(result)

    # Save all results to JSON
    with open(output_file, 'w') as out_file:
        json.dump(results, out_file, indent=2)

    print(f"\n✅ Batch processing complete. Results saved to '{output_file}'")

def main():
    # --- CONFIGURATION ---
    # IMPORTANT: Replace with your actual credentials and API keys
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"  # Replace with your actual Neo4j password
    GEMINI_API_KEY = "AIzaSyBURv6gmG4M0hDZyY0Gctb5gYxNqoUTD_0"    # Replace with your actual Gemini API key

    orchestrator = None
    try:
        orchestrator = OrchestratorAgent(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            gemini_api_key=GEMINI_API_KEY
        )
        batch_process_queries(
            orchestrator,
            input_file='new_queries.txt',       # Provide your queries text file
            output_file='new_query_results.txt'  # Output file for results
        )
    except Exception as e:
        print(f"\n❌ FATAL: Failed to initialize the Orchestrator Agent: {e}")
        print("   Please check your Neo4j connection, credentials, and API keys.")
        import traceback
        traceback.print_exc()
    finally:
        if orchestrator:
            orchestrator.close()

if __name__ == "__main__":
    main()
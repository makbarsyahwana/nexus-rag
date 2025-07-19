import asyncio
from typing import Dict, List, Optional, Any, Union
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, ClientError
import structlog

logger = structlog.get_logger(__name__)

class Neo4jClient:
    """Neo4j graph database client for knowledge graph operations."""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50
    ):
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j database URI
            user: Username for authentication
            password: Password for authentication
            database: Database name
            max_connection_lifetime: Connection lifetime in seconds
            max_connection_pool_size: Maximum connection pool size
        """
        self.uri = uri
        self.user = user
        self.database = database
        
        self.driver: AsyncDriver = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_lifetime=max_connection_lifetime,
            max_connection_pool_size=max_connection_pool_size
        )
        
        logger.info("Neo4j client initialized", uri=uri, database=database)
    
    async def close(self):
        """Close the database connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        fetch_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            fetch_results: Whether to fetch and return results
            
        Returns:
            List of result records
        """
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, parameters or {})
                
                if fetch_results:
                    records = await result.data()
                    logger.info("Query executed", query_preview=query[:100], results_count=len(records))
                    return records
                else:
                    await result.consume()
                    logger.info("Query executed without fetching results", query_preview=query[:100])
                    return []
                    
        except (ServiceUnavailable, ClientError) as e:
            logger.error("Neo4j query error", error=str(e), query=query[:100])
            raise
        except Exception as e:
            logger.error("Unexpected error in query execution", error=str(e))
            raise
    
    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
        return_node: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Create a node with given label and properties.
        
        Args:
            label: Node label
            properties: Node properties
            return_node: Whether to return the created node
            
        Returns:
            Created node data if return_node=True
        """
        try:
            # Build property string for Cypher query
            prop_string = ", ".join([f"{key}: ${key}" for key in properties.keys()])
            
            if return_node:
                query = f"CREATE (n:{label} {{{prop_string}}}) RETURN n"
                result = await self.execute_query(query, properties)
                return result[0]["n"] if result else None
            else:
                query = f"CREATE (:{label} {{{prop_string}}})"
                await self.execute_query(query, properties, fetch_results=False)
                return None
                
        except Exception as e:
            logger.error("Failed to create node", label=label, error=str(e))
            raise
    
    async def create_relationship(
        self,
        start_node_label: str,
        start_node_properties: Dict[str, Any],
        end_node_label: str,
        end_node_properties: Dict[str, Any],
        relationship_type: str,
        relationship_properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a relationship between two nodes.
        
        Args:
            start_node_label: Label of start node
            start_node_properties: Properties to match start node
            end_node_label: Label of end node
            end_node_properties: Properties to match end node
            relationship_type: Type of relationship
            relationship_properties: Optional relationship properties
            
        Returns:
            True if successful
        """
        try:
            # Build match conditions
            start_match = " AND ".join([f"start.{k} = $start_{k}" for k in start_node_properties.keys()])
            end_match = " AND ".join([f"end.{k} = $end_{k}" for k in end_node_properties.keys()])
            
            # Build relationship properties
            rel_props = ""
            if relationship_properties:
                rel_prop_string = ", ".join([f"{k}: $rel_{k}" for k in relationship_properties.keys()])
                rel_props = f" {{{rel_prop_string}}}"
            
            query = f"""
            MATCH (start:{start_node_label}), (end:{end_node_label})
            WHERE {start_match} AND {end_match}
            CREATE (start)-[r:{relationship_type}{rel_props}]->(end)
            """
            
            # Prepare parameters
            parameters = {}
            parameters.update({f"start_{k}": v for k, v in start_node_properties.items()})
            parameters.update({f"end_{k}": v for k, v in end_node_properties.items()})
            if relationship_properties:
                parameters.update({f"rel_{k}": v for k, v in relationship_properties.items()})
            
            await self.execute_query(query, parameters, fetch_results=False)
            logger.info("Relationship created", type=relationship_type)
            return True
            
        except Exception as e:
            logger.error("Failed to create relationship", type=relationship_type, error=str(e))
            return False
    
    async def find_nodes(
        self,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find nodes by label and properties.
        
        Args:
            label: Node label
            properties: Properties to match
            limit: Maximum number of results
            
        Returns:
            List of matching nodes
        """
        try:
            where_clause = ""
            parameters = {}
            
            if properties:
                conditions = []
                for key, value in properties.items():
                    conditions.append(f"n.{key} = ${key}")
                    parameters[key] = value
                where_clause = f"WHERE {' AND '.join(conditions)}"
            
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            query = f"MATCH (n:{label}) {where_clause} RETURN n {limit_clause}"
            
            results = await self.execute_query(query, parameters)
            return [record["n"] for record in results]
            
        except Exception as e:
            logger.error("Failed to find nodes", label=label, error=str(e))
            return []
    
    async def find_relationships(
        self,
        start_node_label: Optional[str] = None,
        relationship_type: Optional[str] = None,
        end_node_label: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find relationships by pattern.
        
        Args:
            start_node_label: Optional start node label
            relationship_type: Optional relationship type
            end_node_label: Optional end node label
            limit: Maximum number of results
            
        Returns:
            List of relationship patterns
        """
        try:
            start_pattern = f"(start:{start_node_label})" if start_node_label else "(start)"
            rel_pattern = f"[r:{relationship_type}]" if relationship_type else "[r]"
            end_pattern = f"(end:{end_node_label})" if end_node_label else "(end)"
            
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            query = f"""
            MATCH {start_pattern}-{rel_pattern}->{end_pattern}
            RETURN start, r, end {limit_clause}
            """
            
            return await self.execute_query(query)
            
        except Exception as e:
            logger.error("Failed to find relationships", error=str(e))
            return []
    
    async def get_node_neighbors(
        self,
        node_label: str,
        node_properties: Dict[str, Any],
        relationship_type: Optional[str] = None,
        direction: str = "both",
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get neighbors of a node.
        
        Args:
            node_label: Node label
            node_properties: Properties to identify the node
            relationship_type: Optional relationship type filter
            direction: Direction of relationships ('in', 'out', 'both')
            limit: Maximum number of results
            
        Returns:
            List of neighboring nodes with relationship info
        """
        try:
            # Build match condition for source node
            node_match = " AND ".join([f"n.{k} = ${k}" for k in node_properties.keys()])
            
            # Build relationship pattern based on direction
            rel_type = f":{relationship_type}" if relationship_type else ""
            if direction == "out":
                rel_pattern = f"-[r{rel_type}]->(neighbor)"
            elif direction == "in":
                rel_pattern = f"<-[r{rel_type}]-(neighbor)"
            else:  # both
                rel_pattern = f"-[r{rel_type}]-(neighbor)"
            
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            query = f"""
            MATCH (n:{node_label}){rel_pattern}
            WHERE {node_match}
            RETURN neighbor, r {limit_clause}
            """
            
            return await self.execute_query(query, node_properties)
            
        except Exception as e:
            logger.error("Failed to get node neighbors", label=node_label, error=str(e))
            return []
    
    async def delete_nodes(
        self,
        label: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Delete nodes by label and properties.
        
        Args:
            label: Node label
            properties: Properties to match
            
        Returns:
            Number of deleted nodes
        """
        try:
            where_clause = ""
            parameters = {}
            
            if properties:
                conditions = []
                for key, value in properties.items():
                    conditions.append(f"n.{key} = ${key}")
                    parameters[key] = value
                where_clause = f"WHERE {' AND '.join(conditions)}"
            
            query = f"MATCH (n:{label}) {where_clause} DELETE n RETURN count(n) as deleted_count"
            
            result = await self.execute_query(query, parameters)
            deleted_count = result[0]["deleted_count"] if result else 0
            
            logger.info("Nodes deleted", label=label, count=deleted_count)
            return deleted_count
            
        except Exception as e:
            logger.error("Failed to delete nodes", label=label, error=str(e))
            return 0
    
    async def create_indexes(self, indexes: List[Dict[str, str]]) -> bool:
        """
        Create indexes for better query performance.
        
        Args:
            indexes: List of index definitions with 'label' and 'property' keys
            
        Returns:
            True if all indexes created successfully
        """
        try:
            for index in indexes:
                label = index["label"]
                property_name = index["property"]
                
                query = f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{property_name})"
                await self.execute_query(query, fetch_results=False)
                
                logger.info("Index created", label=label, property=property_name)
            
            return True
            
        except Exception as e:
            logger.error("Failed to create indexes", error=str(e))
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Neo4j service health."""
        try:
            result = await self.execute_query("RETURN 1 as test")
            
            if result and result[0]["test"] == 1:
                return {
                    "status": "healthy",
                    "uri": self.uri,
                    "database": self.database
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "Unexpected query result"
                }
                
        except Exception as e:
            logger.error("Neo4j health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }

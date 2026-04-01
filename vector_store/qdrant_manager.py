"""
Qdrant Vector Store Integration for Multi-Modal Embeddings
Stores and retrieves radiology, genomics, and fused embeddings
"""
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams
)
import numpy as np
from loguru import logger

from config.settings import config


class VectorStoreManager:
    """
    Manages Qdrant vector collections for multi-modal diagnostic embeddings
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        embedding_dim: int = None
    ):
        self.host = host or config.qdrant.host
        self.port = port or config.qdrant.port
        self.embedding_dim = embedding_dim or config.qdrant.embedding_dim
        
        self.client = QdrantClient(host=self.host, port=self.port)
        
        self.radiology_collection = config.qdrant.radiology_collection
        self.genomics_collection = config.qdrant.genomics_collection
        self.fusion_collection = config.qdrant.fusion_collection
        
        logger.info(f"VectorStoreManager connected to Qdrant at {self.host}:{self.port}")
    
    def create_collections(self) -> None:
        """Create all required collections if they don't exist"""
        collections_config = [
            (self.radiology_collection, "Radiology imaging embeddings"),
            (self.genomics_collection, "Genomic sequence embeddings"),
            (self.fusion_collection, "Multi-modal fused embeddings")
        ]
        
        for collection_name, description in collections_config:
            self._create_collection_if_not_exists(collection_name, description)
        
        logger.success("All vector collections initialized")
    
    def _create_collection_if_not_exists(
        self,
        collection_name: str,
        description: str
    ) -> None:
        """Create a collection if it doesn't already exist"""
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        existing_names = [c.name for c in collections]
        
        if collection_name not in existing_names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                ),
                optimizers_config={
                    "default_segment_number": 5,
                    "indexing_threshold": 20000
                }
            )
            
            # Create payload indexes for efficient filtering
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="patient_id",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="modality",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="hospital_id",
                field_schema="keyword"
            )
            
            logger.info(f"Created collection '{collection_name}': {description}")
        else:
            logger.debug(f"Collection '{collection_name}' already exists")
    
    def store_radiology_embedding(
        self,
        patient_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        hospital_id: str = "default"
    ) -> str:
        """
        Store radiology embedding with metadata
        
        Args:
            patient_id: Patient identifier
            embedding: Radiology feature vector
            metadata: Additional metadata (study date, modality, etc.)
            hospital_id: Hospital/client identifier
            
        Returns:
            Point ID of stored vector
        """
        point_id = f"rad_{patient_id}_{metadata.get('study_date', 'unknown')}"
        
        payload = {
            "patient_id": patient_id,
            "hospital_id": hospital_id,
            "modality": "radiology",
            **metadata
        }
        
        self.client.upsert(
            collection_name=self.radiology_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        
        logger.debug(f"Stored radiology embedding for patient {patient_id}")
        return point_id
    
    def store_genomics_embedding(
        self,
        patient_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        hospital_id: str = "default"
    ) -> str:
        """
        Store genomics embedding with metadata
        
        Args:
            patient_id: Patient identifier
            embedding: Genomics feature vector
            metadata: Additional metadata (gene panel, variant count, etc.)
            hospital_id: Hospital/client identifier
            
        Returns:
            Point ID of stored vector
        """
        point_id = f"gen_{patient_id}_{metadata.get('panel_type', 'unknown')}"
        
        payload = {
            "patient_id": patient_id,
            "hospital_id": hospital_id,
            "modality": "genomics",
            **metadata
        }
        
        self.client.upsert(
            collection_name=self.genomics_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        
        logger.debug(f"Stored genomics embedding for patient {patient_id}")
        return point_id
    
    def store_fusion_embedding(
        self,
        patient_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        hospital_id: str = "default"
    ) -> str:
        """
        Store multi-modal fused embedding
        
        Args:
            patient_id: Patient identifier
            embedding: Fused feature vector from both modalities
            metadata: Combined metadata
            hospital_id: Hospital/client identifier
            
        Returns:
            Point ID of stored vector
        """
        point_id = f"fusion_{patient_id}_{metadata.get('timestamp', 'unknown')}"
        
        payload = {
            "patient_id": patient_id,
            "hospital_id": hospital_id,
            "modality": "fusion",
            **metadata
        }
        
        self.client.upsert(
            collection_name=self.fusion_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        
        logger.debug(f"Stored fusion embedding for patient {patient_id}")
        return point_id
    
    def search_similar_patients(
        self,
        collection_name: str,
        query_embedding: List[float],
        patient_id: Optional[str] = None,
        limit: int = 10,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar patients based on embeddings
        
        Args:
            collection_name: Which collection to search
            query_embedding: Query vector
            patient_id: Exclude this patient from results
            limit: Maximum number of results
            min_score: Minimum similarity score threshold
            
        Returns:
            List of similar patients with scores and metadata
        """
        # Build filter
        must_conditions = []
        
        if patient_id:
            # Exclude the query patient
            must_conditions.append(
                FieldCondition(
                    key="patient_id",
                    match=MatchValue(value=patient_id)
                )
            )
        
        search_filter = Filter(must=must_conditions) if must_conditions else None
        
        # Perform search
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
            params=SearchParams(
                hnsw_ef=128,
                exact=False
            )
        )
        
        # Format results
        formatted_results = []
        for result in results:
            if result.score >= min_score:
                formatted_results.append({
                    "patient_id": result.payload.get("patient_id"),
                    "hospital_id": result.payload.get("hospital_id"),
                    "score": result.score,
                    "metadata": {k: v for k, v in result.payload.items() 
                                if k not in ["patient_id", "hospital_id", "modality"]}
                })
        
        logger.debug(f"Found {len(formatted_results)} similar patients in {collection_name}")
        return formatted_results
    
    def get_patient_embeddings(
        self,
        patient_id: str,
        include_vectors: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve all embeddings for a specific patient
        
        Args:
            patient_id: Patient identifier
            include_vectors: Whether to include actual embedding vectors
            
        Returns:
            Dictionary with radiology, genomics, and fusion data
        """
        results = {}
        
        for collection_name, modality in [
            (self.radiology_collection, "radiology"),
            (self.genomics_collection, "genomics"),
            (self.fusion_collection, "fusion")
        ]:
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="patient_id",
                            match=MatchValue(value=patient_id)
                        )
                    ]
                ),
                limit=10,
                with_payload=True,
                with_vectors=include_vectors
            )
            
            points = scroll_result[0]
            results[modality] = [
                {
                    "id": point.id,
                    "payload": point.payload,
                    "vector": point.vector if include_vectors else None
                }
                for point in points
            ]
        
        return results
    
    def delete_patient_data(self, patient_id: str) -> bool:
        """
        Delete all data for a patient (for privacy compliance)
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Success status
        """
        for collection_name in [
            self.radiology_collection,
            self.genomics_collection,
            self.fusion_collection
        ]:
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="patient_id",
                            match=MatchValue(value=patient_id)
                        )
                    ]
                )
            )
        
        logger.info(f"Deleted all data for patient {patient_id}")
        return True
    
    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all collections"""
        stats = {}
        
        for collection_name in [
            self.radiology_collection,
            self.genomics_collection,
            self.fusion_collection
        ]:
            try:
                info = self.client.get_collection(collection_name)
                stats[collection_name] = {
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "status": info.status
                }
            except Exception as e:
                stats[collection_name] = {"error": str(e)}
        
        return stats


async def demo_vector_store():
    """Demonstrate vector store functionality"""
    
    # Note: Requires running Qdrant instance
    try:
        store = VectorStoreManager()
        store.create_collections()
        
        # Example: Store synthetic embeddings
        patient_id = "DEMO-001"
        radiology_emb = np.random.rand(768).tolist()
        genomics_emb = np.random.rand(768).tolist()
        fusion_emb = np.random.rand(768).tolist()
        
        store.store_radiology_embedding(
            patient_id=patient_id,
            embedding=radiology_emb,
            metadata={"study_date": "2024-01-15", "modality_type": "CT"}
        )
        
        store.store_genomics_embedding(
            patient_id=patient_id,
            embedding=genomics_emb,
            metadata={"panel_type": "oncology_500", "variant_count": 12}
        )
        
        store.store_fusion_embedding(
            patient_id=patient_id,
            embedding=fusion_emb,
            metadata={"timestamp": "2024-01-15T10:30:00", "model_version": "v1.0"}
        )
        
        # Search for similar patients
        similar = store.search_similar_patients(
            collection_name=store.fusion_collection,
            query_embedding=fusion_emb,
            patient_id=patient_id
        )
        
        print(f"Found {len(similar)} similar patients")
        
        # Get stats
        stats = store.get_collection_stats()
        print(f"\nCollection Stats: {stats}")
        
    except Exception as e:
        logger.warning(f"Vector store demo skipped (Qdrant may not be running): {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_vector_store())

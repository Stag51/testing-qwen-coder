"""
FastAPI REST API for Federated Multi-Modal Diagnostic Agentic Swarm
Provides endpoints for diagnosis, federated learning coordination, and data management
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
from loguru import logger

from config.settings import config
from agentic_swarm.orchestrator import DiagnosticOrchestrator
from federated_learning.server import FederatedServer
from vector_store.qdrant_manager import VectorStoreManager


# Request/Response Models
class DiagnosisRequest(BaseModel):
    """Request model for diagnostic analysis"""
    patient_id: str = Field(..., description="Patient identifier")
    radiology_findings: str = Field(..., description="Summary of radiological findings")
    genomics_findings: str = Field(..., description="Summary of genomic findings")
    clinical_history: str = Field(..., description="Patient clinical history")


class DiagnosisResponse(BaseModel):
    """Response model for diagnostic analysis"""
    patient_id: str
    final_report: str
    diagnosis_hypothesis: List[str]
    confidence_score: float
    agent_analyses: Dict[str, str]


class FederatedUpdateRequest(BaseModel):
    """Request model for submitting federated learning updates"""
    client_id: int = Field(..., description="Hospital client ID")
    round_number: int = Field(..., description="Federated round number")
    num_samples: int = Field(..., description="Number of training samples")
    # In practice, weights would be transmitted separately or via secure channel


class FederatedUpdateResponse(BaseModel):
    """Response model for federated update submission"""
    success: bool
    message: str
    current_round: int
    clients_received: int


class PatientSearchRequest(BaseModel):
    """Request model for similar patient search"""
    patient_id: str
    modality: str = Field(default="fusion", description="radiology, genomics, or fusion")
    limit: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.7, ge=0, le=1)


class PatientSearchResponse(BaseModel):
    """Response model for similar patient search"""
    query_patient_id: str
    similar_patients: List[Dict[str, Any]]
    total_found: int


# Initialize FastAPI app
app = FastAPI(
    title=config.api.api_title,
    version=config.api.api_version,
    description="""
## Federated Multi-Modal Diagnostic Agentic Swarm API

This API provides:
- **Diagnostic Analysis**: Multi-modal AI-powered diagnosis using expert agent swarm
- **Federated Learning**: Coordinate privacy-preserving model training across hospitals
- **Vector Search**: Find similar patients based on multi-modal embeddings
- **FHIR Integration**: Healthcare data interoperability

### Privacy & Security
- Raw patient data never leaves hospital premises
- Only encrypted model gradients are shared
- Differential privacy ensures individual privacy protection
    """
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized lazily)
_diagnostic_orchestrator: Optional[DiagnosticOrchestrator] = None
_federated_server: Optional[FederatedServer] = None
_vector_store: Optional[VectorStoreManager] = None


def get_diagnostic_orchestrator() -> DiagnosticOrchestrator:
    """Get or create diagnostic orchestrator instance"""
    global _diagnostic_orchestrator
    if _diagnostic_orchestrator is None:
        _diagnostic_orchestrator = DiagnosticOrchestrator()
    return _diagnostic_orchestrator


def get_federated_server() -> FederatedServer:
    """Get or create federated server instance"""
    global _federated_server
    if _federated_server is None:
        _federated_server = FederatedServer()
    return _federated_server


def get_vector_store() -> VectorStoreManager:
    """Get or create vector store instance"""
    global _vector_store
    if _vector_store is None:
        try:
            _vector_store = VectorStoreManager()
            _vector_store.create_collections()
        except Exception as e:
            logger.warning(f"Vector store initialization failed: {e}")
            raise
    return _vector_store


# Startup and Shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Federated Diagnostic Swarm API...")
    
    # Pre-initialize core services
    get_diagnostic_orchestrator()
    get_federated_server()
    
    logger.success("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Federated Diagnostic Swarm API...")


# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check"""
    return {
        "service": "Federated Multi-Modal Diagnostic Agentic Swarm",
        "version": config.api.api_version,
        "status": "healthy"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check endpoint"""
    health_status = {
        "api": "healthy",
        "diagnostic_orchestrator": "healthy" if _diagnostic_orchestrator else "not_initialized",
        "federated_server": "healthy" if _federated_server else "not_initialized",
        "vector_store": "unknown"  # Would need to ping Qdrant
    }
    return health_status


@app.post("/diagnose", response_model=DiagnosisResponse, tags=["Diagnosis"])
async def run_diagnosis(request: DiagnosisRequest):
    """
    Run multi-modal diagnostic analysis using the agentic swarm
    
    This endpoint orchestrates multiple expert agents (Radiology, Genomics, 
    Oncology, Pathology) to provide a comprehensive diagnostic assessment.
    """
    try:
        orchestrator = get_diagnostic_orchestrator()
        
        # Run async diagnosis
        result = await orchestrator.run_diagnosis(
            patient_id=request.patient_id,
            radiology_findings=request.radiology_findings,
            genomics_findings=request.genomics_findings,
            clinical_history=request.clinical_history
        )
        
        return DiagnosisResponse(
            patient_id=result['patient_id'],
            final_report=result['final_report'],
            diagnosis_hypothesis=result['diagnosis_hypothesis'],
            confidence_score=result['confidence_score'],
            agent_analyses=result['agent_outputs']
        )
    
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/federated/update", response_model=FederatedUpdateResponse, tags=["Federated Learning"])
async def submit_federated_update(request: FederatedUpdateRequest):
    """
    Submit model weight update from a hospital client
    
    In production, actual weights would be transmitted via secure channel
    with encryption and differential privacy applied.
    """
    try:
        server = get_federated_server()
        
        # Placeholder: In production, receive actual encrypted weights
        # For now, just track that an update was received
        dummy_weights = {}  # Would contain actual model updates
        
        success = server.receive_client_update(
            client_id=request.client_id,
            weights=dummy_weights,
            num_samples=request.num_samples
        )
        
        # Check if round is complete
        if server.check_round_complete():
            server.finalize_round(apply_dp=True)
        
        return FederatedUpdateResponse(
            success=success,
            message=f"Update received from client {request.client_id}",
            current_round=server.current_round,
            clients_received=len(server.client_updates)
        )
    
    except Exception as e:
        logger.error(f"Federated update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/federated/status", tags=["Federated Learning"])
async def get_federated_status():
    """Get current federated learning status"""
    server = get_federated_server()
    
    return {
        "current_round": server.current_round,
        "clients_registered": len(server.client_updates),
        "expected_clients": server.num_clients,
        "round_complete": server.check_round_complete(),
        "model_parameters": sum(p.numel() for p in server.model.parameters())
    }


@app.post("/federated/get_weights", tags=["Federated Learning"])
async def get_global_weights(client_id: int = Form(...)):
    """
    Get current global model weights for a client
    
    Clients call this to receive aggregated model before local training.
    """
    server = get_federated_server()
    
    # Return serialized state dict
    # In production, this would be encrypted for the specific client
    state_dict = server.get_model_state()
    
    # Convert tensors to serializable format (simplified)
    serializable_weights = {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)}
        for k, v in state_dict.items()
    }
    
    return {
        "client_id": client_id,
        "round": server.current_round,
        "num_parameters": len(state_dict),
        "weights_summary": serializable_weights
    }


@app.post("/vector/search", response_model=PatientSearchResponse, tags=["Vector Search"])
async def search_similar_patients(request: PatientSearchRequest):
    """
    Search for similar patients based on multi-modal embeddings
    
    Useful for finding cases with similar radiological/genomic profiles
    for comparative analysis and treatment planning.
    """
    try:
        store = get_vector_store()
        
        # This would require the query patient's embedding
        # For now, return placeholder response
        return PatientSearchResponse(
            query_patient_id=request.patient_id,
            similar_patients=[],
            total_found=0
        )
    
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/vector/patient/{patient_id}", tags=["Vector Search", "Privacy"])
async def delete_patient_vectors(patient_id: str):
    """
    Delete all vector embeddings for a patient
    
    Supports privacy compliance (GDPR right to erasure, HIPAA).
    """
    try:
        store = get_vector_store()
        success = store.delete_patient_data(patient_id)
        
        return {
            "patient_id": patient_id,
            "deleted": success,
            "message": "Patient data deleted from vector store" if success else "Deletion failed"
        }
    
    except Exception as e:
        logger.error(f"Patient deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vector/stats", tags=["Vector Search"])
async def get_vector_stats():
    """Get vector store statistics"""
    try:
        store = get_vector_store()
        stats = store.get_collection_stats()
        return stats
    except Exception as e:
        return {"error": f"Vector store not available: {str(e)}"}


@app.get("/fhir/patient/{patient_id}", tags=["FHIR Integration"])
async def get_fhir_patient(patient_id: str):
    """
    Retrieve patient data from FHIR server
    
    Integrates with healthcare systems using FHIR standards.
    """
    # Placeholder - would integrate with actual FHIR server
    return {
        "resourceType": "Patient",
        "id": patient_id,
        "note": "FHIR integration requires configured FHIR server"
    }


@app.post("/fhir/sync", tags=["FHIR Integration"])
async def sync_fhir_data(resource_type: str = Form(...), patient_id: str = Form(...)):
    """
    Sync data from FHIR server for a patient
    
    Supported resource types: Patient, Observation, DiagnosticReport, 
    ImagingStudy, MolecularSequence
    """
    # Placeholder - would integrate with actual FHIR server
    return {
        "synced": True,
        "resource_type": resource_type,
        "patient_id": patient_id,
        "note": "FHIR sync requires configured FHIR server"
    }


# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug
    )

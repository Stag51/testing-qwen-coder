# Federated Multi-Modal Diagnostic Agentic Swarm - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Core Components](#core-components)
5. [API Reference](#api-reference)
6. [Federated Learning Guide](#federated-learning-guide)
7. [Agentic Swarm Workflow](#agentic-swarm-workflow)
8. [Data Ingestion Pipeline](#data-ingestion-pipeline)
9. [Vector Store Management](#vector-store-management)
10. [Security & Privacy](#security--privacy)
11. [Deployment Guide](#deployment-guide)
12. [Troubleshooting](#troubleshooting)
13. [Contributing](#contributing)
14. [License](#license)

---

## Overview

### What is This System?

The **Federated Multi-Modal Diagnostic Agentic Swarm** is a privacy-preserving healthcare AI system that enables hospitals to collaboratively train diagnostic models on sensitive radiologic and genomic data without sharing raw patient information. The system uses an agentic swarm architecture where specialized "Expert Agents" collaborate to provide comprehensive diagnostic assessments.

### Key Value Propositions

- **Privacy-Preserving**: Raw patient data never leaves hospital premises
- **Multi-Modal Analysis**: Combines radiology (DICOM, MRI, CT) and genomics data
- **Collaborative Learning**: Hospitals benefit from collective intelligence without data sharing
- **Expert Agent Swarm**: Multiple specialized AI agents provide comprehensive diagnosis
- **Healthcare Standards**: FHIR-compliant for seamless integration with existing systems

### Use Cases

1. **Multi-Institutional Research**: Collaborate on rare disease studies without data sharing barriers
2. **Diagnostic Support**: Provide clinicians with AI-powered multi-modal diagnostic insights
3. **Personalized Medicine**: Integrate genomic and imaging data for tailored treatment recommendations
4. **Quality Assurance**: Benchmark local model performance against global standards

---

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HOSPITAL NODE A                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ DICOM/PACS   │    │ Genomic      │    │ EHR/FHIR     │              │
│  │ Scanner      │    │ Sequencer    │    │ System       │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         └───────────────────┼───────────────────┘                       │
│                             │                                           │
│                  ┌──────────▼──────────┐                               │
│                  │  Local Feature       │                               │
│                  │  Extraction          │                               │
│                  └──────────┬──────────┘                               │
│                             │                                           │
│                  ┌──────────▼──────────┐                               │
│                  │  Federated Client    │                               │
│                  │  (Encrypted Gradients)│                              │
│                  └──────────┬──────────┘                               │
└─────────────────────────────┼───────────────────────────────────────────┘
                              │
                    ╔═════════▼═════════╗
                    ║  Secure Aggregation ║
                    ║  Server            ║
                    ╚═════════┬═════════╝
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Global Model  │    │ Agentic       │    │ Vector        │
│ Update        │    │ Orchestrator  │    │ Store (Qdrant)│
└───────────────┘    └───────┬───────┘    └───────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │Radiology │  │Genomics  │  │Oncology  │
       │ Agent    │  │ Agent    │  │ Agent    │
       └──────────┘  └──────────┘  └──────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                      ┌──────▼──────┐
                      │ Pathology   │
                      │ Agent       │
                      └──────┬──────┘
                             │
                      ┌──────▼──────┐
                      │ Final Report│
                      │ Generation  │
                      └─────────────┘
```

### End-to-End Data Flow

1. **Regional Sensor/DICOM Ingest**: Medical imaging devices and genomic sequencers generate raw data
2. **Local Feature Extraction**: Hospital nodes extract features locally without exposing raw data
3. **Encrypted Gradient Sync**: Model updates are encrypted and differentially privatized
4. **Global Model Update**: Central server aggregates updates using secure aggregation
5. **Agentic Task Dispatch**: Orchestrator assigns analysis tasks to expert agents
6. **Cross-Modal Fusion**: Multi-modal embeddings are fused for comprehensive analysis
7. **Clinical Support**: Final diagnostic report delivered to clinicians

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Core Language | Python 3.9+ | Primary development language |
| Deep Learning | PyTorch 2.0+ | Neural network framework |
| Federated Learning | PySyft 0.8+ | Privacy-preserving computations |
| Agent Orchestration | LangGraph | Workflow management for agents |
| Vector Database | Qdrant 1.7+ | Multi-modal embedding storage |
| Biomedical Models | NVIDIA BioNeMo | Domain-specific foundation models |
| REST API | FastAPI 0.100+ | High-performance API framework |
| Healthcare Standard | FHIR R4 | Interoperability with EHR systems |
| Logging | Loguru | Structured logging |
| Configuration | Pydantic Settings | Type-safe configuration management |

---

## Installation & Setup

### Prerequisites

- **Python**: Version 3.9 or higher
- **GPU**: NVIDIA GPU with CUDA 11.8+ (recommended for training)
- **Memory**: Minimum 16GB RAM, 32GB recommended
- **Storage**: 100GB+ for model weights and vector database
- **Docker**: Optional, for containerized deployment

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/federated-diagnostic-swarm.git
cd federated-diagnostic-swarm
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

Create a `.env` file in the project root:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Federated Learning
NUM_CLIENTS=5
BATCH_SIZE=32
LEARNING_RATE=0.001
DP_EPSILON=1.0
DP_DELTA=1e-5

# Qdrant Vector Store
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your-api-key-here

# NVIDIA BioNeMo (if using)
BIONEMO_API_KEY=your-bionemo-key
BIONEMO_ENDPOINT=https://api.bionemo.nvidia.com

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### Step 5: Start Qdrant (Vector Database)

Using Docker:

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  --name qdrant \
  qdrant/qdrant:latest
```

### Step 6: Verify Installation

```bash
python -c "from models.diagnostic_model import FederatedDiagnosticModel; print('✓ Model import successful')"
python -c "from agentic_swarm.orchestrator import DiagnosticOrchestrator; print('✓ Orchestrator import successful')"
python -c "from federated_learning.server import FederatedServer; print('✓ Server import successful')"
```

---

## Core Components

### 1. Federated Learning Module

**Location**: `federated_learning/`

#### FederatedServer

Coordinates secure aggregation of model updates from hospital clients.

**Key Features**:
- Weighted averaging based on client dataset sizes
- Differential privacy noise injection
- Round management and completion tracking
- Model state serialization/deserialization

**Usage Example**:

```python
from federated_learning.server import FederatedServer
from models.diagnostic_model import FederatedDiagnosticModel

# Initialize server
model = FederatedDiagnosticModel()
server = FederatedServer(model)

# Receive update from client
client_id = 1
weights = {...}  # Model weight updates
num_samples = 1000

server.receive_client_update(client_id, weights, num_samples)

# Check if round complete and aggregate
if server.check_round_complete():
    aggregated_updates, success = server.finalize_round(apply_dp=True)
```

#### FederatedClient

Hospital-side client for local training and gradient computation.

**Key Features**:
- Local model training on private data
- Gradient clipping and differential privacy
- Secure weight update computation
- Synthetic data simulation for testing

**Usage Example**:

```python
from federated_learning.client import FederatedClient
from torch.utils.data import DataLoader

# Initialize client
client = FederatedClient(client_id=1, device='cuda')

# Receive global weights
global_weights = server.get_model_state()
client.set_global_weights(global_weights)

# Train locally
updates = client.train_local(dataloader, epochs=5, learning_rate=0.001)

# Apply differential privacy
dp_updates = client.apply_differential_privacy(
    updates, epsilon=1.0, delta=1e-5
)

# Send to server
server.receive_client_update(client.client_id, dp_updates, num_samples)
```

### 2. Agentic Swarm Module

**Location**: `agentic_swarm/`

#### Expert Agents

Four specialized agents collaborate for comprehensive diagnosis:

1. **RadiologyAgent** ("Dr. Radio"): Analyzes medical imaging findings
2. **GenomicsAgent** ("Dr. Geno"): Interprets genomic variants and mutations
3. **OncologyAgent** ("Dr. Onco"): Synthesizes multi-modal data for cancer diagnosis
4. **PathologyAgent** ("Dr. Patho"): Correlates imaging and molecular findings

#### DiagnosticOrchestrator

Manages the workflow using LangGraph state machine.

**Workflow States**:
```python
class DiagnosticState(TypedDict):
    patient_id: str
    radiology_findings: str
    genomics_findings: str
    clinical_history: str
    agent_outputs: Dict[str, str]
    diagnosis_hypothesis: List[str]
    final_report: str
    confidence_score: float
    messages: List[BaseMessage]
```

**Usage Example**:

```python
from agentic_swarm.orchestrator import DiagnosticOrchestrator

orchestrator = DiagnosticOrchestrator()

result = await orchestrator.run_diagnosis(
    patient_id="PATIENT-001",
    radiology_findings="3.2 cm spiculated mass in right upper lobe",
    genomics_findings="EGFR exon 19 deletion detected",
    clinical_history="67-year-old former smoker with persistent cough"
)

print(result['final_report'])
print(f"Confidence: {result['confidence_score']:.2f}")
```

### 3. Multi-Modal Model

**Location**: `models/diagnostic_model.py`

#### Architecture

```
Radiology Input (3D CNN) ──► Embedding (768) ──┐
                                               ├──► Cross-Modal Attention ──► Fusion Layers ──► Classification
Genomics Input (Transformer) ──► Embedding (768) ──┘
```

**Components**:
- **RadiologyEncoder**: 3D CNN for volumetric medical images
- **GenomicsEncoder**: Transformer encoder for sequence data
- **MultiModalFusion**: Cross-modal attention mechanism
- **Classifier**: Multi-layer perceptron for diagnosis prediction

### 4. Vector Store Manager

**Location**: `vector_store/qdrant_manager.py`

Manages multi-modal patient embeddings in Qdrant.

**Collections**:
- `radiology_embeddings`: Imaging feature vectors
- `genomics_embeddings`: Genomic profile vectors
- `fusion_embeddings`: Combined multi-modal vectors

**Usage Example**:

```python
from vector_store.qdrant_manager import VectorStoreManager

store = VectorStoreManager()
store.create_collections()

# Store patient embedding
store.store_embedding(
    patient_id="PATIENT-001",
    embedding=radiology_embedding,
    modality="radiology",
    metadata={"diagnosis": "lung_cancer", "stage": "IIA"}
)

# Search similar patients
similar = store.search_similar(
    query_embedding=query_emb,
    modality="radiology",
    limit=10,
    min_score=0.7
)
```

### 5. Data Ingestion

**Location**: `data_ingestion/`

#### DICOM Loader

Handles medical image loading and preprocessing.

**Features**:
- DICOM standard compliance
- Multi-frame support
- Intensity normalization
- Resampling to standard spacing

**Usage**:

```python
from data_ingestion.dicom_loader import DICOMLoader

loader = DICOMLoader()
volume, metadata = loader.load_series("path/to/dicom/series")
processed = loader.preprocess(volume, target_spacing=(1.0, 1.0, 1.0))
```

### 6. FastAPI REST API

**Location**: `api/main.py`

Provides HTTP endpoints for all system functionality.

**Base URL**: `http://localhost:8000`

**Interactive Docs**: `http://localhost:8000/docs`

---

## API Reference

### Health Endpoints

#### GET `/`

Root endpoint - API health check.

**Response**:
```json
{
  "service": "Federated Multi-Modal Diagnostic Agentic Swarm",
  "version": "1.0.0",
  "status": "healthy"
}
```

#### GET `/health`

Detailed health check.

**Response**:
```json
{
  "api": "healthy",
  "diagnostic_orchestrator": "healthy",
  "federated_server": "healthy",
  "vector_store": "healthy"
}
```

### Diagnosis Endpoints

#### POST `/diagnose`

Run multi-modal diagnostic analysis.

**Request Body**:
```json
{
  "patient_id": "PATIENT-001",
  "radiology_findings": "3.2 cm spiculated mass in right upper lobe",
  "genomics_findings": "EGFR exon 19 deletion, TP53 R175H mutation",
  "clinical_history": "67-year-old former smoker with persistent cough"
}
```

**Response**:
```json
{
  "patient_id": "PATIENT-001",
  "final_report": "...",
  "diagnosis_hypothesis": [
    "Primary malignancy (based on imaging and genomic profile)",
    "Metastatic disease (to be ruled out)",
    "Benign condition with atypical features"
  ],
  "confidence_score": 0.85,
  "agent_analyses": {
    "Radiology": "...",
    "Genomics": "...",
    "Oncology": "...",
    "Pathology": "..."
  }
}
```

### Federated Learning Endpoints

#### POST `/federated/update`

Submit model weight update from hospital client.

**Request Body**:
```json
{
  "client_id": 1,
  "round_number": 5,
  "num_samples": 1000
}
```

**Response**:
```json
{
  "success": true,
  "message": "Update received from client 1",
  "current_round": 5,
  "clients_received": 3
}
```

#### GET `/federated/status`

Get current federated learning status.

**Response**:
```json
{
  "current_round": 5,
  "clients_registered": 3,
  "expected_clients": 5,
  "round_complete": false,
  "model_parameters": 15000000
}
```

#### POST `/federated/get_weights`

Get current global model weights.

**Query Parameter**: `client_id` (form data)

**Response**:
```json
{
  "client_id": 1,
  "round": 5,
  "num_parameters": 150,
  "weights_summary": {
    "radiology_encoder.conv_layers.0.weight": {
      "shape": [64, 1, 3, 3, 3],
      "dtype": "torch.float32"
    }
  }
}
```

### Vector Search Endpoints

#### POST `/vector/search`

Search for similar patients.

**Request Body**:
```json
{
  "patient_id": "PATIENT-001",
  "modality": "fusion",
  "limit": 10,
  "min_score": 0.7
}
```

**Response**:
```json
{
  "query_patient_id": "PATIENT-001",
  "similar_patients": [...],
  "total_found": 5
}
```

#### DELETE `/vector/patient/{patient_id}`

Delete patient data (privacy compliance).

**Response**:
```json
{
  "patient_id": "PATIENT-001",
  "deleted": true,
  "message": "Patient data deleted from vector store"
}
```

### FHIR Integration Endpoints

#### GET `/fhir/patient/{patient_id}`

Retrieve patient data from FHIR server.

#### POST `/fhir/sync`

Sync data from FHIR server.

**Form Data**:
- `resource_type`: Patient, Observation, DiagnosticReport, etc.
- `patient_id`: Patient identifier

---

## Federated Learning Guide

### How Federated Learning Works

1. **Initialization**: Server initializes global model and distributes weights to clients
2. **Local Training**: Each hospital trains on local data without sharing raw data
3. **Gradient Computation**: Clients compute weight updates (gradients)
4. **Privacy Protection**: Differential privacy noise added to updates
5. **Secure Aggregation**: Server aggregates updates using weighted averaging
6. **Global Update**: Global model updated with aggregated gradients
7. **Iteration**: Process repeats for multiple rounds

### Differential Privacy

The system implements Gaussian differential privacy:

**Parameters**:
- **ε (epsilon)**: Privacy budget (default: 1.0)
  - Lower = more privacy, less accuracy
  - Higher = less privacy, more accuracy
- **δ (delta)**: Privacy failure probability (default: 1e-5)
- **Sensitivity**: Maximum change in output (default: 1.0)

**Noise Scale Calculation**:
```python
sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
```

### Gradient Clipping

To bound sensitivity, gradients are clipped:

```python
update_norm = torch.norm(update)
if update_norm > clip_norm:
    update = update * (clip_norm / update_norm)
```

### Best Practices

1. **Client Selection**: Ensure diverse hospital representation
2. **Round Size**: Balance between convergence speed and privacy
3. **Learning Rate**: Use lower LR than centralized training
4. **Monitoring**: Track both global and local model performance
5. **Anomaly Detection**: Identify and exclude malicious clients

### Troubleshooting Federated Learning

**Issue**: Slow convergence
- **Solution**: Increase number of clients per round, adjust learning rate

**Issue**: Privacy-accuracy tradeoff too severe
- **Solution**: Increase epsilon slightly, reduce delta

**Issue**: Client dropout
- **Solution**: Implement asynchronous aggregation, reduce expected clients

---

## Agentic Swarm Workflow

### Agent Collaboration Flow

```
                    ┌─────────────────┐
                    │  Patient Case   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Radiology Agent │
                    │  (Dr. Radio)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Genomics Agent  │
                    │  (Dr. Geno)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Oncology Agent  │
                    │  (Dr. Onco)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Pathology Agent │
                    │  (Dr. Patho)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Report Generator│
                    └─────────────────┘
```

### Customizing Agents

To add a new expert agent:

```python
class CardiologyAgent(ExpertAgent):
    def __init__(self):
        super().__init__(
            name="Dr. Cardio",
            specialty="Cardiology",
            system_prompt="You are an expert cardiologist..."
        )
    
    async def analyze(self, state: DiagnosticState) -> str:
        # Implement analysis logic
        return analysis

# Add to orchestrator
self.agents['Cardiology'] = CardiologyAgent()
workflow.add_node("cardiology_analysis", self._run_cardiology_agent)
```

### Agent Communication Pattern

Agents can access previous analyses through the shared state:

```python
async def analyze(self, state: DiagnosticState) -> str:
    radio_analysis = state['agent_outputs'].get('Radiology', '')
    genomic_analysis = state['agent_outputs'].get('Genomics', '')
    
    # Build upon previous analyses
    synthesis = f"""
    Based on radiology findings: {radio_analysis}
    And genomic findings: {genomic_analysis}
    
    My integrated assessment is...
    """
    return synthesis
```

---

## Data Ingestion Pipeline

### DICOM Processing

**Supported Modalities**:
- CT (Computed Tomography)
- MRI (Magnetic Resonance Imaging)
- PET (Positron Emission Tomography)
- CR/DR (X-Ray)
- US (Ultrasound)

**Preprocessing Steps**:
1. Load DICOM series
2. Sort by instance number
3. Convert to Hounsfield units (CT) or normalized intensity (MRI)
4. Resample to isotropic spacing
5. Crop/pad to standard size
6. Normalize to [0, 1] range

### Genomic Data Processing

**Supported Formats**:
- VCF (Variant Call Format)
- FASTQ (Raw sequencing reads)
- BAM/SAM (Aligned reads)
- MAF (Mutation Annotation Format)

**Preprocessing Steps**:
1. Parse variant calls
2. Encode nucleotides (A=0, C=1, G=2, T=3)
3. Pad/truncate to fixed length
4. Generate sequence embeddings

### FHIR Integration

**Supported Resources**:
- Patient: Demographics and identifiers
- Observation: Clinical measurements
- DiagnosticReport: Lab and imaging reports
- ImagingStudy: DICOM references
- MolecularSequence: Genomic data

**Example FHIR Query**:

```python
import requests

fhir_base = "https://fhir-server.example.com/r4"

# Get patient observations
response = requests.get(
    f"{fhir_base}/Observation",
    params={"patient": "patient-123"},
    headers={"Authorization": f"Bearer {token}"}
)
observations = response.json()
```

---

## Vector Store Management

### Qdrant Configuration

**Connection Settings**:
```python
from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    port=6333,
    api_key="your-api-key",
    https=False
)
```

### Collection Schema

```python
collection_config = {
    "vectors": {
        "size": 768,
        "distance": "Cosine"
    },
    "hnsw_config": {
        "m": 16,
        "ef_construct": 100
    },
    "optimizers_config": {
        "indexing_threshold": 20000
    }
}
```

### Embedding Storage

Each patient can have multiple embeddings:

```python
# Store radiology embedding
store.store_embedding(
    collection_name="radiology_embeddings",
    patient_id="PATIENT-001",
    vector=radiology_embedding.tolist(),
    payload={
        "modality": "CT",
        "body_part": "chest",
        "findings": "lung nodule",
        "timestamp": "2024-01-15T10:30:00Z"
    }
)

# Store genomics embedding
store.store_embedding(
    collection_name="genomics_embeddings",
    patient_id="PATIENT-001",
    vector=genomics_embedding.tolist(),
    payload={
        "panel": "oncology_500",
        "variants": ["EGFR_L858R", "TP53_R175H"],
        "tmb": 8.5,
        "timestamp": "2024-01-15T11:00:00Z"
    }
)
```

### Similarity Search

```python
# Find similar patients
results = store.search_similar(
    collection_name="fusion_embeddings",
    query_vector=query_embedding.tolist(),
    limit=10,
    score_threshold=0.7,
    filter_payload={
        "diagnosis": {"$eq": "lung_adenocarcinoma"}
    }
)

for result in results:
    print(f"Patient: {result.payload['patient_id']}, Score: {result.score}")
```

### Privacy Compliance

Implement right to erasure:

```python
# Delete all patient data
store.delete_patient_data(patient_id="PATIENT-001")

# Verify deletion
exists = store.patient_exists("PATIENT-001")
assert not exists
```

---

## Security & Privacy

### Privacy Guarantees

1. **Data Locality**: Raw patient data never leaves hospital premises
2. **Differential Privacy**: Mathematical guarantee of individual privacy
3. **Encryption**: All communications encrypted (TLS 1.3)
4. **Access Control**: Role-based access control (RBAC)
5. **Audit Logging**: Comprehensive activity logging

### Threat Model

**Protected Against**:
- Honest-but-curious server
- Malicious clients attempting to infer others' data
- External attackers intercepting communications
- Model inversion attacks

**Not Protected Against**:
- Compromised hospital infrastructure
- Insider threats at hospital level
- Side-channel attacks (timing, power)

### Security Best Practices

1. **Network Security**:
   - Use VPN or private network for inter-hospital communication
   - Implement firewall rules
   - Enable DDoS protection

2. **Authentication**:
   - Require mutual TLS authentication
   - Implement OAuth 2.0 for API access
   - Rotate credentials regularly

3. **Data Protection**:
   - Encrypt data at rest (AES-256)
   - Use secure enclaves for sensitive computations
   - Implement data retention policies

4. **Compliance**:
   - HIPAA compliance for US healthcare
   - GDPR compliance for EU data
   - Local healthcare regulations

### Audit Logging

All operations are logged:

```python
from loguru import logger

logger.info("Federated update received", extra={
    "client_id": client_id,
    "round": round_num,
    "num_samples": num_samples,
    "timestamp": datetime.utcnow().isoformat()
})
```

---

## Deployment Guide

### Development Deployment

For local development and testing:

```bash
# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Start API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run federated training simulation
python federated_learning/server.py
python federated_learning/client.py
```

### Production Deployment

#### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - QDRANT_HOST=qdrant
      - API_HOST=0.0.0.0
      - API_PORT=8000
    depends_on:
      - qdrant
    volumes:
      - ./logs:/app/logs

  federated-server:
    build: .
    command: python -m federated_learning.server
    environment:
      - QDRANT_HOST=qdrant
    depends_on:
      - qdrant

volumes:
  qdrant_storage:
```

Deploy:

```bash
docker-compose up -d
```

#### Kubernetes Deployment

For large-scale deployments:

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: diagnostic-swarm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: diagnostic-swarm
  template:
    metadata:
      labels:
        app: diagnostic-swarm
    spec:
      containers:
      - name: api
        image: your-registry/diagnostic-swarm:latest
        ports:
        - containerPort: 8000
        env:
        - name: QDRANT_HOST
          value: "qdrant-service"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

### Monitoring & Observability

#### Prometheus Metrics

Export metrics for monitoring:

```python
from prometheus_fastapi_instrumentator import Instrumentator

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
```

#### Health Checks

Configure liveness and readiness probes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

### Scaling Considerations

1. **Horizontal Scaling**: Deploy multiple API instances behind load balancer
2. **Database Scaling**: Use Qdrant cluster mode for high availability
3. **Federated Learning**: Implement hierarchical aggregation for many clients
4. **Caching**: Cache frequently accessed model weights and embeddings

---

## Troubleshooting

### Common Issues

#### Issue: Import Errors

**Symptom**: `ModuleNotFoundError: No module named '...'`

**Solution**:
```bash
pip install -r requirements.txt --upgrade
python -m pip install --upgrade pip
```

#### Issue: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce batch size in config
- Use gradient accumulation
- Enable mixed precision training
- Use CPU for inference if GPU memory insufficient

#### Issue: Qdrant Connection Failed

**Symptom**: `ConnectionRefusedError: Could not connect to Qdrant`

**Solution**:
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker restart qdrant

# Check logs
docker logs qdrant
```

#### Issue: Slow Federated Convergence

**Symptom**: Model accuracy not improving after many rounds

**Solution**:
- Increase number of participating clients
- Adjust learning rate (try 0.0001 instead of 0.001)
- Reduce differential privacy noise (increase epsilon)
- Check for data heterogeneity across clients

#### Issue: Agent Workflow Hangs

**Symptom**: Diagnostic workflow doesn't complete

**Solution**:
- Check LangGraph version compatibility
- Verify async/await usage
- Add timeout to workflow execution
- Check logs for errors in agent methods

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
```

View detailed logs:

```bash
tail -f logs/app.log
```

### Performance Profiling

Profile slow operations:

```python
import cProfile
import pstats

 profiler = cProfile.Profile()
profiler.enable()

# Run operation
result = await orchestrator.run_diagnosis(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

---

## Contributing

### Code Style

Follow PEP 8 guidelines:

```bash
# Install linting tools
pip install black flake8 mypy

# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### Testing

Run tests:

```bash
pytest tests/ -v --cov=.
```

Write unit tests:

```python
def test_federated_aggregation():
    server = FederatedServer()
    
    # Simulate client updates
    for client_id in range(5):
        weights = {...}
        server.receive_client_update(client_id, weights, 1000)
    
    # Aggregate
    updates, success = server.finalize_round()
    
    assert success
    assert len(updates) > 0
```

### Pull Request Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Documentation

Update documentation for all new features:
- Add docstrings to functions
- Update API reference
- Add examples to relevant sections

---

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contact & Support

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and community support
- **Email**: support@federated-diagnostic.example.com
- **Documentation**: https://docs.federated-diagnostic.example.com

---

*Last Updated: January 2024*
*Version: 1.0.0*

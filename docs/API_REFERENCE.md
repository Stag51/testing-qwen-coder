# API Reference Documentation

## Base URL

```
http://localhost:8000
```

## Interactive Documentation

Once the server is running, access interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Authentication

Currently, the API does not require authentication for local development. For production deployments, implement OAuth 2.0 or API key authentication.

---

## Endpoints

### Health & Status

#### GET `/`

Root endpoint providing basic service information.

**Response**:
```json
{
  "service": "Federated Multi-Modal Diagnostic Agentic Swarm",
  "version": "1.0.0",
  "status": "healthy"
}
```

**Status Codes**:
- `200 OK`: Service is running

---

#### GET `/health`

Detailed health check for all system components.

**Response**:
```json
{
  "api": "healthy",
  "diagnostic_orchestrator": "healthy",
  "federated_server": "healthy",
  "vector_store": "healthy"
}
```

**Status Codes**:
- `200 OK`: All components healthy
- `503 Service Unavailable`: One or more components unhealthy

---

### Diagnosis Endpoints

#### POST `/diagnose`

Run comprehensive multi-modal diagnostic analysis using the agentic swarm.

**Request Body**:
```json
{
  "patient_id": "PATIENT-001",
  "radiology_findings": "3.2 cm spiculated mass in right upper lobe with associated lymphadenopathy",
  "genomics_findings": "EGFR exon 19 deletion, TP53 R175H mutation, KRAS wild-type, PD-L1 TPS 45%",
  "clinical_history": "67-year-old former smoker (40 pack-year) presenting with persistent cough and weight loss"
}
```

**Request Schema**:
```typescript
{
  patient_id: string (required),      // Unique patient identifier
  radiology_findings: string (required),  // Summary of imaging findings
  genomics_findings: string (required),   // Summary of genomic variants
  clinical_history: string (required)     // Patient clinical background
}
```

**Response**:
```json
{
  "patient_id": "PATIENT-001",
  "final_report": "================================================================================\n                    MULTI-MODAL DIAGNOSTIC REPORT\n================================================================================\n\nPatient ID: PATIENT-001\n...",
  "diagnosis_hypothesis": [
    "Primary malignancy (based on imaging and genomic profile)",
    "Metastatic disease (to be ruled out)",
    "Benign condition with atypical features"
  ],
  "confidence_score": 0.85,
  "agent_analyses": {
    "Radiology": "RADIOLOGY ANALYSIS by Dr. Radio:\n=====================================\n...",
    "Genomics": "GENOMICS ANALYSIS by Dr. Geno:\n=====================================\n...",
    "Oncology": "ONCOLOGY SYNTHESIS by Dr. Onco:\n=====================================\n...",
    "Pathology": "PATHOLOGY CORRELATION by Dr. Patho:\n=====================================\n..."
  }
}
```

**Response Schema**:
```typescript
{
  patient_id: string,
  final_report: string,              // Complete diagnostic report
  diagnosis_hypothesis: string[],    // List of potential diagnoses
  confidence_score: number,          // Confidence level (0.0-1.0)
  agent_analyses: {                  // Individual agent outputs
    Radiology: string,
    Genomics: string,
    Oncology: string,
    Pathology: string
  }
}
```

**Status Codes**:
- `200 OK`: Analysis completed successfully
- `400 Bad Request`: Invalid request body
- `500 Internal Server Error`: Analysis failed

**Example (curl)**:
```bash
curl -X POST "http://localhost:8000/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PATIENT-001",
    "radiology_findings": "3.2 cm spiculated mass",
    "genomics_findings": "EGFR exon 19 deletion",
    "clinical_history": "67-year-old former smoker"
  }'
```

**Example (Python)**:
```python
import requests

response = requests.post(
    "http://localhost:8000/diagnose",
    json={
        "patient_id": "PATIENT-001",
        "radiology_findings": "3.2 cm spiculated mass",
        "genomics_findings": "EGFR exon 19 deletion",
        "clinical_history": "67-year-old former smoker"
    }
)

result = response.json()
print(result['final_report'])
```

---

### Federated Learning Endpoints

#### POST `/federated/update`

Submit model weight update from a hospital client to the federated server.

**Request Body**:
```json
{
  "client_id": 1,
  "round_number": 5,
  "num_samples": 1000
}
```

**Request Schema**:
```typescript
{
  client_id: number (required),      // Unique hospital client identifier
  round_number: number (required),   // Federated learning round
  num_samples: number (required)     // Number of training samples used
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

**Response Schema**:
```typescript
{
  success: boolean,         // Whether update was accepted
  message: string,          // Status message
  current_round: number,    // Current federated round
  clients_received: number  // Number of clients submitted this round
}
```

**Status Codes**:
- `200 OK`: Update received successfully
- `400 Bad Request`: Invalid request body
- `500 Internal Server Error`: Failed to process update

**Notes**:
- In production, actual model weights would be transmitted via secure channel
- Weights are encrypted and differentially privatized before transmission

---

#### GET `/federated/status`

Get current status of the federated learning process.

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

**Response Schema**:
```typescript
{
  current_round: number,       // Current round number
  clients_registered: number,  // Clients submitted this round
  expected_clients: number,    // Expected total clients
  round_complete: boolean,     // Whether round is complete
  model_parameters: number     // Total model parameters
}
```

**Status Codes**:
- `200 OK`: Status retrieved successfully

---

#### POST `/federated/get_weights`

Retrieve current global model weights for a client.

**Query Parameters** (Form Data):
- `client_id` (integer, required): Client identifier

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
    },
    "genomics_encoder.embedding.weight": {
      "shape": [4, 256],
      "dtype": "torch.float32"
    }
  }
}
```

**Status Codes**:
- `200 OK`: Weights retrieved successfully
- `400 Bad Request`: Missing client_id parameter

**Example (curl)**:
```bash
curl -X POST "http://localhost:8000/federated/get_weights" \
  -F "client_id=1"
```

---

### Vector Search Endpoints

#### POST `/vector/search`

Search for similar patients based on multi-modal embeddings.

**Request Body**:
```json
{
  "patient_id": "PATIENT-001",
  "modality": "fusion",
  "limit": 10,
  "min_score": 0.7
}
```

**Request Schema**:
```typescript
{
  patient_id: string (required),  // Query patient identifier
  modality?: string,              // 'radiology', 'genomics', or 'fusion' (default: 'fusion')
  limit?: number,                 // Maximum results (1-100, default: 10)
  min_score?: number              // Minimum similarity score (0-1, default: 0.7)
}
```

**Response**:
```json
{
  "query_patient_id": "PATIENT-001",
  "similar_patients": [
    {
      "patient_id": "PATIENT-042",
      "score": 0.92,
      "metadata": {
        "diagnosis": "lung_adenocarcinoma",
        "stage": "IIA",
        "similarity_reason": "Similar EGFR mutation and imaging pattern"
      }
    },
    {
      "patient_id": "PATIENT-087",
      "score": 0.85,
      "metadata": {...}
    }
  ],
  "total_found": 2
}
```

**Status Codes**:
- `200 OK`: Search completed successfully
- `400 Bad Request`: Invalid search parameters
- `500 Internal Server Error`: Search failed

---

#### DELETE `/vector/patient/{patient_id}`

Delete all vector embeddings for a patient (privacy compliance).

**Path Parameters**:
- `patient_id` (string, required): Patient identifier

**Response**:
```json
{
  "patient_id": "PATIENT-001",
  "deleted": true,
  "message": "Patient data deleted from vector store"
}
```

**Status Codes**:
- `200 OK`: Deletion successful
- `404 Not Found`: Patient not found
- `500 Internal Server Error`: Deletion failed

**Use Cases**:
- GDPR right to erasure compliance
- HIPAA privacy rule compliance
- Patient consent withdrawal

---

#### GET `/vector/stats`

Get vector store statistics.

**Response**:
```json
{
  "collections": {
    "radiology_embeddings": {
      "count": 1500,
      "vectors_size_bytes": 4608000
    },
    "genomics_embeddings": {
      "count": 1500,
      "vectors_size_bytes": 4608000
    },
    "fusion_embeddings": {
      "count": 1500,
      "vectors_size_bytes": 4608000
    }
  },
  "total_patients": 1500
}
```

**Status Codes**:
- `200 OK`: Statistics retrieved successfully
- `500 Internal Server Error`: Vector store unavailable

---

### FHIR Integration Endpoints

#### GET `/fhir/patient/{patient_id}`

Retrieve patient data from configured FHIR server.

**Path Parameters**:
- `patient_id` (string, required): Patient identifier

**Response**:
```json
{
  "resourceType": "Patient",
  "id": "PATIENT-001",
  "identifier": [
    {
      "system": "http://hospital.example.org/mrn",
      "value": "MRN123456"
    }
  ],
  "name": [
    {
      "family": "Doe",
      "given": ["John"]
    }
  ],
  "note": "FHIR integration requires configured FHIR server"
}
```

**Status Codes**:
- `200 OK`: Patient data retrieved
- `404 Not Found`: Patient not found
- `500 Internal Server Error`: FHIR server unavailable

---

#### POST `/fhir/sync`

Sync data from FHIR server for a specific resource type.

**Form Data**:
- `resource_type` (string, required): FHIR resource type
- `patient_id` (string, required): Patient identifier

**Supported Resource Types**:
- `Patient`: Demographics
- `Observation`: Clinical measurements
- `DiagnosticReport`: Lab and imaging reports
- `ImagingStudy`: DICOM references
- `MolecularSequence`: Genomic data

**Response**:
```json
{
  "synced": true,
  "resource_type": "DiagnosticReport",
  "patient_id": "PATIENT-001",
  "resources_synced": 5,
  "note": "FHIR sync requires configured FHIR server"
}
```

**Status Codes**:
- `200 OK`: Sync completed successfully
- `400 Bad Request`: Invalid resource type
- `500 Internal Server Error`: Sync failed

---

## Error Handling

All endpoints return errors in a consistent format:

```json
{
  "detail": "Error description message"
}
```

### Common HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request succeeded |
| 400 | Bad Request | Invalid request parameters |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation error |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Error Response Examples

**Validation Error (422)**:
```json
{
  "detail": [
    {
      "loc": ["body", "patient_id"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Internal Error (500)**:
```json
{
  "detail": "Vector store connection failed: Connection refused"
}
```

---

## Rate Limiting

For production deployments, implement rate limiting to prevent abuse:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/diagnose")
@limiter.limit("10/minute")
async def run_diagnosis(request: DiagnosisRequest):
    ...
```

---

## CORS Configuration

The API supports Cross-Origin Resource Sharing (CORS):

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Configure allowed origins in `.env`:
```bash
CORS_ORIGINS=["https://your-frontend.com"]
```

---

## Versioning

API version is included in responses and can be checked via the root endpoint.

Current version: `1.0.0`

For breaking changes, use URL versioning:
```
/api/v1/diagnose
/api/v2/diagnose
```

---

## Client Libraries

### Python Client Example

```python
from typing import Dict, Any
import requests

class DiagnosticSwarmClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def diagnose(self, patient_id: str, radiology: str, 
                 genomics: str, history: str) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/diagnose",
            json={
                "patient_id": patient_id,
                "radiology_findings": radiology,
                "genomics_findings": genomics,
                "clinical_history": history
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_federated_status(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/federated/status")
        response.raise_for_status()
        return response.json()
    
    def submit_update(self, client_id: int, round_num: int, 
                     num_samples: int) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/federated/update",
            json={
                "client_id": client_id,
                "round_number": round_num,
                "num_samples": num_samples
            }
        )
        response.raise_for_status()
        return response.json()

# Usage
client = DiagnosticSwarmClient()
result = client.diagnose(
    patient_id="P001",
    radiology="2cm lung nodule",
    genomics="EGFR mutation",
    history="65yo smoker"
)
print(result['confidence_score'])
```

---

*Last Updated: January 2024*
*API Version: 1.0.0*

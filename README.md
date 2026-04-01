# Federated Multi-Modal Diagnostic Agentic Swarm

A privacy-preserving system where hospitals train a collective model on sensitive radiologic and genomic data without sharing raw data. An agentic swarm orchestrates diagnosis by consulting specialized "Expert Agents."

## 📚 Documentation

- **[Complete Documentation](docs/COMPLETE_DOCUMENTATION.md)** - Comprehensive guide covering architecture, installation, configuration, and usage
- **[API Reference](docs/API_REFERENCE.md)** - Detailed API endpoint documentation with examples
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Development setup, testing, debugging, and contribution guidelines

## Architecture

```
Regional Sensor/DICOM Ingest -> Local Feature Extraction -> Encrypted Gradient Sync -> 
Global Model Update -> Agentic Task Dispatch -> Cross-Modal Fusion -> Clinical Support
```

## Tech Stack

- **Python** - Core language
- **PySyft** - Federated learning and privacy-preserving computations
- **PyTorch** - Deep learning framework
- **LangGraph** - Agentic workflow orchestration
- **Qdrant** - Vector database for multi-modal embeddings
- **NVIDIA BioNeMo** - Biomedical foundation models
- **FastAPI** - REST API framework
- **FHIR** - Healthcare data interoperability standard

## Project Structure

```
/
├── federated_learning/     # Federated training, encrypted gradient sync
├── agentic_swarm/          # Expert agents, LangGraph workflows
├── data_ingestion/         # DICOM handling, FHIR integration
├── api/                    # FastAPI endpoints
├── vector_store/           # Qdrant integration
├── config/                 # Configuration files
├── utils/                  # Utility functions
├── models/                 # Model definitions
└── main.py                 # Entry point
```

## Features

1. **Privacy-Preserving Training**: Hospitals train models locally without sharing raw patient data
2. **Encrypted Gradient Synchronization**: Secure aggregation of model updates using PySyft
3. **Multi-Modal Fusion**: Combines radiologic (DICOM) and genomic data
4. **Agentic Diagnosis**: Specialized expert agents collaborate for comprehensive analysis
5. **FHIR Compliance**: Healthcare data standards for interoperability
6. **Vector Search**: Qdrant-powered semantic search across medical embeddings

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Start the federated learning server
python -m federated_learning.server

# Start a hospital client node
python -m federated_learning.client

# Launch the agentic diagnostic swarm
python -m agentic_swarm.orchestrator

# Run the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## License

MIT License

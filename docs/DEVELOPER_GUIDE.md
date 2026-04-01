# Developer Guide

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Code Structure](#code-structure)
3. [Adding New Features](#adding-new-features)
4. [Testing Guidelines](#testing-guidelines)
5. [Debugging Tips](#debugging-tips)
6. [Performance Optimization](#performance-optimization)
7. [Code Examples](#code-examples)

---

## Development Environment Setup

### Prerequisites

- Python 3.9+
- Git
- Docker (optional, for Qdrant)
- NVIDIA GPU with CUDA (recommended)

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/federated-diagnostic-swarm.git
cd federated-diagnostic-swarm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

### IDE Configuration

#### VS Code Settings

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.tabSize": 4
  }
}
```

#### PyCharm Configuration

1. Open Settings → Project → Python Interpreter
2. Add → Existing Environment → Select `venv/bin/python`
3. Enable auto-import and code formatting

---

## Code Structure

```
/workspace
├── agentic_swarm/           # Agent orchestration
│   ├── __init__.py
│   └── orchestrator.py      # Main orchestrator class
├── api/                     # FastAPI endpoints
│   ├── __init__.py
│   └── main.py             # API routes
├── config/                  # Configuration
│   ├── __init__.py
│   └── settings.py         # Pydantic settings
├── data_ingestion/          # Data loading
│   ├── __init__.py
│   └── dicom_loader.py     # DICOM handling
├── docs/                    # Documentation
│   ├── COMPLETE_DOCUMENTATION.md
│   ├── API_REFERENCE.md
│   └── DEVELOPER_GUIDE.md
├── federated_learning/      # FL components
│   ├── __init__.py
│   ├── client.py           # Hospital client
│   └── server.py           # Aggregation server
├── models/                  # Neural networks
│   ├── __init__.py
│   └── diagnostic_model.py # Multi-modal model
├── utils/                   # Utilities
│   ├── __init__.py
│   └── helpers.py          # Helper functions
├── vector_store/            # Vector database
│   ├── __init__.py
│   └── qdrant_manager.py   # Qdrant operations
├── tests/                   # Test suite
│   ├── test_federated.py
│   ├── test_agents.py
│   └── test_api.py
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
└── README.md               # Project overview
```

### Module Responsibilities

#### `agentic_swarm/`
- Implements expert agents (Radiology, Genomics, Oncology, Pathology)
- Manages LangGraph workflow state machine
- Generates diagnostic reports

#### `federated_learning/`
- Server: Aggregates model updates with differential privacy
- Client: Trains locally, computes gradients, applies privacy

#### `models/`
- RadiologyEncoder: 3D CNN for medical images
- GenomicsEncoder: Transformer for sequence data
- MultiModalFusion: Cross-modal attention mechanism

#### `api/`
- REST endpoints for diagnosis, FL coordination, vector search
- Request/response validation with Pydantic
- Error handling and logging

#### `vector_store/`
- Qdrant collection management
- Embedding storage and retrieval
- Similarity search operations

---

## Adding New Features

### Adding a New Expert Agent

1. Create new agent class in `agentic_swarm/orchestrator.py`:

```python
class CardiologyAgent(ExpertAgent):
    """Expert agent for cardiac analysis"""
    
    def __init__(self):
        super().__init__(
            name="Dr. Cardio",
            specialty="Cardiology",
            system_prompt="""You are an expert cardiologist..."""
        )
    
    async def analyze(self, state: DiagnosticState) -> str:
        # Access previous analyses
        radio = state['agent_outputs'].get('Radiology', '')
        
        # Implement cardiology-specific logic
        analysis = f"""
        CARDIOLOGY ANALYSIS by {self.name}:
        =====================================
        
        Cardiac Assessment:
        - Evaluated cardiac structures from imaging
        - Assessed function and morphology
        
        Integration with Other Findings:
        {radio if radio else 'No radiology findings provided'}
        
        Recommendations:
        - Consider echocardiography
        - Cardiac MRI if indicated
        """
        return analysis
```

2. Register agent in `DiagnosticOrchestrator.__init__()`:

```python
self.agents['Cardiology'] = CardiologyAgent()
```

3. Add workflow node in `_build_workflow()`:

```python
workflow.add_node("cardiology_analysis", self._run_cardiology_agent)
workflow.add_edge("oncology_synthesis", "cardiology_analysis")
```

4. Implement runner method:

```python
async def _run_cardiology_agent(self, state: DiagnosticState) -> Dict:
    agent = self.agents['Cardiology']
    output = await agent.analyze(state)
    state['agent_outputs']['Cardiology'] = output
    return {"agent_outputs": state['agent_outputs']}
```

### Adding a New API Endpoint

1. Define request/response models in `api/main.py`:

```python
class BatchDiagnosisRequest(BaseModel):
    patient_ids: List[str]
    include_images: bool = False

class BatchDiagnosisResponse(BaseModel):
    results: List[DiagnosisResponse]
    total_processed: int
    failed: List[str]
```

2. Implement endpoint:

```python
@app.post("/diagnose/batch", response_model=BatchDiagnosisResponse)
async def batch_diagnose(request: BatchDiagnosisRequest):
    """Run diagnosis on multiple patients"""
    orchestrator = get_diagnostic_orchestrator()
    results = []
    failed = []
    
    for pid in request.patient_ids:
        try:
            result = await orchestrator.run_diagnosis(...)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed for {pid}: {e}")
            failed.append(pid)
    
    return BatchDiagnosisResponse(
        results=results,
        total_processed=len(results),
        failed=failed
    )
```

3. Add to OpenAPI tags:

```python
@app.post("/diagnose/batch", ..., tags=["Diagnosis", "Batch Operations"])
```

### Adding Federated Learning Enhancements

1. Implement new aggregation strategy in `FederatedServer`:

```python
def aggregate_median(self, client_weights: Dict) -> Dict:
    """Median-based aggregation for robustness"""
    aggregated = {}
    param_names = list(next(iter(client_weights.values())).keys())
    
    for param_name in param_names:
        stacked = torch.stack([
            weights[param_name] 
            for weights in client_weights.values()
            if param_name in weights
        ])
        aggregated[param_name] = torch.median(stacked, dim=0).values
    
    return aggregated
```

2. Add configuration option in `config/settings.py`:

```python
AGGREGATION_METHOD: str = Field(
    default="weighted_average",
    description="Aggregation method: weighted_average, median, trimmed_mean"
)
```

---

## Testing Guidelines

### Test Structure

```python
# tests/test_federated.py
import pytest
import torch
from federated_learning.server import FederatedServer
from federated_learning.client import FederatedClient

class TestFederatedServer:
    
    @pytest.fixture
    def server(self):
        return FederatedServer()
    
    def test_aggregate_updates(self, server):
        # Arrange
        client_weights = {
            0: {"layer.weight": torch.ones(10)},
            1: {"layer.weight": torch.ones(10) * 2},
            2: {"layer.weight": torch.ones(10) * 3}
        }
        client_sizes = {0: 100, 1: 100, 2: 100}
        
        # Act
        aggregated = server.aggregate_updates(client_weights, client_sizes)
        
        # Assert
        assert "layer.weight" in aggregated
        assert torch.allclose(aggregated["layer.weight"], torch.ones(10) * 2)
    
    def test_differential_privacy(self, server):
        # Arrange
        updates = {"layer.weight": torch.ones(10)}
        
        # Act
        noised = server.apply_differential_privacy(updates, epsilon=1.0)
        
        # Assert
        assert "layer.weight" in noised
        assert not torch.equal(noised["layer.weight"], updates["layer.weight"])
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_federated.py -v

# With coverage
pytest tests/ -v --cov=. --cov-report=html

# Watch mode (requires pytest-watch)
ptw
```

### Mocking External Services

```python
from unittest.mock import Mock, patch
import pytest

@pytest.fixture
def mock_qdrant():
    with patch('vector_store.qdrant_manager.QdrantClient') as mock:
        yield mock

def test_vector_search(mock_qdrant):
    # Configure mock
    mock_instance = mock_qdrant.return_value
    mock_instance.search.return_value = [...]
    
    # Test code
    store = VectorStoreManager()
    results = store.search_similar(...)
    
    # Verify
    mock_instance.search.assert_called_once()
```

---

## Debugging Tips

### Logging Configuration

```python
from loguru import logger
import sys

logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="DEBUG"
)
logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="DEBUG")
```

### Debug Mode

Enable debug mode in `.env`:

```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

### Inspecting Model Gradients

```python
def inspect_gradients(model):
    """Print gradient statistics for debugging"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            print(f"{name}:")
            print(f"  Mean: {grad.mean().item():.6f}")
            print(f"  Std: {grad.std().item():.6f}")
            print(f"  Max: {grad.max().item():.6f}")
            print(f"  Min: {grad.min().item():.6f}")
            print(f"  NaN: {torch.isnan(grad).any().item()}")
            print(f"  Inf: {torch.isinf(grad).any().item()}")
```

### Profiling Performance

```python
import cProfile
import pstats
from io import StringIO

def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    
    stats_stream = StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    print(stats_stream.getvalue())
    return result

# Usage
profile_function(orchestrator.run_diagnosis, ...)
```

---

## Performance Optimization

### GPU Memory Management

```python
import torch
from contextlib import contextmanager

@contextmanager
def gpu_memory_cleanup():
    """Context manager for GPU memory cleanup"""
    try:
        yield
    finally:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Usage
with gpu_memory_cleanup():
    result = model.forward(data)
```

### Batch Processing

```python
def process_in_batches(items, batch_size, process_fn):
    """Process items in batches to manage memory"""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_fn(batch)
        results.extend(batch_results)
        torch.cuda.empty_cache()
    return results
```

### Async Operations

```python
import asyncio
from asyncio import Semaphore

class RateLimitedProcessor:
    def __init__(self, max_concurrent=5):
        self.semaphore = Semaphore(max_concurrent)
    
    async def process_with_limit(self, item):
        async with self.semaphore:
            return await self.process(item)
    
    async def process_batch(self, items):
        tasks = [self.process_with_limit(item) for item in items]
        return await asyncio.gather(*tasks)
```

---

## Code Examples

### Complete Federated Training Loop

```python
import asyncio
from federated_learning.server import FederatedServer
from federated_learning.client import FederatedClient
from torch.utils.data import DataLoader

async def run_federated_training(num_rounds=10, num_clients=5):
    server = FederatedServer()
    clients = [FederatedClient(i) for i in range(num_clients)]
    
    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1}/{num_rounds} ===")
        
        # Distribute global weights
        global_weights = server.get_model_state()
        for client in clients:
            client.set_global_weights(global_weights)
        
        # Local training
        tasks = []
        for client in clients:
            task = client.train_local(dataloader, epochs=5)
            tasks.append(task)
        
        updates = await asyncio.gather(*tasks)
        
        # Submit updates
        for i, (client, update) in enumerate(zip(clients, updates)):
            dp_update = client.apply_differential_privacy(update)
            server.receive_client_update(i, dp_update, num_samples=1000)
        
        # Aggregate
        if server.check_round_complete():
            aggregated, success = server.finalize_round()
            if success:
                print(f"✓ Round {round_num + 1} completed")
    
    return server
```

### Custom Loss Function

```python
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Usage in client
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

### Multi-GPU Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_ddp(model, dataloader, rank):
    setup_ddp(rank, torch.cuda.device_count())
    
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    for batch in dataloader:
        outputs = ddp_model(batch)
        # ... training loop
    
    dist.destroy_process_group()
```

---

## Best Practices

### Code Quality

1. **Type Hints**: Use type annotations for all functions
2. **Docstrings**: Follow Google style for docstrings
3. **Error Handling**: Use specific exceptions, log errors appropriately
4. **Logging**: Use structured logging with appropriate levels

### Security

1. **Input Validation**: Validate all user inputs
2. **Secrets Management**: Never commit secrets, use environment variables
3. **Dependency Updates**: Regularly update dependencies
4. **Access Control**: Implement proper authentication for production

### Performance

1. **Profiling**: Profile before optimizing
2. **Caching**: Cache expensive computations
3. **Batching**: Process data in batches
4. **Async I/O**: Use async for I/O-bound operations

---

*Last Updated: January 2024*
*Version: 1.0.0*

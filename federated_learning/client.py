"""
Federated Learning Client for Hospital Nodes
Trains local model on sensitive data and sends encrypted updates
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Tuple
import numpy as np
from loguru import logger

from models.diagnostic_model import FederatedDiagnosticModel
from config.settings import config


class HospitalDataset(Dataset):
    """
    Placeholder for hospital-specific dataset
    In production, this would load DICOM images and genomic data
    """
    
    def __init__(self, data_path: str, modality: str = 'multi_modal'):
        self.data_path = data_path
        self.modality = modality
        # Placeholder - in production, load actual patient data
        self.samples = []
        self.labels = []
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            radiology_data, genomics_data, label
        """
        return self.samples[idx]


class FederatedClient:
    """
    Hospital client for federated learning
    Trains locally without sharing raw patient data
    """
    
    def __init__(
        self,
        client_id: int,
        global_model: Optional[FederatedDiagnosticModel] = None,
        device: str = 'cpu'
    ):
        self.client_id = client_id
        self.device = torch.device(device)
        self.model = global_model or FederatedDiagnosticModel().to(self.device)
        self.initial_weights: Optional[Dict[str, torch.Tensor]] = None
        
        logger.info(f"FederatedClient {client_id} initialized on {device}")
    
    def set_global_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Receive and set global model weights from server"""
        self.model.load_state_dict(weights)
        self.initial_weights = {k: v.clone() for k, v in weights.items()}
        logger.info(f"Client {self.client_id} received global model weights")
    
    def train_local(
        self,
        dataloader: DataLoader,
        epochs: int = 5,
        learning_rate: float = 0.001
    ) -> Dict[str, torch.Tensor]:
        """
        Train model on local hospital data
        
        Args:
            dataloader: Local data loader with patient data
            epochs: Number of local training epochs
            learning_rate: Learning rate for training
            
        Returns:
            Weight updates (delta from initial weights)
        """
        if self.initial_weights is None:
            raise ValueError("Must receive global weights before local training")
        
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch in dataloader:
                radiology_data, genomics_data, labels = batch
                
                # Move to device
                radiology_data = radiology_data.to(self.device)
                genomics_data = genomics_data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                radiology_emb, genomics_emb, fused, logits = self.model(
                    radiology_data, genomics_data
                )
                
                # Compute loss
                loss = criterion(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
            
            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            logger.info(f"Client {self.client_id} - Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss
            num_batches += 1
        
        # Calculate weight updates (gradients)
        current_weights = self.model.state_dict()
        weight_updates = {}
        
        for param_name in self.initial_weights.keys():
            if param_name in current_weights:
                # Compute delta (update direction)
                weight_updates[param_name] = current_weights[param_name] - self.initial_weights[param_name]
        
        avg_loss = total_loss / max(num_batches, 1)
        logger.success(f"Client {self.client_id} completed local training, avg loss: {avg_loss:.4f}")
        
        return weight_updates
    
    def get_weight_updates(self) -> Dict[str, torch.Tensor]:
        """
        Get the difference between current and initial weights
        This is what gets sent to the federated server
        """
        if self.initial_weights is None:
            raise ValueError("No initial weights set")
        
        current_weights = self.model.state_dict()
        updates = {}
        
        for param_name in self.initial_weights.keys():
            if param_name in current_weights:
                updates[param_name] = current_weights[param_name] - self.initial_weights[param_name]
        
        return updates
    
    def apply_differential_privacy(
        self,
        updates: Dict[str, torch.Tensor],
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Apply local differential privacy before sending updates
        
        Args:
            updates: Weight updates
            epsilon: Privacy budget
            delta: Privacy failure probability
            clip_norm: Gradient clipping norm
            
        Returns:
            Noised and clipped updates
        """
        sigma = clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        noised_updates = {}
        for param_name, update in updates.items():
            # Clip gradients
            update_norm = torch.norm(update)
            if update_norm > clip_norm:
                update = update * (clip_norm / update_norm)
            
            # Add Gaussian noise
            noise = torch.randn_like(update) * sigma
            noised_updates[param_name] = update + noise
        
        logger.info(f"Applied local DP with epsilon={epsilon}")
        return noised_updates
    
    def get_num_samples(self) -> int:
        """Return number of samples in local dataset"""
        # In production, return actual dataset size
        return 0


async def simulate_client_training(
    client_id: int,
    num_epochs: int = 5,
    num_samples: int = 100
):
    """
    Simulate a client training round
    
    Args:
        client_id: Client identifier
        num_epochs: Number of local epochs
        num_samples: Number of synthetic samples for simulation
    """
    from federated_learning.server import FederatedServer
    
    # Initialize server and client
    server = FederatedServer()
    client = FederatedClient(client_id=client_id, global_model=server.model)
    
    # Receive global weights
    global_weights = server.get_model_state()
    client.set_global_weights(global_weights)
    
    # Create synthetic data for simulation
    batch_size = config.federated.batch_size
    radiology_shape = (batch_size, 1, 32, 32, 32)  # Small 3D volumes
    genomics_shape = (batch_size, 100)  # Short sequences
    num_classes = 10
    
    # Synthetic dataset
    class SyntheticDataset(Dataset):
        def __init__(self, num_samples: int):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            radiology = torch.randn(1, 32, 32, 32)
            genomics = torch.randint(0, 4, (100,))
            label = torch.randint(0, num_classes, (1,)).squeeze()
            return radiology, genomics, label
    
    dataset = SyntheticDataset(num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train locally
    updates = client.train_local(dataloader, epochs=num_epochs)
    
    # Apply differential privacy
    dp_updates = client.apply_differential_privacy(
        updates,
        epsilon=config.federated.differential_privacy_epsilon,
        delta=config.federated.differential_privacy_delta
    )
    
    # Send to server
    server.receive_client_update(client_id, dp_updates, num_samples)
    
    logger.info(f"Client {client_id} sent updates to server")
    return updates


if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Simulate multiple clients
        clients = [0, 1, 2, 3, 4]
        tasks = [simulate_client_training(cid) for cid in clients]
        
        await asyncio.gather(*tasks)
        print("All clients completed training")
    
    asyncio.run(main())

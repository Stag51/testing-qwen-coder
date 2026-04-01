"""
Federated Learning Server for Multi-Modal Diagnostic Model
Coordinates secure aggregation of model updates from hospital clients
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import asyncio
from loguru import logger

from models.diagnostic_model import FederatedDiagnosticModel
from config.settings import config


class FederatedServer:
    """
    Central server for federated learning coordination
    Implements secure aggregation with differential privacy
    """
    
    def __init__(self, model: Optional[FederatedDiagnosticModel] = None):
        self.model = model or FederatedDiagnosticModel()
        self.client_updates: Dict[int, Dict[str, torch.Tensor]] = {}
        self.current_round = 0
        self.num_clients = config.federated.num_clients
        
        logger.info("FederatedServer initialized with model")
        
    def aggregate_updates(
        self,
        client_weights: Dict[int, Dict[str, torch.Tensor]],
        client_sizes: Dict[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model updates from multiple clients using weighted averaging
        
        Args:
            client_weights: Dictionary mapping client_id to their model weight updates
            client_sizes: Dictionary mapping client_id to their local dataset sizes
            
        Returns:
            Aggregated weight updates
        """
        total_samples = sum(client_sizes.values())
        aggregated = {}
        
        # Get all parameter names from first client
        param_names = list(next(iter(client_weights.values())).keys())
        
        for param_name in param_names:
            weighted_sum = None
            
            for client_id, weights in client_weights.items():
                if param_name in weights:
                    weight = client_sizes[client_id] / total_samples
                    if weighted_sum is None:
                        weighted_sum = weight * weights[param_name]
                    else:
                        weighted_sum += weight * weights[param_name]
            
            if weighted_sum is not None:
                aggregated[param_name] = weighted_sum
        
        return aggregated
    
    def apply_differential_privacy(
        self,
        updates: Dict[str, torch.Tensor],
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Apply differential privacy noise to aggregated updates
        
        Args:
            updates: Aggregated model updates
            epsilon: Privacy budget
            delta: Privacy failure probability
            sensitivity: Sensitivity of the query
            
        Returns:
            Noised updates
        """
        # Calculate Gaussian mechanism noise scale
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        noised_updates = {}
        for param_name, update in updates.items():
            noise = torch.randn_like(update) * sigma
            noised_updates[param_name] = update + noise
            
        logger.info(f"Applied differential privacy with epsilon={epsilon}, sigma={sigma:.4f}")
        return noised_updates
    
    def update_global_model(self, aggregated_updates: Dict[str, torch.Tensor]) -> None:
        """
        Update the global model with aggregated updates
        
        Args:
            aggregated_updates: Aggregated and possibly noised weight updates
        """
        with torch.no_grad():
            for param_name, param in self.model.named_parameters():
                if param_name in aggregated_updates:
                    param.add_(aggregated_updates[param_name])
                    
        logger.info(f"Global model updated with {len(aggregated_updates)} parameters")
    
    def receive_client_update(
        self,
        client_id: int,
        weights: Dict[str, torch.Tensor],
        num_samples: int
    ) -> bool:
        """
        Receive and store update from a client
        
        Args:
            client_id: Unique identifier for the client
            weights: Model weight updates from the client
            num_samples: Number of samples used for training
            
        Returns:
            True if update was successfully received
        """
        self.client_updates[client_id] = {
            'weights': weights,
            'num_samples': num_samples,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        logger.info(f"Received update from client {client_id} ({num_samples} samples)")
        return True
    
    def check_round_complete(self) -> bool:
        """Check if all expected clients have submitted updates"""
        return len(self.client_updates) >= self.num_clients
    
    def finalize_round(
        self,
        apply_dp: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], bool]:
        """
        Finalize the current federated round
        
        Args:
            apply_dp: Whether to apply differential privacy
            
        Returns:
            Tuple of (aggregated updates, success flag)
        """
        if not self.check_round_complete():
            logger.warning(f"Round {self.current_round} incomplete: {len(self.client_updates)}/{self.num_clients} clients")
            return {}, False
        
        # Extract weights and sample counts
        client_weights = {
            cid: data['weights'] for cid, data in self.client_updates.items()
        }
        client_sizes = {
            cid: data['num_samples'] for cid, data in self.client_updates.items()
        }
        
        # Aggregate
        aggregated = self.aggregate_updates(client_weights, client_sizes)
        
        # Apply differential privacy
        if apply_dp:
            aggregated = self.apply_differential_privacy(
                aggregated,
                epsilon=config.federated.differential_privacy_epsilon,
                delta=config.federated.differential_privacy_delta
            )
        
        # Update global model
        self.update_global_model(aggregated)
        
        # Reset for next round
        self.client_updates.clear()
        self.current_round += 1
        
        logger.success(f"Round {self.current_round - 1} completed successfully")
        return aggregated, True
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current global model state dict"""
        return self.model.state_dict()
    
    def load_model_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load model state dict"""
        self.model.load_state_dict(state_dict)
        logger.info("Global model state loaded")


async def run_federated_training(num_rounds: int = 10):
    """
    Example federated training loop
    
    Args:
        num_rounds: Number of federated rounds to run
    """
    server = FederatedServer()
    
    logger.info(f"Starting federated training for {num_rounds} rounds")
    
    for round_num in range(num_rounds):
        logger.info(f"=== Round {round_num + 1}/{num_rounds} ===")
        
        # Simulate waiting for client updates
        # In production, this would be handled via gRPC/REST API
        await asyncio.sleep(1)  # Placeholder for actual client communication
        
        # Finalize round
        updates, success = server.finalize_round(apply_dp=True)
        
        if success:
            logger.info(f"Round {round_num + 1} aggregated {len(updates)} parameters")
        else:
            logger.warning(f"Round {round_num + 1} failed to complete")
    
    logger.success("Federated training completed")
    return server


if __name__ == "__main__":
    asyncio.run(run_federated_training(num_rounds=5))

"""
Main entry point for Federated Multi-Modal Diagnostic Agentic Swarm

This module provides the main application entry points for:
1. Running the API server
2. Running federated learning server
3. Running diagnostic swarm demonstrations
"""
import asyncio
import argparse
from loguru import logger
from config.settings import config


def run_api_server(host: str = None, port: int = None, reload: bool = False):
    """Run the FastAPI server"""
    import uvicorn
    
    host = host or config.api.host
    port = port or config.api.port
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload
    )


async def run_federated_training_demo(num_rounds: int = 5):
    """Run a demonstration of federated training"""
    from federated_learning.server import run_federated_training
    
    logger.info(f"Running federated training demo with {num_rounds} rounds")
    server = await run_federated_training(num_rounds=num_rounds)
    return server


async def run_diagnostic_swarm_demo():
    """Run a demonstration of the diagnostic agentic swarm"""
    from agentic_swarm.orchestrator import demo_diagnostic_swarm
    
    logger.info("Running diagnostic swarm demo")
    result = await demo_diagnostic_swarm()
    return result


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Federated Multi-Modal Diagnostic Agentic Swarm"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # API Server command
    api_parser = subparsers.add_parser("api", help="Run the API server")
    api_parser.add_argument("--host", type=str, default=None, help="API host")
    api_parser.add_argument("--port", type=int, default=None, help="API port")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Federated training command
    fl_parser = subparsers.add_parser("federated", help="Run federated learning")
    fl_parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    
    # Diagnostic demo command
    diag_parser = subparsers.add_parser("diagnose", help="Run diagnostic demo")
    
    args = parser.parse_args()
    
    if args.command == "api":
        run_api_server(host=args.host, port=args.port, reload=args.reload)
    
    elif args.command == "federated":
        asyncio.run(run_federated_training_demo(num_rounds=args.rounds))
    
    elif args.command == "diagnose":
        asyncio.run(run_diagnostic_swarm_demo())
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

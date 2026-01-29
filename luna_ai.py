"""
LunaAI Main Application
Entry point for training, chat GUI, and MCP server.
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(args):
    """Train the LunaAI model."""
    from model_trainer import LunaAITrainer
    
    logger.info("Starting model training...")
    
    trainer = LunaAITrainer(
        model_name=args.base_model,
        output_dir=args.output_dir,
        max_length=args.max_length,
    )
    
    trainer.train_model(
        num_samples=args.num_samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    logger.info(f"Training completed! Model saved to {args.output_dir}")


def launch_chat_gui(args):
    """Launch the chat GUI."""
    from chat_gui import LunaAIChatGUI
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found at {args.model_path}")
        logger.info("Please train the model first using: python luna_ai.py train")
        sys.exit(1)
    
    logger.info("Launching chat GUI...")
    app = LunaAIChatGUI(model_path=args.model_path)
    app.run()


def start_mcp_server(args):
    """Start the MCP server."""
    import asyncio
    from mcp_server import LunaAIMCPServer
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found at {args.model_path}")
        logger.info("Please train the model first using: python luna_ai.py train")
        sys.exit(1)
    
    logger.info("Starting MCP server...")
    
    server = LunaAIMCPServer(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
    )
    
    asyncio.run(server.start_server())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LunaAI - AI Chat Application with Model Training and MCP Server Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python luna_ai.py train --num-samples 5000 --epochs 2
  
  # Launch chat GUI
  python luna_ai.py chat
  
  # Start MCP server
  python luna_ai.py mcp --port 8765
  
  # Train with custom base model
  python luna_ai.py train --base-model microsoft/DialoGPT-medium
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the AI model')
    train_parser.add_argument(
        '--base-model',
        type=str,
        default='microsoft/DialoGPT-small',
        help='Base model to fine-tune (default: microsoft/DialoGPT-small)'
    )
    train_parser.add_argument(
        '--output-dir',
        type=str,
        default='./luna_model',
        help='Output directory for trained model (default: ./luna_model)'
    )
    train_parser.add_argument(
        '--num-samples',
        type=int,
        default=5000,
        help='Number of training samples (default: 5000)'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='Number of training epochs (default: 2)'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size (default: 4)'
    )
    train_parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Launch the chat GUI')
    chat_parser.add_argument(
        '--model-path',
        type=str,
        default='./luna_model',
        help='Path to trained model (default: ./luna_model)'
    )
    
    # MCP server command
    mcp_parser = subparsers.add_parser('mcp', help='Start the MCP server')
    mcp_parser.add_argument(
        '--model-path',
        type=str,
        default='./luna_model',
        help='Path to trained model (default: ./luna_model)'
    )
    mcp_parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Server host (default: 127.0.0.1)'
    )
    mcp_parser.add_argument(
        '--port',
        type=int,
        default=8765,
        help='Server port (default: 8765)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'train':
        train_model(args)
    elif args.command == 'chat':
        launch_chat_gui(args)
    elif args.command == 'mcp':
        start_mcp_server(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Command-line interface for Agentic Scent Analytics.
"""

import argparse
import asyncio
import sys
import logging
from pathlib import Path

from . import ScentAnalyticsFactory, QualityControlAgent, AgentOrchestrator


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def run_demo(args):
    """Run a demonstration of the system."""
    print("🔬 Agentic Scent Analytics Demo")
    print("=" * 40)
    
    # Initialize system
    factory = ScentAnalyticsFactory(
        production_line=args.line,
        e_nose_config={
            'sensors': ['MOS', 'PID', 'EC'],
            'sampling_rate': args.rate,
            'channels': args.channels
        }
    )
    
    qc_agent = QualityControlAgent()
    factory.register_agent(qc_agent)
    
    # Start monitoring
    await qc_agent.start()
    
    print(f"🏭 Monitoring {args.line} for {args.duration} seconds...")
    
    reading_count = 0
    start_time = asyncio.get_event_loop().time()
    
    async for reading in factory.sensor_stream():
        current_time = asyncio.get_event_loop().time()
        if current_time - start_time > args.duration:
            break
            
        reading_count += 1
        analysis = await qc_agent.analyze(reading)
        
        if analysis and analysis.anomaly_detected:
            print(f"⚠️  Anomaly detected at reading {reading_count} (confidence: {analysis.confidence:.3f})")
    
    await factory.stop_monitoring()
    await qc_agent.stop()
    
    print(f"🏁 Demo completed. Processed {reading_count} readings.")


def run_example(args):
    """Run an example script."""
    example_path = Path(__file__).parent.parent / "examples" / f"{args.example}.py"
    
    if not example_path.exists():
        print(f"❌ Example '{args.example}' not found.")
        print("Available examples:")
        examples_dir = Path(__file__).parent.parent / "examples"
        if examples_dir.exists():
            for example_file in examples_dir.glob("*.py"):
                if example_file.name != "__init__.py":
                    print(f"  - {example_file.stem}")
        return 1
    
    print(f"🚀 Running example: {args.example}")
    print("-" * 40)
    
    # Execute the example
    import subprocess
    result = subprocess.run([sys.executable, str(example_path)], 
                          capture_output=False)
    
    return result.returncode


def show_status(args):
    """Show system status."""
    print("📊 Agentic Scent Analytics Status")
    print("=" * 40)
    print("🟢 System: Online")
    print("📦 Version: 0.1.0")
    print("🔧 Available sensors: Mock E-nose, Temperature, Humidity")
    print("🤖 Available agents: Quality Control, Mock LLM")
    print("📈 Analytics: Fingerprinting, Predictive Quality")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic Scent Analytics - Industrial AI Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  agentic-scent demo --line pharma_coating --duration 30
  agentic-scent example basic_usage
  agentic-scent example multi_agent_demo
  agentic-scent status
        """
    )
    
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set logging level")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run system demonstration")
    demo_parser.add_argument("--line", default="demo_line",
                            help="Production line name (default: demo_line)")
    demo_parser.add_argument("--duration", type=int, default=30,
                            help="Demo duration in seconds (default: 30)")
    demo_parser.add_argument("--rate", type=float, default=1.0,
                            help="Sampling rate in Hz (default: 1.0)")
    demo_parser.add_argument("--channels", type=int, default=32,
                            help="Number of sensor channels (default: 32)")
    
    # Example command
    example_parser = subparsers.add_parser("example", help="Run example scripts")
    example_parser.add_argument("example", 
                               choices=["basic_usage", "multi_agent_demo", "fingerprinting_demo"],
                               help="Example to run")
    
    # Status command
    subparsers.add_parser("status", help="Show system status")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Handle commands
    if args.command == "demo":
        return asyncio.run(run_demo(args))
    elif args.command == "example":
        return run_example(args)
    elif args.command == "status":
        show_status(args)
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
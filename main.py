"""
Main Entry Point
Can be used to run the system or evaluation.

Usage:
  python main.py --mode cli           # Run CLI interface
  python main.py --mode web           # Run web interface
  python main.py --mode evaluate      # Run evaluation
"""

import argparse
import asyncio
import sys
from pathlib import Path


def run_cli():
    """Run CLI interface."""
    from src.ui.cli import main as cli_main
    cli_main()


def run_web():
    """Run web interface."""
    import subprocess
    print("Starting Streamlit web interface...")
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"])


async def run_evaluation():
    """Run system evaluation."""
    import yaml
    from dotenv import load_dotenv
    from src.orchestrator import Orchestrator
    from src.evaluation.evaluator import SystemEvaluator
    
    # Load environment variables
    load_dotenv()

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize sequential orchestrator
    print("Initializing sequential orchestrator...")
    orchestrator = Orchestrator(config)
    
    # Initialize evaluator
    print("Initializing system evaluator...")
    evaluator = SystemEvaluator(config, orchestrator=orchestrator)
    
    # Print workflow visualization
    print("\n" + "=" * 70)
    print("SYSTEM WORKFLOW")
    print("=" * 70)
    print(orchestrator.visualize_workflow())
    
    # Run full evaluation
    print("\n" + "=" * 70)
    print("RUNNING FULL EVALUATION")
    print("=" * 70)
    print("\nThis will evaluate the system on test queries from data/example_queries.json")
    print("Evaluation may take several minutes depending on the number of queries...\n")
    
    # Run evaluation
    report = await evaluator.evaluate_system("data/example_queries.json")
    
    # Display results summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    
    summary = report.get("summary", {})
    print(f"\nTotal Queries: {summary.get('total_queries', 0)}")
    print(f"Successful: {summary.get('successful', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print(f"Success Rate: {summary.get('success_rate', 0.0):.2%}")
    
    scores = report.get("scores", {})
    print(f"\nOverall Average Score: {scores.get('overall_average', 0.0):.3f}")
    print(f"Score Range: {scores.get('overall_min', 0.0):.3f} - {scores.get('overall_max', 0.0):.3f}")
    print(f"Standard Deviation: {scores.get('overall_std', 0.0):.3f}")
    
    print("\nScores by Criterion:")
    for criterion, score in scores.get("by_criterion", {}).items():
        print(f"  {criterion}: {score:.3f}")
    
    print("\nScore Distribution:")
    distribution = scores.get("distribution", {})
    for range_name, count in distribution.items():
        print(f"  {range_name}: {count}")
    
    print(f"\nDetailed results saved to outputs/")
    print("=" * 70)


def run_autogen():
    """Run AutoGen example."""
    import subprocess
    print("Running AutoGen example...")
    subprocess.run([sys.executable, "example_autogen.py"])


def run_sequential():
    """Run sequential orchestrator test."""
    import yaml
    from dotenv import load_dotenv
    from src.orchestrator import Orchestrator
    
    # Load environment variables
    load_dotenv()

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize sequential orchestrator
    print("Initializing sequential orchestrator...")
    orchestrator = Orchestrator(config)
    
    # Print workflow visualization
    print(orchestrator.visualize_workflow())
    
    # Example query
    query = "What are the latest trends in human-computer interaction research?"
    
    print(f"\nProcessing query: {query}\n")
    print("=" * 70)
    
    # Process query
    result = orchestrator.process_query(query)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nMetadata:")
    print(f"  - Iterations: {result['metadata']['iterations']}")
    print(f"  - Sources: {result['metadata']['num_sources']}")
    print(f"  - Citations: {len(result['metadata'].get('citations', []))}")
    print(f"  - Elapsed time: {result['metadata']['elapsed_time']:.2f}s")
    
    # Print workflow trace
    print("\n" + "=" * 70)
    print("WORKFLOW TRACE")
    print("=" * 70)
    print(orchestrator.get_workflow_trace_summary())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "web", "evaluate", "autogen", "sequential"],
        default="autogen",
        help="Mode to run: cli, web, evaluate, autogen, or sequential"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    elif args.mode == "web":
        run_web()
    elif args.mode == "evaluate":
        asyncio.run(run_evaluation())
    elif args.mode == "autogen":
        run_autogen()
    elif args.mode == "sequential":
        run_sequential()


if __name__ == "__main__":
    main()

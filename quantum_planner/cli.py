"""Command-line interface for quantum task planner."""

import asyncio
import click
import json
import sys
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

from .core.planner import QuantumTaskPlanner
from .core.task import Task, TaskPriority, TaskStatus
from .core.config import PlannerConfig
from .agents.task_agent import TaskAgent


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.pass_context
def cli(ctx, config, debug):
    """Quantum Task Planner - Advanced task optimization using quantum computing principles."""
    ctx.ensure_object(dict)
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    planner_config = PlannerConfig.create_default()
    if config:
        planner_config.load_from_file(config)
    
    ctx.obj['config'] = planner_config


@cli.command()
@click.option('--tasks', type=int, default=5, help='Number of sample tasks to create')
@click.option('--agents', type=int, default=3, help='Number of agents to create')
@click.option('--duration', type=int, default=30, help='Simulation duration in seconds')
@click.pass_context
def demo(ctx, tasks, agents, duration):
    """Run a demonstration of the quantum task planner."""
    click.echo(f"🚀 Starting Quantum Task Planner Demo")
    click.echo(f"   Tasks: {tasks}, Agents: {agents}, Duration: {duration}s")
    
    async def run_demo():
        config = ctx.obj['config']
        planner = QuantumTaskPlanner(config)
        
        # Create sample tasks
        click.echo("\n📋 Creating sample tasks...")
        sample_tasks = _create_sample_tasks(tasks)
        
        for task in sample_tasks:
            await planner.add_task(task)
            click.echo(f"   ✓ Added task: {task.name}")
        
        # Create and register agents
        click.echo(f"\n🤖 Creating {agents} task agents...")
        for i in range(agents):
            agent = TaskAgent(config=config)
            await planner.coordinator.register_agent(agent)
            click.echo(f"   ✓ Registered agent: {agent.agent_id[:8]}")
        
        # Optimize schedule
        click.echo("\n⚡ Optimizing task schedule with quantum algorithms...")
        schedule = await planner.optimize_schedule()
        
        if schedule:
            click.echo(f"   ✓ Generated schedule for {len(schedule)} tasks")
            _display_schedule(schedule, sample_tasks)
        else:
            click.echo("   ⚠️  No schedule generated")
            return
        
        # Execute tasks
        click.echo(f"\n🎯 Executing tasks for {duration} seconds...")
        start_time = datetime.now()
        
        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                planner.execute_schedule(schedule),
                timeout=duration
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Display results
            _display_results(results, execution_time)
            
        except asyncio.TimeoutError:
            click.echo(f"   ⏰ Demo completed after {duration}s timeout")
            
            # Get current status
            status = await planner.get_status()
            click.echo(f"\n📊 Final Status:")
            click.echo(f"   Completed: {status['completed']}")
            click.echo(f"   Running: {status['running']}")
            click.echo(f"   Pending: {status['pending']}")
    
    try:
        asyncio.run(run_demo())
        click.echo("\n✅ Demo completed successfully!")
    except Exception as e:
        click.echo(f"\n❌ Demo failed: {e}")
        if ctx.obj['config'].debug_mode:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('task_file', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), help='Output file for results')
@click.option('--agents', type=int, default=3, help='Number of agents to use')
@click.pass_context
def execute(ctx, task_file, output, agents):
    """Execute tasks from a JSON file."""
    click.echo(f"📁 Loading tasks from {task_file}")
    
    async def run_execution():
        config = ctx.obj['config']
        planner = QuantumTaskPlanner(config)
        
        # Load tasks from file
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        tasks = []
        for task_dict in task_data.get('tasks', []):
            task = Task.from_dict(task_dict)
            await planner.add_task(task)
            tasks.append(task)
        
        click.echo(f"   ✓ Loaded {len(tasks)} tasks")
        
        # Create agents
        for i in range(agents):
            agent = TaskAgent(config=config)
            await planner.coordinator.register_agent(agent)
        
        click.echo(f"   ✓ Created {agents} agents")
        
        # Execute
        click.echo("⚡ Optimizing and executing tasks...")
        results = await planner.execute_schedule()
        
        # Save results
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            click.echo(f"   ✓ Results saved to {output}")
        
        _display_results(results)
    
    try:
        asyncio.run(run_execution())
        click.echo("✅ Execution completed!")
    except Exception as e:
        click.echo(f"❌ Execution failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--name', required=True, help='Task name')
@click.option('--description', default='', help='Task description')
@click.option('--priority', type=click.Choice(['1', '2', '3', '4']), default='2', help='Task priority (1=low, 4=critical)')
@click.option('--duration', type=int, default=60, help='Estimated duration in minutes')
@click.option('--depends-on', multiple=True, help='Task dependencies (task IDs)')
@click.pass_context
def create_task(ctx, name, description, priority, duration, depends_on):
    """Create a new task interactively."""
    task = Task(
        name=name,
        description=description,
        priority=TaskPriority(int(priority)),
        estimated_duration=timedelta(minutes=duration)
    )
    
    # Add dependencies
    for dep_id in depends_on:
        task.add_dependency(dep_id)
    
    # Save task to file
    task_file = Path(f"task_{task.id[:8]}.json")
    with open(task_file, 'w') as f:
        json.dump(task.to_dict(), f, indent=2, default=str)
    
    click.echo(f"✅ Created task: {task.name}")
    click.echo(f"   ID: {task.id}")
    click.echo(f"   File: {task_file}")


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and configuration."""
    config = ctx.obj['config']
    
    click.echo("🔧 Quantum Task Planner Status")
    click.echo("=" * 40)
    
    click.echo(f"Configuration:")
    click.echo(f"  Max Iterations: {config.max_iterations}")
    click.echo(f"  Convergence Threshold: {config.convergence_threshold}")
    click.echo(f"  Quantum Annealing Strength: {config.quantum_annealing_strength}")
    click.echo(f"  Max Concurrent Tasks: {config.max_concurrent_tasks}")
    click.echo(f"  Storage Backend: {config.storage_backend}")
    click.echo(f"  Debug Mode: {config.debug_mode}")
    
    click.echo(f"\nQuantum Parameters:")
    quantum_params = config.get_quantum_params()
    for key, value in quantum_params.items():
        click.echo(f"  {key}: {value}")
    
    click.echo(f"\nScheduling Parameters:")
    sched_params = config.get_scheduling_params()
    for key, value in sched_params.items():
        click.echo(f"  {key}: {value}")


@cli.command()
@click.option('--output', type=click.Path(), default='config.json', help='Output configuration file')
@click.pass_context
def generate_config(ctx, output):
    """Generate a sample configuration file."""
    config = PlannerConfig.create_default()
    config.save_to_file(output)
    
    click.echo(f"✅ Generated configuration file: {output}")
    click.echo("Edit this file to customize your quantum planner settings.")


@cli.command()
@click.option('--format', type=click.Choice(['json', 'yaml']), default='json', help='Output format')
@click.option('--count', type=int, default=10, help='Number of sample tasks')
@click.option('--output', type=click.Path(), help='Output file')
def generate_tasks(format, count, output):
    """Generate sample tasks for testing."""
    tasks = _create_sample_tasks(count)
    
    task_data = {
        "tasks": [task.to_dict() for task in tasks],
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "count": len(tasks),
            "generator": "quantum-planner-cli"
        }
    }
    
    if output:
        with open(output, 'w') as f:
            json.dump(task_data, f, indent=2, default=str)
        click.echo(f"✅ Generated {count} sample tasks in {output}")
    else:
        click.echo(json.dumps(task_data, indent=2, default=str))


def _create_sample_tasks(count: int) -> list[Task]:
    """Create sample tasks for demonstration."""
    import random
    
    task_templates = [
        ("Data Processing", "Process and analyze incoming data stream"),
        ("Model Training", "Train machine learning model with new data"),
        ("Quality Check", "Validate output quality and correctness"),
        ("Report Generation", "Generate comprehensive analysis report"),
        ("System Backup", "Backup system state and configurations"),
        ("Performance Optimization", "Optimize system performance parameters"),
        ("Security Scan", "Perform security vulnerability assessment"),
        ("Database Cleanup", "Clean up old records and optimize database"),
        ("API Integration", "Integrate with external API services"),
        ("Monitoring Setup", "Configure monitoring and alerting systems")
    ]
    
    tasks = []
    
    for i in range(count):
        template = random.choice(task_templates)
        
        task = Task(
            name=f"{template[0]} #{i+1}",
            description=template[1],
            priority=TaskPriority(random.randint(1, 4)),
            estimated_duration=timedelta(minutes=random.randint(15, 180)),
            success_probability=random.uniform(0.8, 1.0),
            amplitude=random.uniform(0.5, 1.0),
            phase=random.uniform(0, 6.28)
        )
        
        # Add some resource requirements
        task.resources_required = {
            "cpu": random.uniform(0.1, 0.8),
            "memory": random.uniform(0.1, 0.6),
            "io": random.uniform(0.05, 0.3)
        }
        
        # Add random tags
        all_tags = ["urgent", "batch", "realtime", "ml", "data", "security", "optimization"]
        task.tags = set(random.sample(all_tags, random.randint(1, 3)))
        
        tasks.append(task)
    
    # Add some dependencies
    for i in range(1, min(count, 5)):
        tasks[i].add_dependency(tasks[i-1].id)
    
    return tasks


def _display_schedule(schedule: Dict[str, datetime], tasks: list[Task]):
    """Display the generated schedule."""
    click.echo("\n📅 Generated Schedule:")
    
    # Sort by scheduled time
    sorted_schedule = sorted(schedule.items(), key=lambda x: x[1])
    
    task_lookup = {task.id: task for task in tasks}
    
    for task_id, scheduled_time in sorted_schedule[:10]:  # Show first 10
        task = task_lookup.get(task_id)
        if task:
            time_str = scheduled_time.strftime("%H:%M:%S")
            click.echo(f"   {time_str} - {task.name} (Priority: {task.priority.value})")
    
    if len(sorted_schedule) > 10:
        click.echo(f"   ... and {len(sorted_schedule) - 10} more tasks")


def _display_results(results: Dict[str, Any], execution_time: Optional[float] = None):
    """Display execution results."""
    click.echo("\n📊 Execution Results:")
    
    if execution_time:
        click.echo(f"   Execution Time: {execution_time:.2f}s")
    
    status = results.get('status', 'unknown')
    completed = results.get('completed', 0)
    failed = results.get('failed', 0)
    
    click.echo(f"   Status: {status}")
    click.echo(f"   Completed: {completed}")
    click.echo(f"   Failed: {failed}")
    
    if 'metrics' in results:
        metrics = results['metrics']
        click.echo(f"   Success Rate: {metrics.get('success_rate', 0):.2%}")
        click.echo(f"   Average Execution Time: {metrics.get('average_execution_time', 0):.2f}s")
        click.echo(f"   Throughput: {metrics.get('throughput', 0):.2f} tasks/s")


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
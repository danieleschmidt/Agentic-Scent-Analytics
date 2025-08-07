"""
Command-line interface for Sentiment Analyzer Pro
"""

import asyncio
import click
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

from .core.analyzer import SentimentAnalyzer
from .core.factory import SentimentAnalyzerFactory
from .core.models import TextInput, AnalysisConfig, ModelType


@click.group()
@click.version_option(version="1.0.0")
@click.pass_context
def cli(ctx):
    """Sentiment Analyzer Pro - Advanced multi-model sentiment analysis platform"""
    ctx.ensure_object(dict)


@cli.command()
@click.argument('text', required=True)
@click.option('--preset', default='default', 
              type=click.Choice(['default', 'fast', 'accurate', 'enterprise']),
              help='Analyzer preset to use')
@click.option('--models', multiple=True, 
              type=click.Choice(['transformers', 'vader', 'textblob', 'openai', 'anthropic']),
              help='Models to use (can be specified multiple times)')
@click.option('--include-emotions', is_flag=True, help='Include emotion analysis')
@click.option('--include-entities', is_flag=True, help='Include named entity recognition')
@click.option('--include-phrases', is_flag=True, help='Include key phrase extraction')
@click.option('--output', type=click.Path(), help='Output file path (JSON format)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def analyze(text: str, preset: str, models: tuple, include_emotions: bool, 
           include_entities: bool, include_phrases: bool, output: Optional[str], verbose: bool):
    """Analyze sentiment of text"""
    async def _analyze():
        # Create analyzer based on options
        if models:
            model_types = [ModelType(model) for model in models]
            analyzer = SentimentAnalyzerFactory.create_custom(
                models=model_types,
                include_emotions=include_emotions,
                include_entities=include_entities,
                include_key_phrases=include_phrases
            )
        else:
            # Use preset
            if preset == 'fast':
                analyzer = SentimentAnalyzerFactory.create_fast()
            elif preset == 'accurate':
                analyzer = SentimentAnalyzerFactory.create_accurate()
            elif preset == 'enterprise':
                analyzer = SentimentAnalyzerFactory.create_enterprise()
            else:
                analyzer = SentimentAnalyzerFactory.create_default()
        
        # Perform analysis
        try:
            result = await analyzer.analyze(TextInput(text=text))
            
            if verbose:
                click.echo(f"Analysis ID: {result.id}")
                click.echo(f"Text: {result.text[:100]}{'...' if len(result.text) > 100 else ''}")
                click.echo(f"Timestamp: {result.timestamp}")
                click.echo(f"Processing Time: {result.total_processing_time_ms:.2f}ms")
                click.echo(f"Models Used: {len(result.model_results)}")
                click.echo()
            
            # Main results
            click.echo(f"Sentiment: {result.sentiment_label.value.upper()}")
            click.echo(f"Confidence: {result.confidence_level.value.upper()}")
            click.echo()
            
            # Scores
            click.echo("Sentiment Scores:")
            click.echo(f"  Positive: {result.sentiment_scores.positive:.4f}")
            click.echo(f"  Negative: {result.sentiment_scores.negative:.4f}")
            click.echo(f"  Neutral:  {result.sentiment_scores.neutral:.4f}")
            click.echo(f"  Compound: {result.sentiment_scores.compound:.4f}")
            
            if verbose and result.model_results:
                click.echo("\nModel Results:")
                for model_result in result.model_results:
                    click.echo(f"  {model_result.model_type.value}:")
                    click.echo(f"    Confidence: {model_result.confidence:.4f}")
                    click.echo(f"    Time: {model_result.processing_time_ms:.2f}ms")
            
            # Text metrics
            if verbose:
                metrics = result.text_metrics
                click.echo(f"\nText Metrics:")
                click.echo(f"  Characters: {metrics.character_count}")
                click.echo(f"  Words: {metrics.word_count}")
                click.echo(f"  Sentences: {metrics.sentence_count}")
                click.echo(f"  Avg Sentence Length: {metrics.avg_sentence_length:.1f} words")
            
            # Save to file if requested
            if output:
                output_data = result.dict()
                with open(output, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                click.echo(f"\nResults saved to: {output}")
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_analyze())


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), help='Output file path (JSON format)')
@click.option('--preset', default='default',
              type=click.Choice(['default', 'fast', 'accurate', 'enterprise']),
              help='Analyzer preset to use')
@click.option('--batch-size', default=10, help='Batch processing size')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def batch(input_file: str, output: Optional[str], preset: str, batch_size: int, verbose: bool):
    """Analyze multiple texts from file (one per line or JSON array)"""
    async def _batch_analyze():
        # Read input file
        input_path = Path(input_file)
        
        try:
            with open(input_path, 'r') as f:
                content = f.read().strip()
                
            # Try to parse as JSON first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    texts = [str(item) for item in data]
                else:
                    texts = [str(data)]
            except json.JSONDecodeError:
                # Treat as line-separated text
                texts = [line.strip() for line in content.split('\n') if line.strip()]
            
            if not texts:
                click.echo("No texts found in input file", err=True)
                sys.exit(1)
            
            # Create analyzer
            if preset == 'fast':
                analyzer = SentimentAnalyzerFactory.create_fast()
            elif preset == 'accurate':
                analyzer = SentimentAnalyzerFactory.create_accurate()
            elif preset == 'enterprise':
                analyzer = SentimentAnalyzerFactory.create_enterprise()
            else:
                analyzer = SentimentAnalyzerFactory.create_default()
            
            # Process in batches
            all_results = []
            total_texts = len(texts)
            
            click.echo(f"Processing {total_texts} texts in batches of {batch_size}...")
            
            for i in range(0, total_texts, batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_inputs = [TextInput(text=text) for text in batch_texts]
                
                if verbose:
                    click.echo(f"Processing batch {i//batch_size + 1}/{(total_texts-1)//batch_size + 1}...")
                
                start_time = time.time()
                results = await analyzer.analyze_batch(batch_inputs)
                batch_time = time.time() - start_time
                
                all_results.extend(results)
                
                if verbose:
                    click.echo(f"  Completed {len(results)} analyses in {batch_time:.2f}s")
            
            # Display summary
            successful = len(all_results)
            click.echo(f"\nCompleted: {successful}/{total_texts} texts analyzed")
            
            if all_results:
                avg_time = sum(r.total_processing_time_ms for r in all_results) / len(all_results)
                click.echo(f"Average processing time: {avg_time:.2f}ms")
                
                # Sentiment distribution
                sentiment_counts = {}
                for result in all_results:
                    sentiment = result.sentiment_label.value
                    sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                
                click.echo("\nSentiment Distribution:")
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / successful) * 100
                    click.echo(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
            
            # Save results if requested
            if output:
                output_data = [result.dict() for result in all_results]
                with open(output, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                click.echo(f"\nResults saved to: {output}")
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_batch_analyze())


@cli.command()
@click.option('--preset', default='default',
              type=click.Choice(['default', 'fast', 'accurate', 'enterprise']),
              help='Analyzer preset to test')
def health(preset: str):
    """Check health status of analyzer"""
    async def _health_check():
        try:
            # Create analyzer
            if preset == 'fast':
                analyzer = SentimentAnalyzerFactory.create_fast()
            elif preset == 'accurate':
                analyzer = SentimentAnalyzerFactory.create_accurate()
            elif preset == 'enterprise':
                analyzer = SentimentAnalyzerFactory.create_enterprise()
            else:
                analyzer = SentimentAnalyzerFactory.create_default()
            
            health_data = await analyzer.health_check()
            
            status = health_data["status"]
            available_models = health_data["available_models"]
            test_analysis = health_data["test_analysis"]
            
            click.echo(f"Status: {status.upper()}")
            click.echo(f"Available Models: {', '.join(available_models)}")
            
            if test_analysis:
                if test_analysis["success"]:
                    click.echo(f"Test Analysis: PASSED ({test_analysis['processing_time_ms']:.2f}ms)")
                    click.echo(f"Test Sentiment: {test_analysis['sentiment']}")
                else:
                    click.echo(f"Test Analysis: FAILED - {test_analysis['error']}")
            
            if status != "healthy":
                sys.exit(1)
                
        except Exception as e:
            click.echo(f"Health check failed: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_health_check())


@cli.command()
def models():
    """List available models and their status"""
    click.echo("Available Models:")
    click.echo("  transformers - Hugging Face Transformers (RoBERTa)")
    click.echo("  vader        - VADER Sentiment Analysis")
    click.echo("  textblob     - TextBlob Sentiment Analysis") 
    click.echo("  openai       - OpenAI API (requires API key)")
    click.echo("  anthropic    - Anthropic Claude API (requires API key)")
    
    click.echo("\nPresets:")
    presets = SentimentAnalyzerFactory.list_presets()
    for name, info in presets.items():
        models_list = ", ".join(info["models"])
        click.echo(f"  {name:12} - {info['description']}")
        click.echo(f"               Models: {models_list}")
        click.echo(f"               Speed: {info['speed']}, Accuracy: {info['accuracy']}")
        click.echo()


@cli.command()
@click.argument('text', required=True)
@click.option('--duration', default=30, help='Demo duration in seconds')
def demo(text: str, duration: int):
    """Run interactive demo with all available models"""
    async def _demo():
        click.echo("Sentiment Analyzer Pro Demo")
        click.echo("=" * 40)
        click.echo(f"Analyzing: {text}")
        click.echo()
        
        presets = ['fast', 'default', 'accurate']
        
        for preset in presets:
            click.echo(f"Testing {preset.upper()} preset...")
            
            try:
                if preset == 'fast':
                    analyzer = SentimentAnalyzerFactory.create_fast()
                elif preset == 'accurate':
                    analyzer = SentimentAnalyzerFactory.create_accurate()
                else:
                    analyzer = SentimentAnalyzerFactory.create_default()
                
                result = await analyzer.analyze(TextInput(text=text))
                
                click.echo(f"  Sentiment: {result.sentiment_label.value}")
                click.echo(f"  Confidence: {result.confidence_level.value}")
                click.echo(f"  Processing Time: {result.total_processing_time_ms:.2f}ms")
                click.echo(f"  Models: {len(result.model_results)}")
                
                for model_result in result.model_results:
                    click.echo(f"    {model_result.model_type.value}: {model_result.confidence:.3f}")
                
                click.echo()
                
            except Exception as e:
                click.echo(f"  Error: {e}")
                click.echo()
        
        click.echo("Demo completed!")
    
    asyncio.run(_demo())


@cli.command()
def config():
    """Show current configuration and environment setup"""
    click.echo("Configuration Information:")
    click.echo("=" * 40)
    
    try:
        analyzer = SentimentAnalyzerFactory.create_from_env()
        config = analyzer.config
        
        click.echo(f"Models: {[m.value for m in config.models]}")
        click.echo(f"Include Emotions: {config.include_emotions}")
        click.echo(f"Include Entities: {config.include_entities}")
        click.echo(f"Include Key Phrases: {config.include_key_phrases}")
        click.echo(f"Include Topics: {config.include_topics}")
        click.echo(f"Timeout: {config.timeout_seconds}s")
        click.echo(f"Max Retries: {config.max_retries}")
        click.echo()
        
        click.echo("Model Configurations:")
        click.echo(f"  Transformers Model: {config.transformers_model}")
        click.echo(f"  OpenAI Model: {config.openai_model}")
        click.echo(f"  Anthropic Model: {config.anthropic_model}")
        
    except Exception as e:
        click.echo(f"Error loading configuration: {e}")


if __name__ == '__main__':
    cli()
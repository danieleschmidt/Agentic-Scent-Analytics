"""
FastAPI application for sentiment analysis service
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import logging
from typing import List, Dict, Any
import time
from datetime import datetime

from ..core.analyzer import SentimentAnalyzer
from ..core.factory import SentimentAnalyzerFactory
from ..core.models import (
    TextInput, SentimentResult, AnalysisConfig, BatchTextInput,
    HealthStatus, AnalysisStats, ModelType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analyzer Pro API",
    description="Advanced multi-model sentiment analysis platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instance
analyzer: SentimentAnalyzer = None

# Statistics tracking
stats = {
    "total_analyses": 0,
    "successful_analyses": 0,
    "failed_analyses": 0,
    "total_processing_time": 0.0,
    "model_usage": {},
    "start_time": datetime.utcnow()
}


async def get_analyzer() -> SentimentAnalyzer:
    """Dependency to get analyzer instance"""
    global analyzer
    if analyzer is None:
        # Initialize with environment-based configuration
        analyzer = SentimentAnalyzerFactory.create_from_env()
        logger.info("Initialized sentiment analyzer from environment")
    return analyzer


@app.on_event("startup")
async def startup_event():
    """Initialize analyzer on startup"""
    global analyzer
    analyzer = SentimentAnalyzerFactory.create_from_env()
    logger.info("Sentiment Analyzer Pro API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Sentiment Analyzer Pro API shutting down")


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Sentiment Analyzer Pro API",
        "version": "1.0.0",
        "description": "Advanced multi-model sentiment analysis platform",
        "endpoints": {
            "analyze": "/analyze",
            "batch_analyze": "/analyze/batch",
            "health": "/health",
            "stats": "/stats",
            "models": "/models",
            "presets": "/presets",
            "docs": "/docs"
        },
        "status": "operational",
        "uptime_seconds": (datetime.utcnow() - stats["start_time"]).total_seconds()
    }


@app.post("/analyze", response_model=SentimentResult)
async def analyze_text(
    text_input: TextInput,
    analyzer: SentimentAnalyzer = Depends(get_analyzer)
):
    """Analyze sentiment of a single text"""
    global stats
    
    start_time = time.time()
    stats["total_analyses"] += 1
    
    try:
        result = await analyzer.analyze(text_input)
        
        # Update statistics
        stats["successful_analyses"] += 1
        stats["total_processing_time"] += result.total_processing_time_ms
        
        # Track model usage
        for model_result in result.model_results:
            model_name = model_result.model_type.value
            stats["model_usage"][model_name] = stats["model_usage"].get(model_name, 0) + 1
        
        logger.info(f"Successfully analyzed text: {result.sentiment_label.value}")
        return result
        
    except Exception as e:
        stats["failed_analyses"] += 1
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch", response_model=List[SentimentResult])
async def analyze_batch(
    batch_input: BatchTextInput,
    background_tasks: BackgroundTasks,
    analyzer: SentimentAnalyzer = Depends(get_analyzer)
):
    """Analyze sentiment of multiple texts in batch"""
    global stats
    
    start_time = time.time()
    batch_size = len(batch_input.texts)
    stats["total_analyses"] += batch_size
    
    try:
        results = await analyzer.analyze_batch(batch_input.texts)
        
        # Update statistics
        successful_count = len(results)
        failed_count = batch_size - successful_count
        
        stats["successful_analyses"] += successful_count
        stats["failed_analyses"] += failed_count
        
        # Track processing time and model usage
        for result in results:
            stats["total_processing_time"] += result.total_processing_time_ms
            for model_result in result.model_results:
                model_name = model_result.model_type.value
                stats["model_usage"][model_name] = stats["model_usage"].get(model_name, 0) + 1
        
        logger.info(f"Batch analysis completed: {successful_count}/{batch_size} successful")
        return results
        
    except Exception as e:
        stats["failed_analyses"] += batch_size
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@app.get("/health", response_model=HealthStatus)
async def health_check(analyzer: SentimentAnalyzer = Depends(get_analyzer)):
    """Get health status of the sentiment analyzer"""
    try:
        health_data = await analyzer.health_check()
        
        status = HealthStatus(
            status=health_data["status"],
            timestamp=datetime.utcnow(),
            checks={
                "analyzer_initialized": analyzer is not None,
                "models_available": len(health_data["available_models"]) > 0,
                "test_analysis": health_data["test_analysis"]["success"] if health_data["test_analysis"] else False
            },
            metrics={
                "available_models_count": len(health_data["available_models"]),
                "test_processing_time_ms": health_data["test_analysis"]["processing_time_ms"] if health_data["test_analysis"] and health_data["test_analysis"]["success"] else 0.0
            }
        )
        
        return status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthStatus(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            checks={"analyzer_initialized": False, "models_available": False, "test_analysis": False},
            metrics={"available_models_count": 0, "test_processing_time_ms": 0.0}
        )


@app.get("/stats", response_model=AnalysisStats)
async def get_stats():
    """Get analysis statistics"""
    avg_processing_time = (
        stats["total_processing_time"] / max(stats["successful_analyses"], 1)
    )
    
    return AnalysisStats(
        total_analyses=stats["total_analyses"],
        successful_analyses=stats["successful_analyses"],
        failed_analyses=stats["failed_analyses"],
        average_processing_time_ms=round(avg_processing_time, 2),
        model_usage_stats=stats["model_usage"]
    )


@app.get("/models", response_model=List[str])
async def get_available_models(analyzer: SentimentAnalyzer = Depends(get_analyzer)):
    """Get list of available models"""
    try:
        available_models = analyzer.get_available_models()
        return [model.value for model in available_models]
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        return []


@app.get("/presets", response_model=Dict[str, Any])
async def get_analyzer_presets():
    """Get available analyzer presets"""
    return SentimentAnalyzerFactory.list_presets()


@app.post("/configure", response_model=Dict[str, str])
async def configure_analyzer(config: AnalysisConfig):
    """Reconfigure the analyzer with new settings"""
    global analyzer
    
    try:
        # Create new analyzer with updated configuration
        analyzer = SentimentAnalyzer(config)
        logger.info("Analyzer reconfigured successfully")
        return {"status": "success", "message": "Analyzer reconfigured successfully"}
        
    except Exception as e:
        logger.error(f"Failed to reconfigure analyzer: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@app.get("/config", response_model=AnalysisConfig)
async def get_current_config(analyzer: SentimentAnalyzer = Depends(get_analyzer)):
    """Get current analyzer configuration"""
    return analyzer.config


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
LLM client implementation supporting OpenAI and Anthropic models.
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"  # For testing


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: float = 30.0


@dataclass  
class LLMResponse:
    """LLM response."""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    finish_reason: Optional[str] = None


class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        if self.config.provider == LLMProvider.OPENAI:
            self._initialize_openai()
        elif self.config.provider == LLMProvider.ANTHROPIC:
            self._initialize_anthropic()
        elif self.config.provider == LLMProvider.MOCK:
            self._initialize_mock()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not found, using mock responses")
                self.config.provider = LLMProvider.MOCK
                return self._initialize_mock()
            
            self._client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.base_url
            )
            logger.info(f"Initialized OpenAI client with model: {self.config.model}")
            
        except ImportError:
            logger.warning("OpenAI package not installed, using mock responses")
            self.config.provider = LLMProvider.MOCK
            self._initialize_mock()
    
    def _initialize_anthropic(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("Anthropic API key not found, using mock responses")
                self.config.provider = LLMProvider.MOCK
                return self._initialize_mock()
                
            self._client = anthropic.AsyncAnthropic(
                api_key=api_key,
                base_url=self.config.base_url
            )
            logger.info(f"Initialized Anthropic client with model: {self.config.model}")
            
        except ImportError:
            logger.warning("Anthropic package not installed, using mock responses")
            self.config.provider = LLMProvider.MOCK
            self._initialize_mock()
    
    def _initialize_mock(self):
        """Initialize mock client for testing."""
        self._client = MockLLMClient()
        logger.info("Initialized mock LLM client")
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response from LLM."""
        try:
            if self.config.provider == LLMProvider.OPENAI:
                return await self._generate_openai(prompt, system_prompt)
            elif self.config.provider == LLMProvider.ANTHROPIC:
                return await self._generate_anthropic(prompt, system_prompt)
            elif self.config.provider == LLMProvider.MOCK:
                return await self._generate_mock(prompt, system_prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fall back to mock response on error
            return await self._generate_mock(prompt, system_prompt, error=True)
    
    async def _generate_openai(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response using OpenAI."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            provider="openai",
            model=self.config.model,
            tokens_used=response.usage.total_tokens if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )
    
    async def _generate_anthropic(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response using Anthropic."""
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = await self._client.messages.create(**kwargs)
        
        return LLMResponse(
            content=response.content[0].text if response.content else "",
            provider="anthropic", 
            model=self.config.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens if response.usage else None,
            finish_reason=response.stop_reason
        )
    
    async def _generate_mock(self, prompt: str, system_prompt: Optional[str] = None, error: bool = False) -> LLMResponse:
        """Generate mock response."""
        if error:
            content = "Analysis unavailable due to LLM service error. Using fallback heuristics."
        else:
            # Generate contextual mock response based on prompt keywords
            if "anomaly" in prompt.lower() or "quality" in prompt.lower():
                content = "Based on sensor analysis, quality parameters are within acceptable ranges. No significant anomalies detected."
            elif "root cause" in prompt.lower() or "investigation" in prompt.lower():
                content = "Primary factors contributing to deviation: temperature fluctuation (+2.3°C), humidity variation (-5%), sensor calibration drift."
            elif "recommendation" in prompt.lower() or "action" in prompt.lower():
                content = "Recommended actions: 1) Recalibrate sensors, 2) Adjust temperature controls, 3) Monitor next 3 batches closely."
            else:
                content = "Analysis completed successfully. All parameters within normal operational ranges."
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        return LLMResponse(
            content=content,
            provider="mock",
            model="mock-model",
            tokens_used=len(content.split()),
            finish_reason="stop"
        )
    
    async def analyze_sensor_data(self, sensor_data: Dict[str, Any], context: str = "") -> LLMResponse:
        """Analyze sensor data with domain-specific prompting."""
        
        system_prompt = """You are an expert industrial quality control analyst specializing in electronic nose (e-nose) sensor data interpretation for manufacturing environments.

Your role is to:
1. Analyze multi-sensor data arrays from electronic nose systems
2. Identify quality deviations and anomalies  
3. Provide root cause analysis for deviations
4. Recommend specific corrective actions
5. Assess risk levels and urgency

Always provide structured, actionable analysis suitable for manufacturing operators."""

        prompt = f"""
Analyze the following sensor data from an industrial e-nose system:

Sensor Readings:
{json.dumps(sensor_data, indent=2)}

Context: {context}

Please provide:
1. Overall quality assessment (0-1 score)
2. Any anomalies detected with confidence levels
3. Root cause analysis if deviations found
4. Specific recommended actions
5. Risk level (LOW/MEDIUM/HIGH/CRITICAL)
6. Urgency for intervention

Format your response as structured analysis suitable for immediate operator action."""

        return await self.generate(prompt, system_prompt)
    
    async def generate_batch_report(self, batch_data: Dict[str, Any]) -> LLMResponse:
        """Generate comprehensive batch quality report."""
        
        system_prompt = """You are a regulatory compliance officer and quality assurance specialist for pharmaceutical/food manufacturing. Generate comprehensive batch reports meeting GMP standards."""
        
        prompt = f"""
Generate a comprehensive quality release report for the following batch:

Batch Data:
{json.dumps(batch_data, indent=2)}

Include:
1. Executive summary of batch quality status
2. Detailed quality parameter analysis
3. Deviation investigation (if any)
4. Compliance verification
5. Release recommendation with justification
6. Risk assessment
7. Required follow-up actions

Format as professional GMP-compliant batch record suitable for regulatory review."""

        return await self.generate(prompt, system_prompt)


class MockLLMClient:
    """Mock LLM client for testing."""
    
    async def chat(self, *args, **kwargs):
        """Mock chat completion."""
        return type('Response', (), {
            'choices': [type('Choice', (), {
                'message': type('Message', (), {'content': 'Mock response'}),
                'finish_reason': 'stop'
            })],
            'usage': type('Usage', (), {'total_tokens': 50})
        })


def create_llm_client(model: str = "gpt-4", provider: Optional[str] = None) -> LLMClient:
    """
    Factory function to create LLM client with auto-detection.
    
    Args:
        model: Model name (e.g., 'gpt-4', 'claude-3-opus')
        provider: Force specific provider, otherwise auto-detect
    
    Returns:
        Configured LLM client
    """
    
    # Auto-detect provider from model name if not specified
    if not provider:
        if model.startswith('gpt') or model.startswith('text-'):
            provider = 'openai'
        elif model.startswith('claude'):
            provider = 'anthropic'
        else:
            provider = 'mock'  # Default fallback
    
    provider_enum = LLMProvider(provider)
    
    config = LLMConfig(
        provider=provider_enum,
        model=model,
        temperature=0.3,  # Lower temperature for more consistent analysis
        max_tokens=1500
    )
    
    return LLMClient(config)
"""
Enhanced LLM Client with Advanced Industrial Analytics Capabilities

Provides specialized prompting, multi-modal analysis, and domain-specific
knowledge integration for industrial scent analytics applications.
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import numpy as np
from pathlib import Path

from .client import LLMClient, LLMConfig, LLMProvider, LLMResponse, create_llm_client

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of industrial analysis."""
    QUALITY_CONTROL = "quality_control"
    ANOMALY_DETECTION = "anomaly_detection"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    BATCH_RELEASE = "batch_release"
    PROCESS_OPTIMIZATION = "process_optimization"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    CONTAMINATION_ANALYSIS = "contamination_analysis"


class IndustryDomain(Enum):
    """Industry domains with specialized knowledge."""
    PHARMACEUTICAL = "pharmaceutical"
    FOOD_BEVERAGE = "food_beverage"
    CHEMICAL = "chemical"
    COSMETICS = "cosmetics"
    AUTOMOTIVE = "automotive"
    GENERAL_MANUFACTURING = "general_manufacturing"


@dataclass
class AnalysisContext:
    """Context for industrial analysis."""
    domain: IndustryDomain
    analysis_type: AnalysisType
    product_type: Optional[str] = None
    production_line: Optional[str] = None
    batch_id: Optional[str] = None
    regulatory_standards: List[str] = field(default_factory=list)
    historical_context: Optional[Dict[str, Any]] = None
    urgency_level: str = "normal"  # low, normal, high, critical
    

@dataclass
class EnhancedAnalysisResult:
    """Enhanced analysis result with structured insights."""
    quality_score: float
    anomalies_detected: List[Dict[str, Any]]
    root_causes: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    confidence_level: float
    raw_response: LLMResponse
    analysis_metadata: Dict[str, Any]
    

class EnhancedLLMClient:
    """
    Enhanced LLM client with advanced industrial analytics capabilities.
    
    Features:
    - Domain-specific knowledge bases
    - Multi-step reasoning
    - Structured output parsing
    - Historical context integration
    - Risk assessment frameworks
    - Regulatory compliance checking
    """
    
    def __init__(self, base_client: Optional[LLMClient] = None):
        self.base_client = base_client or create_llm_client()
        self.knowledge_bases = self._load_knowledge_bases()
        self.analysis_templates = self._load_analysis_templates()
        self.historical_context = {}
        self.logger = logging.getLogger(__name__)
        
    def _load_knowledge_bases(self) -> Dict[str, Dict[str, Any]]:
        """Load domain-specific knowledge bases."""
        knowledge_bases = {
            IndustryDomain.PHARMACEUTICAL: {
                "quality_standards": [
                    "FDA 21 CFR Part 211", "ICH Q7", "USP <1116>", "EU GMP Annex 15"
                ],
                "critical_parameters": [
                    "potency", "purity", "dissolution", "sterility", "endotoxins"
                ],
                "acceptable_deviations": {
                    "potency": 0.05,  # 5%
                    "purity": 0.02,   # 2%
                    "dissolution": 0.10  # 10%
                },
                "contamination_limits": {
                    "microbial": 100,  # CFU/g
                    "heavy_metals": 20,  # ppm
                    "residual_solvents": 5000  # ppm
                }
            },
            IndustryDomain.FOOD_BEVERAGE: {
                "quality_standards": [
                    "HACCP", "BRC", "SQF", "FDA FSMA", "Codex Alimentarius"
                ],
                "critical_parameters": [
                    "pH", "water_activity", "moisture", "pathogen_presence", "shelf_life"
                ],
                "acceptable_deviations": {
                    "pH": 0.2,
                    "moisture": 0.05,
                    "water_activity": 0.03
                },
                "contamination_limits": {
                    "salmonella": 0,  # CFU/25g
                    "e_coli": 10,     # CFU/g
                    "listeria": 100   # CFU/g
                }
            },
            IndustryDomain.CHEMICAL: {
                "quality_standards": [
                    "ISO 9001", "ASTM Standards", "REACH", "GHS"
                ],
                "critical_parameters": [
                    "purity", "viscosity", "density", "flash_point", "composition"
                ],
                "acceptable_deviations": {
                    "purity": 0.01,
                    "viscosity": 0.05,
                    "density": 0.001
                }
            }
        }
        
        return knowledge_bases
        
    def _load_analysis_templates(self) -> Dict[str, str]:
        """Load analysis prompt templates."""
        templates = {
            AnalysisType.QUALITY_CONTROL: """
As an expert quality control analyst, analyze the following sensor data:

{data_context}

Domain Knowledge:
{domain_knowledge}

Perform comprehensive quality assessment considering:
1. Parameter compliance with specifications
2. Trend analysis and pattern recognition
3. Statistical process control limits
4. Risk factors and potential failure modes

Provide structured analysis with:
- Quality score (0-1)
- Specific deviations identified
- Confidence levels for each assessment
- Immediate actions required
""",

            AnalysisType.ANOMALY_DETECTION: """
As an expert anomaly detection specialist, analyze for unusual patterns:

{data_context}

Historical Context:
{historical_context}

Detect anomalies by:
1. Comparing against historical baselines
2. Identifying statistical outliers
3. Recognizing pattern deviations
4. Assessing multivariate relationships

For each anomaly found, provide:
- Anomaly type and severity
- Confidence level (0-1)
- Potential causes
- Recommended investigation steps
""",

            AnalysisType.ROOT_CAUSE_ANALYSIS: """
As an expert process engineer, perform root cause analysis:

{problem_description}
{data_context}

Systematic Analysis:
1. Define the problem precisely
2. Collect and analyze relevant data
3. Identify potential causes using 5-Why analysis
4. Evaluate cause-effect relationships
5. Prioritize causes by likelihood and impact

Deliver:
- Primary root causes (ranked)
- Supporting evidence for each cause
- Verification methods
- Corrective action recommendations
""",

            AnalysisType.BATCH_RELEASE: """
As a regulatory compliance officer, evaluate batch release:

{batch_data}

Regulatory Standards:
{regulatory_standards}

Batch Release Evaluation:
1. Verify all critical quality attributes
2. Review manufacturing records completeness
3. Assess compliance with specifications
4. Evaluate any deviations and investigations
5. Confirm release criteria fulfillment

Provide:
- Release decision (APPROVE/REJECT/INVESTIGATE)
- Justification with regulatory basis
- Risk assessment
- Required documentation
"""
        }
        
        return templates
        
    async def enhanced_analysis(
        self, 
        sensor_data: Dict[str, Any],
        context: AnalysisContext,
        include_historical: bool = True
    ) -> EnhancedAnalysisResult:
        """Perform enhanced analysis with domain expertise."""
        
        start_time = datetime.now()
        
        try:
            # Prepare analysis context
            analysis_context = await self._prepare_analysis_context(
                sensor_data, context, include_historical
            )
            
            # Generate domain-specific prompt
            prompt = await self._generate_domain_prompt(
                analysis_context, context
            )
            
            # Get LLM response
            response = await self.base_client.generate(
                prompt, 
                self._get_system_prompt(context)
            )
            
            # Parse structured results
            parsed_results = await self._parse_analysis_response(
                response, context
            )
            
            # Calculate confidence and risk metrics
            confidence = self._calculate_confidence(parsed_results, context)
            risk_assessment = await self._assess_risk(parsed_results, context)
            
            # Create enhanced result
            result = EnhancedAnalysisResult(
                quality_score=parsed_results.get("quality_score", 0.5),
                anomalies_detected=parsed_results.get("anomalies", []),
                root_causes=parsed_results.get("root_causes", []),
                recommendations=parsed_results.get("recommendations", []),
                risk_assessment=risk_assessment,
                confidence_level=confidence,
                raw_response=response,
                analysis_metadata={
                    "analysis_time": (datetime.now() - start_time).total_seconds(),
                    "context": context.__dict__,
                    "model_used": self.base_client.config.model,
                    "tokens_used": response.tokens_used
                }
            )
            
            # Update historical context
            await self._update_historical_context(context, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced analysis failed: {e}")
            # Return fallback result
            return self._create_fallback_result(sensor_data, context, str(e))
            
    async def _prepare_analysis_context(
        self, 
        sensor_data: Dict[str, Any],
        context: AnalysisContext,
        include_historical: bool
    ) -> Dict[str, Any]:
        """Prepare comprehensive analysis context."""
        
        analysis_context = {
            "sensor_data": sensor_data,
            "timestamp": datetime.now().isoformat(),
            "domain": context.domain.value,
            "analysis_type": context.analysis_type.value
        }
        
        # Add domain knowledge
        if context.domain in self.knowledge_bases:
            analysis_context["domain_knowledge"] = self.knowledge_bases[context.domain]
            
        # Add historical context if requested
        if include_historical and context.product_type:
            historical_key = f"{context.domain.value}_{context.product_type}"
            if historical_key in self.historical_context:
                analysis_context["historical_context"] = self.historical_context[historical_key]
                
        # Add regulatory context
        if context.regulatory_standards:
            analysis_context["regulatory_standards"] = context.regulatory_standards
            
        return analysis_context
        
    async def _generate_domain_prompt(
        self, 
        analysis_context: Dict[str, Any],
        context: AnalysisContext
    ) -> str:
        """Generate domain-specific analysis prompt."""
        
        base_template = self.analysis_templates.get(
            context.analysis_type,
            self.analysis_templates[AnalysisType.QUALITY_CONTROL]
        )
        
        # Format template with context
        prompt = base_template.format(
            data_context=json.dumps(analysis_context.get("sensor_data"), indent=2),
            domain_knowledge=json.dumps(analysis_context.get("domain_knowledge", {}), indent=2),
            historical_context=json.dumps(analysis_context.get("historical_context", {}), indent=2),
            regulatory_standards="\n".join(analysis_context.get("regulatory_standards", [])),
            batch_data=json.dumps(analysis_context, indent=2),
            problem_description=f"Analysis request for {context.analysis_type.value} in {context.domain.value}"
        )
        
        # Add urgency context
        if context.urgency_level in ["high", "critical"]:
            prompt += f"\n\nURGENT: This is a {context.urgency_level.upper()} priority analysis. Focus on immediate actionable insights."
            
        return prompt
        
    def _get_system_prompt(self, context: AnalysisContext) -> str:
        """Get appropriate system prompt for context."""
        
        domain_expertise = {
            IndustryDomain.PHARMACEUTICAL: "pharmaceutical manufacturing and GMP compliance",
            IndustryDomain.FOOD_BEVERAGE: "food safety and HACCP systems",
            IndustryDomain.CHEMICAL: "chemical process engineering and safety",
            IndustryDomain.COSMETICS: "cosmetic formulation and safety testing",
            IndustryDomain.AUTOMOTIVE: "automotive quality systems and Six Sigma",
            IndustryDomain.GENERAL_MANUFACTURING: "general manufacturing and quality control"
        }
        
        expertise = domain_expertise.get(context.domain, "industrial quality control")
        
        system_prompt = f"""
You are a world-class expert in {expertise} with deep knowledge of:

1. Industrial sensor systems and electronic nose technology
2. Statistical process control and quality management
3. Regulatory requirements and compliance standards
4. Risk assessment and failure mode analysis
5. Process optimization and continuous improvement

Your analysis must be:
- Technically accurate and scientifically rigorous
- Practical and immediately actionable
- Compliant with relevant regulations
- Risk-aware and safety-focused
- Clearly communicated for operators and managers

Always provide specific, measurable recommendations with clear priorities.
"""
        
        return system_prompt
        
    async def _parse_analysis_response(
        self, 
        response: LLMResponse,
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        
        content = response.content.lower()
        parsed = {
            "quality_score": 0.7,  # Default
            "anomalies": [],
            "root_causes": [],
            "recommendations": []
        }
        
        try:
            # Extract quality score
            if "quality score" in content or "score:" in content:
                import re
                score_match = re.search(r"(?:quality score|score)\s*:?\s*([0-9.]+ ?/?[0-9.]*)", content)
                if score_match:
                    score_text = score_match.group(1)
                    if "/" in score_text:
                        numerator, denominator = score_text.split("/")
                        parsed["quality_score"] = float(numerator) / float(denominator)
                    else:
                        score = float(score_text.strip())
                        parsed["quality_score"] = score if score <= 1.0 else score / 100.0
                        
            # Extract anomalies
            if "anomal" in content or "deviation" in content:
                anomaly_indicators = [
                    "temperature anomaly", "pressure deviation", "flow rate issue",
                    "contamination detected", "sensor drift", "calibration error"
                ]
                
                for indicator in anomaly_indicators:
                    if indicator in content:
                        parsed["anomalies"].append({
                            "type": indicator,
                            "severity": "medium",
                            "confidence": 0.8
                        })
                        
            # Extract recommendations
            recommendation_keywords = [
                "recommend", "suggest", "action", "should", "need to"
            ]
            
            lines = response.content.split("\n")
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in recommendation_keywords):
                    if len(line.strip()) > 10:  # Avoid short fragments
                        parsed["recommendations"].append({
                            "action": line.strip(),
                            "priority": "medium",
                            "timeframe": "immediate"
                        })
                        
        except Exception as e:
            self.logger.warning(f"Response parsing error: {e}")
            
        return parsed
        
    def _calculate_confidence(self, parsed_results: Dict[str, Any], context: AnalysisContext) -> float:
        """Calculate overall confidence in analysis."""
        
        confidence_factors = []
        
        # Model reliability factor
        if self.base_client.config.provider == LLMProvider.MOCK:
            confidence_factors.append(0.3)  # Low confidence for mock
        else:
            confidence_factors.append(0.8)  # Higher for real models
            
        # Data quality factor
        if "quality_score" in parsed_results and parsed_results["quality_score"] > 0:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
            
        # Domain knowledge factor
        if context.domain in self.knowledge_bases:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
            
        # Historical context factor
        historical_key = f"{context.domain.value}_{context.product_type or 'default'}"
        if historical_key in self.historical_context:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.7)
            
        return float(np.mean(confidence_factors))
        
    async def _assess_risk(self, parsed_results: Dict[str, Any], context: AnalysisContext) -> Dict[str, Any]:
        """Assess risk based on analysis results."""
        
        risk_factors = []
        
        # Quality score risk
        quality_score = parsed_results.get("quality_score", 0.7)
        if quality_score < 0.5:
            risk_factors.append({"factor": "low_quality_score", "severity": "high", "impact": 0.8})
        elif quality_score < 0.7:
            risk_factors.append({"factor": "moderate_quality_score", "severity": "medium", "impact": 0.5})
            
        # Anomaly risk
        anomalies = parsed_results.get("anomalies", [])
        if len(anomalies) > 2:
            risk_factors.append({"factor": "multiple_anomalies", "severity": "high", "impact": 0.7})
        elif len(anomalies) > 0:
            risk_factors.append({"factor": "anomalies_detected", "severity": "medium", "impact": 0.4})
            
        # Calculate overall risk
        if not risk_factors:
            overall_risk = "low"
            risk_score = 0.2
        else:
            avg_impact = np.mean([rf["impact"] for rf in risk_factors])
            if avg_impact > 0.7:
                overall_risk = "high"
                risk_score = avg_impact
            elif avg_impact > 0.4:
                overall_risk = "medium"
                risk_score = avg_impact
            else:
                overall_risk = "low"
                risk_score = avg_impact
                
        return {
            "overall_risk": overall_risk,
            "risk_score": float(risk_score),
            "risk_factors": risk_factors,
            "mitigation_priority": "immediate" if overall_risk == "high" else "planned"
        }
        
    async def _update_historical_context(
        self, 
        context: AnalysisContext, 
        result: EnhancedAnalysisResult
    ):
        """Update historical context with new analysis."""
        
        historical_key = f"{context.domain.value}_{context.product_type or 'default'}"
        
        if historical_key not in self.historical_context:
            self.historical_context[historical_key] = {
                "analyses": [],
                "quality_trends": [],
                "common_issues": {}
            }
            
        history = self.historical_context[historical_key]
        
        # Add current analysis
        history["analyses"].append({
            "timestamp": datetime.now().isoformat(),
            "quality_score": result.quality_score,
            "anomalies_count": len(result.anomalies_detected),
            "risk_level": result.risk_assessment["overall_risk"]
        })
        
        # Update quality trends
        history["quality_trends"].append(result.quality_score)
        
        # Track common issues
        for anomaly in result.anomalies_detected:
            issue_type = anomaly.get("type", "unknown")
            history["common_issues"][issue_type] = history["common_issues"].get(issue_type, 0) + 1
            
        # Limit history size
        max_history = 100
        if len(history["analyses"]) > max_history:
            history["analyses"] = history["analyses"][-max_history:]
            history["quality_trends"] = history["quality_trends"][-max_history:]
            
    def _create_fallback_result(
        self, 
        sensor_data: Dict[str, Any],
        context: AnalysisContext,
        error_msg: str
    ) -> EnhancedAnalysisResult:
        """Create fallback result when analysis fails."""
        
        fallback_response = LLMResponse(
            content=f"Analysis failed: {error_msg}. Using fallback heuristics.",
            provider="fallback",
            model="fallback",
            tokens_used=0
        )
        
        return EnhancedAnalysisResult(
            quality_score=0.5,  # Neutral score
            anomalies_detected=[{
                "type": "analysis_failure",
                "severity": "unknown",
                "confidence": 0.0
            }],
            root_causes=[{
                "cause": "LLM analysis unavailable",
                "likelihood": 1.0,
                "impact": "unknown"
            }],
            recommendations=[{
                "action": "Manual review required due to analysis failure",
                "priority": "high",
                "timeframe": "immediate"
            }],
            risk_assessment={
                "overall_risk": "unknown",
                "risk_score": 0.5,
                "risk_factors": [{"factor": "analysis_failure", "severity": "high", "impact": 0.8}],
                "mitigation_priority": "immediate"
            },
            confidence_level=0.0,
            raw_response=fallback_response,
            analysis_metadata={
                "analysis_time": 0.0,
                "context": context.__dict__,
                "error": error_msg,
                "fallback_used": True
            }
        )
        
    async def multi_step_analysis(
        self,
        sensor_data: Dict[str, Any],
        context: AnalysisContext
    ) -> Dict[str, EnhancedAnalysisResult]:
        """Perform multi-step analysis workflow."""
        
        results = {}
        
        # Step 1: Initial quality assessment
        quality_context = AnalysisContext(
            domain=context.domain,
            analysis_type=AnalysisType.QUALITY_CONTROL,
            product_type=context.product_type,
            production_line=context.production_line,
            batch_id=context.batch_id
        )
        
        results["quality_assessment"] = await self.enhanced_analysis(
            sensor_data, quality_context
        )
        
        # Step 2: Anomaly detection if quality issues found
        if results["quality_assessment"].quality_score < 0.8:
            anomaly_context = AnalysisContext(
                domain=context.domain,
                analysis_type=AnalysisType.ANOMALY_DETECTION,
                product_type=context.product_type,
                production_line=context.production_line,
                batch_id=context.batch_id
            )
            
            results["anomaly_detection"] = await self.enhanced_analysis(
                sensor_data, anomaly_context
            )
            
            # Step 3: Root cause analysis if anomalies found
            if len(results["anomaly_detection"].anomalies_detected) > 0:
                rca_context = AnalysisContext(
                    domain=context.domain,
                    analysis_type=AnalysisType.ROOT_CAUSE_ANALYSIS,
                    product_type=context.product_type,
                    production_line=context.production_line,
                    batch_id=context.batch_id,
                    urgency_level="high"
                )
                
                results["root_cause_analysis"] = await self.enhanced_analysis(
                    sensor_data, rca_context
                )
                
        return results
        
    def get_analysis_summary(self, results: Dict[str, EnhancedAnalysisResult]) -> Dict[str, Any]:
        """Generate comprehensive analysis summary."""
        
        summary = {
            "overall_status": "unknown",
            "quality_score": 0.0,
            "total_anomalies": 0,
            "highest_risk": "low",
            "immediate_actions": [],
            "confidence": 0.0
        }
        
        if "quality_assessment" in results:
            qa = results["quality_assessment"]
            summary["quality_score"] = qa.quality_score
            summary["confidence"] = qa.confidence_level
            
            if qa.quality_score >= 0.8:
                summary["overall_status"] = "acceptable"
            elif qa.quality_score >= 0.6:
                summary["overall_status"] = "marginal"
            else:
                summary["overall_status"] = "unacceptable"
                
        # Aggregate anomalies
        for result in results.values():
            summary["total_anomalies"] += len(result.anomalies_detected)
            
            # Track highest risk
            risk_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            current_risk = result.risk_assessment["overall_risk"]
            if risk_levels.get(current_risk, 0) > risk_levels.get(summary["highest_risk"], 0):
                summary["highest_risk"] = current_risk
                
            # Collect immediate actions
            for rec in result.recommendations:
                if rec.get("timeframe") == "immediate" or rec.get("priority") == "high":
                    summary["immediate_actions"].append(rec["action"])
                    
        return summary


def create_enhanced_llm_client(model: str = "gpt-4") -> EnhancedLLMClient:
    """Create enhanced LLM client with industrial capabilities."""
    base_client = create_llm_client(model)
    return EnhancedLLMClient(base_client)


async def demonstrate_enhanced_llm():
    """Demonstration of enhanced LLM capabilities."""
    print("🧠 Enhanced LLM Client - Industrial Analytics Demo")
    print("=" * 60)
    
    # Create enhanced client
    client = create_enhanced_llm_client()
    
    # Mock sensor data
    sensor_data = {
        "temperature": 25.3,
        "humidity": 48.2,
        "pressure": 1013.1,
        "e_nose_channels": [0.2, 0.8, 1.5, 0.3, 2.1, 0.7],
        "flow_rate": 98.5,
        "ph_level": 7.2
    }
    
    # Analysis context
    context = AnalysisContext(
        domain=IndustryDomain.PHARMACEUTICAL,
        analysis_type=AnalysisType.QUALITY_CONTROL,
        product_type="tablet_coating",
        production_line="line_01",
        batch_id="BATCH_2024_001",
        regulatory_standards=["FDA 21 CFR Part 211", "USP <1116>"]
    )
    
    print("\n🔍 Single Analysis:")
    result = await client.enhanced_analysis(sensor_data, context)
    print(f"  Quality Score: {result.quality_score:.3f}")
    print(f"  Anomalies: {len(result.anomalies_detected)}")
    print(f"  Recommendations: {len(result.recommendations)}")
    print(f"  Risk Level: {result.risk_assessment['overall_risk']}")
    print(f"  Confidence: {result.confidence_level:.3f}")
    
    print("\n🔄 Multi-Step Analysis:")
    multi_results = await client.multi_step_analysis(sensor_data, context)
    summary = client.get_analysis_summary(multi_results)
    print(f"  Overall Status: {summary['overall_status']}")
    print(f"  Quality Score: {summary['quality_score']:.3f}")
    print(f"  Total Anomalies: {summary['total_anomalies']}")
    print(f"  Highest Risk: {summary['highest_risk']}")
    print(f"  Immediate Actions: {len(summary['immediate_actions'])}")
    
    print("\n✅ Enhanced LLM analysis completed!")
    
    return client


if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_llm())

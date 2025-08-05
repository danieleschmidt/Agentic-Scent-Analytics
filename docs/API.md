# Agentic Scent Analytics - API Documentation

## Overview

The Agentic Scent Analytics platform provides a comprehensive REST API for integration with manufacturing systems, quality control processes, and third-party applications. The API follows RESTful principles and provides both synchronous and asynchronous endpoints for different use cases.

## Base URL

```
Production: https://api.agentic-scent.company.com/v1
Development: http://localhost:8000/v1
```

## Authentication

The API uses token-based authentication with JWT tokens.

### Login
```http
POST /auth/login
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user_id": "uuid",
  "permissions": ["read", "write", "admin"]
}
```

### Using the Token
Include the token in the Authorization header:
```http
Authorization: Bearer <access_token>
```

## Core API Endpoints

### 1. Sensor Management

#### List Sensors
```http
GET /sensors
```

**Response:**
```json
{
  "sensors": [
    {
      "sensor_id": "e_nose_001",
      "sensor_type": "electronic_nose",
      "status": "online",
      "last_reading": "2024-01-15T10:30:00Z",
      "calibration_due": "2024-02-15T00:00:00Z",
      "metadata": {
        "channels": 32,
        "sampling_rate": 10.0
      }
    }
  ],
  "total": 1
}
```

#### Get Sensor Details
```http
GET /sensors/{sensor_id}
```

**Response:**
```json
{
  "sensor_id": "e_nose_001",
  "sensor_type": "electronic_nose",
  "status": "online",
  "current_reading": {
    "timestamp": "2024-01-15T10:30:00Z",
    "values": [100.5, 150.2, 200.1, ...],
    "quality_score": 0.95
  },
  "calibration": {
    "last_calibrated": "2024-01-15T08:00:00Z",
    "next_due": "2024-02-15T00:00:00Z",
    "status": "valid"
  }
}
```

#### Submit Sensor Reading
```http
POST /sensors/{sensor_id}/readings
Content-Type: application/json

{
  "timestamp": "2024-01-15T10:30:00Z",
  "values": [100.5, 150.2, 200.1, 125.7],
  "metadata": {
    "batch_id": "BATCH-001",
    "production_line": "line_1"
  },
  "quality_score": 0.95
}
```

**Response:**
```json
{
  "reading_id": "uuid",
  "status": "accepted",
  "validation_result": {
    "is_valid": true,
    "warnings": [],
    "quality_score": 0.95
  }
}
```

### 2. Quality Control

#### Analyze Reading
```http
POST /quality/analyze
Content-Type: application/json

{
  "sensor_reading": {
    "sensor_id": "e_nose_001",
    "timestamp": "2024-01-15T10:30:00Z",
    "values": [100.5, 150.2, 200.1, 125.7],
    "metadata": {"batch_id": "BATCH-001"}
  },
  "agent_id": "qc_agent_001"
}
```

**Response:**
```json
{
  "analysis_id": "uuid",
  "agent_id": "qc_agent_001",
  "confidence": 0.92,
  "anomaly_detected": false,
  "quality_score": 0.88,
  "analysis_details": {
    "contamination_analysis": {
      "detected": false,
      "confidence": 0.95
    },
    "process_analysis": {
      "within_limits": true,
      "deviation_score": 0.05
    },
    "trend_analysis": {
      "stable": true,
      "trend_direction": "stable"
    }
  },
  "recommendations": [
    "Continue normal operations",
    "Monitor for trend changes"
  ]
}
```

#### Batch Assessment
```http
POST /quality/batch/{batch_id}/assess
Content-Type: application/json

{
  "agent_id": "qc_agent_001",
  "readings": [
    {
      "sensor_id": "e_nose_001",
      "timestamp": "2024-01-15T10:30:00Z",
      "values": [100.5, 150.2, 200.1, 125.7]
    }
  ]
}
```

**Response:**
```json
{
  "assessment_id": "uuid",
  "batch_id": "BATCH-001",
  "overall_quality": 0.92,
  "confidence": 0.89,
  "risk_level": "LOW",
  "recommendation": "APPROVED",
  "passed_checks": [
    "contamination_check",
    "process_parameters_check",
    "trend_analysis_check"
  ],
  "failed_checks": [],
  "detailed_analysis": {
    "total_readings": 50,
    "anomaly_rate": 0.02,
    "quality_trend": "stable"
  }
}
```

#### Get Batch Status
```http
GET /quality/batch/{batch_id}/status
```

**Response:**
```json
{
  "batch_id": "BATCH-001",
  "status": "in_progress",
  "quality_assessments": [
    {
      "agent_id": "qc_agent_001",
      "overall_quality": 0.92,
      "timestamp": "2024-01-15T10:30:00Z",
      "recommendation": "APPROVED"
    }
  ],
  "current_quality": 0.92,
  "readings_count": 50,
  "last_updated": "2024-01-15T10:30:00Z"
}
```

### 3. Analytics and Fingerprinting

#### Create Fingerprint Model
```http
POST /analytics/fingerprints
Content-Type: application/json

{
  "product_id": "aspirin_500mg",
  "training_data": [
    {
      "sensor_id": "e_nose_001",
      "values": [100.5, 150.2, 200.1, 125.7],
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "parameters": {
    "embedding_dim": 64,
    "augmentation": true
  }
}
```

**Response:**
```json
{
  "model_id": "uuid",
  "product_id": "aspirin_500mg",
  "embedding_dim": 64,
  "similarity_threshold": 0.85,
  "training_samples": 100,
  "created_at": "2024-01-15T10:30:00Z",
  "status": "trained"
}
```

#### Compare to Fingerprint
```http
POST /analytics/fingerprints/{model_id}/compare
Content-Type: application/json

{
  "sensor_reading": {
    "sensor_id": "e_nose_001",
    "values": [105.2, 148.9, 202.3, 127.1],
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

**Response:**
```json
{
  "similarity_score": 0.92,
  "is_match": true,
  "confidence": 0.89,
  "deviation_channels": [],
  "analysis": "Sample matches reference fingerprint with high confidence"
}
```

#### List Fingerprint Models
```http
GET /analytics/fingerprints
```

**Response:**
```json
{
  "models": [
    {
      "model_id": "uuid",
      "product_id": "aspirin_500mg",
      "similarity_threshold": 0.85,
      "training_samples": 100,
      "created_at": "2024-01-15T10:30:00Z",
      "last_used": "2024-01-15T11:00:00Z"
    }
  ],
  "total": 1
}
```

### 4. Predictive Analytics

#### Quality Prediction
```http
POST /predictive/quality/predict
Content-Type: application/json

{
  "current_state": {
    "process_parameters": {
      "temperature": 25.0,
      "humidity": 45.0,
      "pressure": 1013.25
    },
    "recent_readings": [
      {
        "sensor_id": "e_nose_001",
        "values": [100.5, 150.2, 200.1, 125.7],
        "timestamp": "2024-01-15T10:30:00Z"
      }
    ]
  },
  "horizons": [1, 6, 24]
}
```

**Response:**
```json
{
  "predictions": {
    "1": {
      "horizon_hours": 1,
      "predicted_metrics": {
        "potency": 0.95,
        "dissolution": 0.92,
        "stability": 0.98,
        "contamination_risk": 0.02
      },
      "confidence_score": 0.89,
      "risk_factors": []
    },
    "6": {
      "horizon_hours": 6,
      "predicted_metrics": {
        "potency": 0.93,
        "dissolution": 0.90,
        "stability": 0.96,
        "contamination_risk": 0.05
      },
      "confidence_score": 0.82,
      "risk_factors": ["process_drift"]
    }
  },
  "insights": {
    "summary": "Quality expected to remain stable with minor decline over 6 hours",
    "intervention_recommended": false,
    "suggested_actions": ["Continue monitoring"],
    "risk_level": "LOW"
  }
}
```

### 5. Agent Management

#### List Agents
```http
GET /agents
```

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "qc_agent_001",
      "agent_type": "quality_control",
      "status": "active",
      "capabilities": ["anomaly_detection", "root_cause_analysis"],
      "performance": {
        "analyses_completed": 1500,
        "accuracy": 0.94,
        "avg_response_time": 0.15
      }
    }
  ],
  "total": 1
}
```

#### Get Agent Status
```http
GET /agents/{agent_id}/status
```

**Response:**
```json
{
  "agent_id": "qc_agent_001",
  "status": "active",
  "last_analysis": "2024-01-15T10:30:00Z",
  "performance_metrics": {
    "analyses_today": 150,
    "anomalies_detected": 3,
    "avg_confidence": 0.87,
    "processing_time_avg": 0.15
  },
  "health_check": {
    "status": "healthy",
    "last_check": "2024-01-15T10:29:00Z"
  }
}
```

#### Agent Configuration
```http
PUT /agents/{agent_id}/config
Content-Type: application/json

{
  "confidence_threshold": 0.8,
  "alert_threshold": 0.95,
  "parameters": {
    "sensitivity": "medium",
    "batch_size": 10
  }
}
```

### 6. Multi-Agent Coordination

#### Coordinate Analysis
```http
POST /coordination/analyze
Content-Type: application/json

{
  "sensor_reading": {
    "sensor_id": "e_nose_001",
    "values": [100.5, 150.2, 200.1, 125.7],
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "agents": ["qc_agent_001", "predictive_agent_001"],
  "coordination_type": "parallel"
}
```

**Response:**
```json
{
  "coordination_id": "uuid",
  "results": {
    "qc_agent_001": {
      "confidence": 0.92,
      "anomaly_detected": false,
      "processing_time": 0.12
    },
    "predictive_agent_001": {
      "confidence": 0.87,
      "risk_factors": [],
      "processing_time": 0.18
    }
  },
  "consensus": {
    "overall_confidence": 0.89,
    "recommendation": "CONTINUE",
    "agreement_level": 0.95
  }
}
```

#### Build Consensus
```http
POST /coordination/consensus
Content-Type: application/json

{
  "decision_prompt": "Should batch BATCH-001 be approved for release?",
  "agents": ["qc_agent_001", "qc_agent_002", "predictive_agent_001"],
  "voting_mechanism": "weighted_confidence",
  "context": {
    "batch_id": "BATCH-001",
    "quality_data": {...}
  }
}
```

**Response:**
```json
{
  "consensus_id": "uuid",
  "decision": "approve",
  "confidence": 0.91,
  "participating_agents": ["qc_agent_001", "qc_agent_002", "predictive_agent_001"],
  "voting_results": {
    "qc_agent_001": {
      "decision": "approve",
      "confidence": 0.94
    },
    "qc_agent_002": {
      "decision": "approve", 
      "confidence": 0.89
    },
    "predictive_agent_001": {
      "decision": "approve",
      "confidence": 0.90
    }
  },
  "reasoning": "Consensus reached: approve (confidence: 0.91). Votes: 3 approve, 0 reject, 0 abstain."
}
```

### 7. System Monitoring

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "response_time_ms": 15
    },
    "cache": {
      "status": "healthy",
      "hit_rate": 0.85
    },
    "agents": {
      "status": "healthy",
      "active_agents": 5,
      "total_agents": 5
    }
  },
  "metrics": {
    "uptime_seconds": 86400,
    "requests_per_minute": 150,
    "error_rate": 0.001
  }
}
```

#### System Metrics
```http
GET /metrics
```

**Response:**
```
# Prometheus format metrics
agentic_scent_analyses_total{agent_id="qc_agent_001",result="normal"} 1500
agentic_scent_analyses_total{agent_id="qc_agent_001",result="anomaly"} 25
agentic_scent_analysis_duration_seconds_sum{agent_id="qc_agent_001"} 225.5
agentic_scent_analysis_duration_seconds_count{agent_id="qc_agent_001"} 1525
agentic_scent_system_cpu_percent 45.2
agentic_scent_system_memory_percent 62.1
```

### 8. Audit and Compliance

#### Query Audit Events
```http
GET /audit/events?start_time=2024-01-15T00:00:00Z&end_time=2024-01-15T23:59:59Z&event_type=quality_decision
```

**Response:**
```json
{
  "events": [
    {
      "event_id": "uuid",
      "event_type": "quality_decision",
      "timestamp": "2024-01-15T10:30:00Z",
      "user_id": "user123",
      "agent_id": "qc_agent_001",
      "action": "batch_approved",
      "details": {
        "batch_id": "BATCH-001",
        "confidence": 0.92,
        "quality_score": 0.88
      },
      "success": true
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 100
}
```

#### Generate Compliance Report
```http
POST /audit/reports
Content-Type: application/json

{
  "start_time": "2024-01-01T00:00:00Z",
  "end_time": "2024-01-31T23:59:59Z",
  "report_type": "quality",
  "include_events": true
}
```

**Response:**
```json
{
  "report_id": "uuid",
  "generated_at": "2024-01-15T10:30:00Z",
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  },
  "summary": {
    "total_events": 5000,
    "quality_decisions": 150,
    "batches_processed": 75,
    "anomalies_detected": 5,
    "compliance_violations": 0
  },
  "download_url": "/audit/reports/uuid/download"
}
```

## WebSocket API

For real-time updates, the platform provides WebSocket endpoints:

### Real-time Monitoring
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/monitoring');

// Authentication
ws.send(JSON.stringify({
  type: 'auth',
  token: 'bearer_token_here'
}));

// Subscribe to events
ws.send(JSON.stringify({
  type: 'subscribe',
  channels: ['quality_alerts', 'sensor_readings', 'agent_updates']
}));

// Receive real-time updates
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Real-time update:', data);
};
```

## Error Handling

The API uses standard HTTP status codes and provides detailed error information:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid sensor reading data",
    "details": {
      "field": "values",
      "issue": "Array must contain at least 4 numeric values"
    },
    "request_id": "uuid",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Common Status Codes:
- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Rate Limiting

API endpoints are rate-limited to ensure system stability:

- **Authentication**: 5 requests per minute per IP
- **Data Submission**: 100 requests per minute per API key
- **Analytics**: 50 requests per minute per API key
- **General API**: 1000 requests per hour per API key

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
```

## SDK and Client Libraries

Official SDKs are available for:
- Python: `pip install agentic-scent-sdk`
- JavaScript/Node.js: `npm install agentic-scent-sdk`
- Java: Maven/Gradle dependency
- .NET: NuGet package

### Python SDK Example:
```python
from agentic_scent_sdk import AgenticScentClient

client = AgenticScentClient(
    base_url="https://api.agentic-scent.company.com/v1",
    api_key="your_api_key"
)

# Submit sensor reading
reading = client.sensors.submit_reading(
    sensor_id="e_nose_001",
    values=[100.5, 150.2, 200.1, 125.7],
    metadata={"batch_id": "BATCH-001"}
)

# Analyze quality
analysis = client.quality.analyze(reading)
print(f"Quality score: {analysis.quality_score}")
```

This API documentation provides comprehensive coverage of all available endpoints and functionality for integrating with the Agentic Scent Analytics platform.
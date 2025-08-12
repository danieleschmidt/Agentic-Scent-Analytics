# Agentic Scent Analytics - Production Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Docker 20.10+ and Docker Compose v2
- Kubernetes 1.20+ (for K8s deployment)
- Python 3.9+ (for local development)
- 4GB+ RAM, 2+ CPU cores

### One-Command Production Deployment
```bash
# Clone and deploy
git clone https://github.com/terragonlabs/agentic-scent-analytics.git
cd agentic-scent-analytics
./deploy.sh -t docker -e production

# Access services
echo "🌐 Dashboard: http://localhost:80"
echo "📊 Grafana: http://localhost:3000 (admin/admin)"
echo "📈 Prometheus: http://localhost:9090"
```

## 📋 Deployment Options

### 1. Docker Compose (Recommended)
```bash
# Production deployment
docker-compose up -d

# Development deployment
docker-compose --profile dev up -d

# Minimal deployment (core services only)
docker-compose up -d agentic-scent redis postgres
```

### 2. Kubernetes (Enterprise)
```bash
# Deploy to production cluster
kubectl apply -f k8s/
kubectl apply -f deploy/k8s-production.yaml

# Check deployment status
kubectl get pods -n agentic-scent
kubectl logs -f deployment/agentic-scent-app -n agentic-scent
```

### 3. Local Development
```bash
# Install dependencies
pip install -e ".[dev,industrial,llm]"

# Run basic demo
python -m agentic_scent.cli demo --duration 60

# Run comprehensive example
python examples/multi_agent_demo.py
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     NGINX       │    │   Load Balancer  │    │   API Gateway   │
│  Reverse Proxy  │    │   (K8s/Docker)   │    │   (Optional)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Agentic Scent   │    │ Agentic Scent   │    │ Agentic Scent   │
│   App Instance  │    │   App Instance  │    │   App Instance  │
│   (Container)   │    │   (Container)   │    │   (Container)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Redis       │    │   PostgreSQL    │    │   Prometheus    │
│    (Cache)      │    │   (Database)    │    │   (Metrics)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### Environment Variables
```bash
# Core Configuration
AGENTIC_SCENT_ENVIRONMENT=production
AGENTIC_SCENT_SITE_ID=factory_001
AGENTIC_SCENT_LOG_LEVEL=INFO

# Database Configuration  
AGENTIC_SCENT_POSTGRES_URL=postgresql://user:pass@host:5432/db
AGENTIC_SCENT_REDIS_URL=redis://host:6379/0

# LLM Configuration (Optional)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Security Configuration
AGENTIC_SCENT_ENABLE_ENCRYPTION=true
AGENTIC_SCENT_JWT_SECRET=your_jwt_secret_here

# Performance Configuration
AGENTIC_SCENT_MAX_WORKERS=4
AGENTIC_SCENT_CACHE_SIZE_MB=256
AGENTIC_SCENT_TASK_POOL_SIZE=10
```

### Configuration Files
```bash
# Main configuration
config/
├── production.yaml      # Production settings
├── development.yaml     # Development settings
├── sensors.yaml         # Sensor configurations
├── agents.yaml          # Agent configurations
└── compliance.yaml      # Compliance settings
```

## 📊 Monitoring and Observability

### Metrics Collection
- **Prometheus**: System and application metrics
- **Grafana**: Dashboards and visualization
- **Built-in Metrics**: Performance, quality, system health

### Key Metrics
```yaml
# System Metrics
- agentic_scent_sensor_readings_total
- agentic_scent_analysis_duration_seconds
- agentic_scent_anomalies_detected_total
- agentic_scent_agent_health_score

# Business Metrics
- agentic_scent_batch_release_time
- agentic_scent_quality_defect_rate
- agentic_scent_compliance_violations_total
- agentic_scent_cost_savings_total
```

### Health Checks
```bash
# Container health check
docker-compose ps

# Application health check
curl http://localhost:8000/health

# Kubernetes health check
kubectl get pods -l app=agentic-scent
```

## 🔒 Security Configuration

### SSL/TLS Setup
```bash
# Generate self-signed certificate (development)
mkdir -p docker/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/ssl/nginx.key \
  -out docker/ssl/nginx.crt

# Production: Use Let's Encrypt or your certificate authority
```

### Security Best Practices
```yaml
# Enable security features
security:
  enable_encryption: true
  enable_audit_trail: true
  enable_rbac: true
  session_timeout_minutes: 60
  max_failed_attempts: 3
  
# Network security
network:
  enable_tls: true
  cors_origins: ["https://yourdomain.com"]
  trusted_proxies: ["10.0.0.0/8"]
```

## 📈 Scaling and Performance

### Horizontal Scaling (Kubernetes)
```yaml
# Auto-scaling configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agentic-scent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agentic-scent-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Performance Tuning
```bash
# CPU-intensive workloads
export AGENTIC_SCENT_MAX_WORKERS=8
export AGENTIC_SCENT_TASK_POOL_SIZE=16

# Memory-optimized
export AGENTIC_SCENT_CACHE_SIZE_MB=512
export AGENTIC_SCENT_ENABLE_COMPRESSION=true

# High-throughput
export AGENTIC_SCENT_BATCH_SIZE=100
export AGENTIC_SCENT_ASYNC_WORKERS=4
```

## 🔄 CI/CD Pipeline

### GitHub Actions (Example)
```yaml
name: Deploy to Production
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and Test
      run: |
        docker build -t agentic-scent:${{ github.sha }} .
        docker run --rm agentic-scent:${{ github.sha }} python -m pytest
        
    - name: Deploy to Production
      run: |
        docker tag agentic-scent:${{ github.sha }} agentic-scent:latest
        docker-compose up -d --build
```

### GitLab CI/CD (Example)
```yaml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  script:
    - python -m pytest tests/
    - python test_security.py

build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

deploy_production:
  stage: deploy
  script:
    - kubectl set image deployment/agentic-scent-app app=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check logs
docker-compose logs agentic-scent
kubectl logs deployment/agentic-scent-app -n agentic-scent

# Check dependencies
docker-compose ps
kubectl get pods -n agentic-scent
```

#### 2. Performance Issues
```bash
# Check resource usage
docker stats
kubectl top pods -n agentic-scent

# Monitor metrics
curl http://localhost:8090/metrics
```

#### 3. Database Connection Issues
```bash
# Test database connectivity
docker-compose exec postgres psql -U agentic -d agentic_scent -c "SELECT 1;"

# Check connection pool
curl http://localhost:8000/health/database
```

#### 4. Memory Issues
```bash
# Reduce cache size
export AGENTIC_SCENT_CACHE_SIZE_MB=128

# Enable compression
export AGENTIC_SCENT_ENABLE_COMPRESSION=true

# Limit worker processes
export AGENTIC_SCENT_MAX_WORKERS=2
```

### Debug Mode
```bash
# Enable debug logging
export AGENTIC_SCENT_LOG_LEVEL=DEBUG

# Enable profiling
export AGENTIC_SCENT_ENABLE_PROFILING=true

# Run single agent for debugging
python -m agentic_scent.cli debug-agent quality_control
```

## 📚 Additional Resources

### Documentation
- [API Documentation](docs/API.md)
- [Architecture Guide](docs/ARCHITECTURE.md)
- [Security Guide](docs/SECURITY.md)
- [Development Guide](docs/DEVELOPMENT.md)

### Support
- **Issues**: https://github.com/terragonlabs/agentic-scent-analytics/issues
- **Discussions**: https://github.com/terragonlabs/agentic-scent-analytics/discussions
- **Security**: security@terragonlabs.com

### Examples
```bash
# Basic usage
python examples/basic_usage.py

# Multi-agent coordination
python examples/multi_agent_demo.py

# Custom sensor integration
python examples/custom_sensor_integration.py

# Advanced analytics
python examples/advanced_analytics.py
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] System requirements met (RAM, CPU, Storage)
- [ ] Docker and Docker Compose installed
- [ ] Environment variables configured
- [ ] SSL certificates prepared (production)
- [ ] Database initialized
- [ ] Backup strategy implemented

### Post-Deployment
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Alerts configured  
- [ ] Performance benchmarks met
- [ ] Security scan passed
- [ ] Documentation updated
- [ ] Team trained on operations

### Production Readiness
- [ ] Load testing completed
- [ ] Disaster recovery tested
- [ ] Monitoring dashboards configured
- [ ] Log aggregation setup
- [ ] Security hardening applied
- [ ] Compliance requirements met
- [ ] Change management process defined

---

**Note**: This deployment guide covers the essentials for getting Agentic Scent Analytics running in production. For enterprise deployments, consider additional features like multi-region setup, advanced security, and custom integrations.

🎉 **Success!** Your Agentic Scent Analytics platform is now production-ready with enterprise-grade scalability, security, and monitoring.
# Agentic Scent Analytics - Production Deployment Guide

## Overview

This guide covers the production deployment of the Agentic Scent Analytics system, a high-performance industrial AI platform for smart factory e-nose deployments.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                        │
│                     (NGINX)                             │
└─────────────────┬───────────────────────────────────────┘
                  │
        ┌─────────▼─────────┐
        │   Application     │
        │    Cluster        │
        │  (3+ instances)   │
        └─────────┬─────────┘
                  │
    ┌─────────────▼─────────────┐
    │     Data Layer            │
    │  ┌─────────┬─────────┐   │
    │  │PostgreSQL│ Redis  │   │
    │  │Database  │ Cache  │   │
    │  └─────────┴─────────┘   │
    └───────────────────────────┘

    ┌─────────────────────────────┐
    │      Monitoring             │
    │  ┌─────────┬─────────┐     │
    │  │Prometheus│Grafana │     │
    │  │Metrics   │Dashboard│     │
    │  └─────────┴─────────┘     │
    └─────────────────────────────┘
```

## Deployment Options

### 1. Docker Compose (Recommended for smaller deployments)
### 2. Kubernetes (Recommended for production at scale)

## Prerequisites

### System Requirements

- **CPU**: 8+ cores recommended
- **RAM**: 16GB+ recommended  
- **Storage**: 100GB+ SSD storage
- **Network**: Stable internet connection for LLM API calls

### Software Requirements

#### For Docker Deployment:
- Docker Engine 20.10+
- Docker Compose 2.0+
- Git

#### For Kubernetes Deployment:
- kubectl 1.25+
- Helm 3.8+ (optional)
- Access to Kubernetes cluster (1.25+)

## Configuration

### Environment Variables

Create and configure environment files in the `deploy/` directory:

#### Required Environment Variables:

```bash
# Application
APP_ENV=production
APP_NAME=agentic-scent-analytics
APP_VERSION=1.0.0

# Database
DATABASE_URL=postgresql://user:password@host:5432/database
DB_POOL_SIZE=20

# Cache
REDIS_URL=redis://host:6379/0

# LLM APIs
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Security
SECRET_KEY=your_super_secret_key_here
ALLOWED_HOSTS=yourdomain.com,127.0.0.1

# Performance
TASK_POOL_MIN_WORKERS=4
TASK_POOL_MAX_WORKERS=16
CACHE_MEMORY_SIZE=2000
```

### Security Configuration

1. **Generate Strong Secrets**:
   ```bash
   # Generate SECRET_KEY
   python -c "import secrets; print(secrets.token_urlsafe(50))"
   
   # Generate database password
   openssl rand -base64 32
   ```

2. **SSL/TLS Configuration**:
   - Obtain SSL certificates from Let's Encrypt or your certificate authority
   - Place certificates in appropriate volumes/secrets

3. **Network Security**:
   - Configure firewall rules
   - Use VPN for database access
   - Enable network policies in Kubernetes

## Docker Deployment

### Quick Start

1. **Clone and prepare**:
   ```bash
   git clone https://github.com/your-org/agentic-scent-analytics.git
   cd agentic-scent-analytics
   ```

2. **Configure environment**:
   ```bash
   cp deploy/production.env.example deploy/production.env
   # Edit deploy/production.env with your configuration
   ```

3. **Deploy**:
   ```bash
   ./scripts/deploy.sh docker production
   ```

### Manual Docker Deployment

1. **Build the application**:
   ```bash
   docker build -t agentic-scent-analytics:latest .
   ```

2. **Start services**:
   ```bash
   cd deploy
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Verify deployment**:
   ```bash
   curl http://localhost:8000/health
   ```

### Docker Services

After deployment, the following services will be available:

- **Application**: http://localhost:8000
- **Grafana Dashboard**: http://localhost:3000
- **Prometheus Metrics**: http://localhost:9090
- **Database**: localhost:5432 (internal)
- **Redis Cache**: localhost:6379 (internal)

## Kubernetes Deployment

### Prerequisites

1. **Kubernetes cluster** with:
   - CNI plugin installed
   - Ingress controller (NGINX recommended)
   - StorageClass for persistent volumes

2. **Configure kubectl**:
   ```bash
   kubectl config use-context your-production-context
   ```

### Deployment Steps

1. **Prepare secrets**:
   ```bash
   # Create namespace
   kubectl create namespace agentic-scent-prod
   
   # Create secrets
   kubectl create secret generic agentic-scent-secrets \
     --from-literal=DB_PASSWORD="your_db_password" \
     --from-literal=OPENAI_API_KEY="your_openai_key" \
     --from-literal=ANTHROPIC_API_KEY="your_anthropic_key" \
     --from-literal=SECRET_KEY="your_secret_key" \
     --namespace=agentic-scent-prod
   ```

2. **Deploy**:
   ```bash
   ./scripts/deploy.sh kubernetes production
   ```

3. **Or manual deployment**:
   ```bash
   kubectl apply -f deploy/k8s-production.yaml
   ```

### Kubernetes Features

- **Auto-scaling**: HPA configured for 3-10 replicas based on CPU/memory
- **Health checks**: Liveness and readiness probes
- **Persistent storage**: Database and logs persistence
- **Network policies**: Secure pod communication
- **Resource limits**: CPU and memory limits configured

### Monitoring Deployment

```bash
# Watch pods
kubectl get pods -n agentic-scent-prod -w

# Check logs
kubectl logs -f deployment/agentic-scent-app -n agentic-scent-prod

# Check ingress
kubectl get ingress -n agentic-scent-prod
```

## Post-Deployment Configuration

### 1. Database Initialization

The system will automatically initialize the database on first start. To manually initialize:

```bash
# Docker
docker-compose exec app python -m agentic_scent.cli init-db

# Kubernetes  
kubectl exec -it deployment/agentic-scent-app -n agentic-scent-prod -- \
  python -m agentic_scent.cli init-db
```

### 2. Configure Monitoring

1. **Access Grafana** (default admin/admin):
   - Docker: http://localhost:3000
   - Kubernetes: Configure ingress or port-forward

2. **Import dashboards**:
   - Import provided dashboard templates
   - Configure alerting rules

3. **Set up log aggregation** (if using Loki):
   - Configure log shipping
   - Set up log retention policies

### 3. SSL/TLS Setup

#### Docker with NGINX:
1. Place certificates in `ssl_certs` volume
2. Update `docker/nginx.conf` with certificate paths
3. Restart NGINX service

#### Kubernetes:
1. Install cert-manager:
   ```bash
   kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.11.0/cert-manager.yaml
   ```

2. Configure Let's Encrypt issuer:
   ```yaml
   apiVersion: cert-manager.io/v1
   kind: ClusterIssuer
   metadata:
     name: letsencrypt-prod
   spec:
     acme:
       server: https://acme-v02.api.letsencrypt.org/directory
       email: your-email@domain.com
       privateKeySecretRef:
         name: letsencrypt-prod
       solvers:
       - http01:
           ingress:
             class: nginx
   ```

## Health Checks and Monitoring

### Health Endpoints

- **Health Check**: `GET /health`
- **Readiness Check**: `GET /ready`
- **Metrics**: `GET /metrics` (Prometheus format)

### Key Metrics to Monitor

1. **Application Metrics**:
   - Response time (target: <200ms)
   - Error rate (target: <1%)
   - Throughput (requests/second)

2. **System Metrics**:
   - CPU utilization (target: <70%)
   - Memory usage (target: <80%)
   - Disk usage (target: <85%)

3. **Business Metrics**:
   - Sensor readings processed/minute
   - Anomaly detection rate
   - LLM API response times

### Alerting Rules

Configure alerts for:
- High error rate (>5%)
- High response time (>500ms)
- Database connection failures
- LLM API failures
- Memory/CPU saturation

## Backup and Recovery

### Database Backups

#### Automated Backups (Docker):
```bash
# Backup service is included in docker-compose
docker-compose exec backup /backup.sh
```

#### Manual Backup:
```bash
# Docker
docker-compose exec postgres pg_dump -U agentic_user agentic_scent_prod > backup.sql

# Kubernetes
kubectl exec postgres-pod-name -n agentic-scent-prod -- \
  pg_dump -U agentic_user agentic_scent_prod > backup.sql
```

### Configuration Backups

```bash
# Kubernetes configurations
kubectl get all,configmap,secret,pvc,ingress -n agentic-scent-prod -o yaml > k8s-backup.yaml

# Docker configurations
cp -r deploy/ backup-configs/
```

### Recovery Procedures

1. **Database Recovery**:
   ```bash
   # Restore from backup
   psql -U agentic_user -d agentic_scent_prod < backup.sql
   ```

2. **Full System Recovery**:
   ```bash
   # Redeploy from backup configurations
   kubectl apply -f k8s-backup.yaml
   ```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**:
   ```bash
   # Check database status
   kubectl get pods -l app=postgres -n agentic-scent-prod
   
   # Check logs
   kubectl logs postgres-pod-name -n agentic-scent-prod
   ```

2. **High Memory Usage**:
   - Check cache configuration
   - Monitor task pool workers
   - Review memory limits

3. **LLM API Failures**:
   - Verify API keys
   - Check rate limits
   - Monitor network connectivity

4. **Performance Issues**:
   - Scale up replicas
   - Check resource limits
   - Optimize queries

### Log Locations

#### Docker:
- Application: `/var/lib/docker/volumes/agentic-scent_app_logs/_data/`
- NGINX: `/var/lib/docker/volumes/agentic-scent_nginx_logs/_data/`

#### Kubernetes:
```bash
# Application logs
kubectl logs -f deployment/agentic-scent-app -n agentic-scent-prod

# Database logs  
kubectl logs -f deployment/postgres -n agentic-scent-prod
```

## Scaling

### Horizontal Scaling

#### Docker:
```bash
docker-compose up -d --scale app=3
```

#### Kubernetes:
```bash
kubectl scale deployment agentic-scent-app --replicas=5 -n agentic-scent-prod
```

### Vertical Scaling

Update resource limits in:
- `docker-compose.prod.yml` for Docker
- `k8s-production.yaml` for Kubernetes

### Auto-scaling (Kubernetes)

HPA is pre-configured to scale based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)
- Custom metrics (optional)

## Security

### Security Checklist

- [ ] Strong passwords and API keys configured
- [ ] SSL/TLS certificates installed
- [ ] Network policies configured (Kubernetes)
- [ ] Database access restricted
- [ ] Regular security updates applied
- [ ] Audit logging enabled
- [ ] Input validation configured
- [ ] Rate limiting enabled

### Security Monitoring

Monitor for:
- Failed authentication attempts
- Unusual API usage patterns
- Database access anomalies
- SSL certificate expiration

## Maintenance

### Regular Maintenance Tasks

1. **Weekly**:
   - Review system metrics
   - Check error logs
   - Verify backups

2. **Monthly**:
   - Update dependencies
   - Review security patches
   - Optimize database

3. **Quarterly**:
   - Performance tuning
   - Capacity planning
   - Security audit

### Update Procedures

1. **Build new version**:
   ```bash
   ./scripts/deploy.sh docker production v1.1.0
   ```

2. **Rolling update (Kubernetes)**:
   ```bash
   kubectl set image deployment/agentic-scent-app \
     app=agentic-scent-analytics:v1.1.0 \
     -n agentic-scent-prod
   ```

## Support

For production support:

1. **Monitor health endpoints**
2. **Check application logs** 
3. **Review system metrics**
4. **Consult troubleshooting guide**
5. **Contact support** with:
   - Error logs
   - System metrics
   - Configuration details
   - Steps to reproduce

## Performance Benchmarks

Expected performance metrics:

- **Throughput**: 1000+ sensor readings/minute
- **Latency**: <100ms analysis response
- **Availability**: 99.9% uptime
- **Scalability**: 10+ concurrent production lines
- **Recovery**: <5 minute RTO, <1 hour RPO

---

For more information, see:
- [API Documentation](./docs/API.md)
- [Architecture Guide](./docs/ARCHITECTURE.md)
- [Configuration Reference](./docs/CONFIGURATION.md)
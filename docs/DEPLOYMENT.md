# Agentic Scent Analytics - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying Agentic Scent Analytics in various environments, from development to production. The platform supports multiple deployment methods including Docker Compose, Kubernetes, and manual installation.

## Prerequisites

### System Requirements

#### Minimum Requirements (Development):
- **CPU**: 2 cores
- **Memory**: 4 GB RAM
- **Storage**: 20 GB available space
- **Network**: Reliable internet connection

#### Recommended Requirements (Production):
- **CPU**: 8 cores
- **Memory**: 16 GB RAM
- **Storage**: 100 GB SSD (with backup strategy)
- **Network**: High-speed, redundant network connectivity

### Software Prerequisites

#### Core Requirements:
- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+)
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **Python**: 3.9+ (for manual installation)

#### Optional (Kubernetes Deployment):
- **Kubernetes**: Version 1.20+
- **kubectl**: Compatible with cluster version
- **Helm**: Version 3.0+ (optional)

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/terragonlabs/agentic-scent-analytics.git
cd agentic-scent-analytics
```

### 2. Run Basic Tests
```bash
python3 run_basic_tests.py
```

### 3. Deploy with Docker Compose
```bash
# Production deployment
./deploy.sh -t docker -e production

# Development deployment  
./deploy.sh -t docker -e development
```

## Docker Deployment

### Development Environment

#### 1. Start Development Environment
```bash
# Using deployment script
./deploy.sh -t docker -e development

# Or manually with docker-compose
docker-compose --profile dev up -d
```

#### 2. Access Services
- **Application**: http://localhost:8001
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090

#### 3. View Logs
```bash
docker-compose logs -f agentic-scent-dev
```

### Production Environment

#### 1. Prepare Configuration
```bash
# Create directories
mkdir -p data logs config

# Copy production configuration
cp docker/production.env .env

# Edit configuration as needed
nano .env
```

#### 2. Deploy Production Stack
```bash
./deploy.sh -t docker -e production
```

#### 3. Verify Deployment
```bash
# Check service health
docker-compose ps

# Check application logs
docker-compose logs agentic-scent

# Test API endpoint
curl http://localhost/health
```

#### 4. SSL/TLS Configuration (Production)
```bash
# Generate SSL certificates (example with Let's Encrypt)
certbot certonly --standalone -d agentic-scent.company.com

# Update nginx configuration
cp docker/ssl/cert.pem docker/ssl/
cp docker/ssl/private.key docker/ssl/

# Restart nginx
docker-compose restart nginx
```

### Docker Configuration Options

#### Environment Variables
```bash
# Core configuration
AGENTIC_SCENT_ENVIRONMENT=production
AGENTIC_SCENT_SITE_ID=production_site
AGENTIC_SCENT_LOG_LEVEL=INFO

# Database connections
AGENTIC_SCENT_REDIS_URL=redis://redis:6379
AGENTIC_SCENT_POSTGRES_URL=postgresql://user:pass@postgres:5432/db

# Security settings
AGENTIC_SCENT_ENABLE_ENCRYPTION=true
AGENTIC_SCENT_REQUIRE_AUTHENTICATION=true

# Performance settings
AGENTIC_SCENT_MAX_CONCURRENT=100
AGENTIC_SCENT_CACHE_SIZE_MB=512
```

#### Volume Configuration
```yaml
volumes:
  # Persistent data
  - ./data:/app/data
  - ./logs:/app/logs
  - ./config:/app/config
  
  # Database storage
  - postgres_data:/var/lib/postgresql/data
  - redis_data:/data
```

## Kubernetes Deployment

### Preparation

#### 1. Create Namespace
```bash
kubectl apply -f k8s/namespace.yaml
```

#### 2. Configure Secrets
```bash
# Create database password secret
kubectl create secret generic agentic-scent-secrets \
  --from-literal=DATABASE_PASSWORD=secure_password \
  --from-literal=REDIS_PASSWORD=redis_password \
  --from-literal=ENCRYPTION_KEY=32-byte-encryption-key \
  -n agentic-scent

# Or apply the secret file (update values first)
kubectl apply -f k8s/configmap.yaml
```

#### 3. Configure Storage
```bash
# Update storage class in PVC files
sed -i 's/fast-ssd/your-storage-class/g' k8s/*.yaml
```

### Deployment Steps

#### 1. Deploy Database Layer
```bash
# Deploy PostgreSQL
kubectl apply -f k8s/postgres.yaml

# Deploy Redis
kubectl apply -f k8s/redis.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgresql -n agentic-scent --timeout=300s
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=redis -n agentic-scent --timeout=300s
```

#### 2. Deploy Application
```bash
# Build and push image (if using private registry)
docker build -t your-registry.com/agentic-scent:v1.0.0 .
docker push your-registry.com/agentic-scent:v1.0.0

# Update deployment with correct image
sed -i 's|image: agentic-scent:latest|image: your-registry.com/agentic-scent:v1.0.0|g' k8s/deployment.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Wait for deployment
kubectl wait --for=condition=available deployment/agentic-scent-app -n agentic-scent --timeout=300s
```

#### 3. Configure Ingress
```bash
# Update ingress with your domain
sed -i 's/agentic-scent.company.com/your-domain.com/g' k8s/service.yaml

# Apply ingress
kubectl apply -f k8s/service.yaml
```

#### 4. Verify Deployment
```bash
# Check pod status
kubectl get pods -n agentic-scent

# Check services
kubectl get services -n agentic-scent

# Check ingress
kubectl get ingress -n agentic-scent

# View logs
kubectl logs -f deployment/agentic-scent-app -n agentic-scent
```

### Kubernetes Operations

#### Scaling
```bash
# Scale application pods
kubectl scale deployment agentic-scent-app --replicas=5 -n agentic-scent

# Configure horizontal pod autoscaling
kubectl autoscale deployment agentic-scent-app --cpu-percent=70 --min=3 --max=10 -n agentic-scent
```

#### Updates
```bash
# Rolling update
kubectl set image deployment/agentic-scent-app agentic-scent=your-registry.com/agentic-scent:v1.1.0 -n agentic-scent

# Check rollout status
kubectl rollout status deployment/agentic-scent-app -n agentic-scent

# Rollback if needed
kubectl rollout undo deployment/agentic-scent-app -n agentic-scent
```

#### Monitoring
```bash
# Check resource usage
kubectl top pods -n agentic-scent
kubectl top nodes

# Check events
kubectl get events -n agentic-scent --sort-by='.lastTimestamp'
```

## Manual Installation

For environments where containerization is not possible:

### 1. System Preparation
```bash
# Install Python 3.9+
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev

# Install system dependencies
sudo apt install gcc g++ git curl postgresql redis-server nginx

# Create application user
sudo useradd -m -s /bin/bash agentic
sudo usermod -aG sudo agentic
```

### 2. Application Setup
```bash
# Switch to application user
sudo su - agentic

# Clone repository
git clone https://github.com/terragonlabs/agentic-scent-analytics.git
cd agentic-scent-analytics

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Database Setup
```bash
# Configure PostgreSQL
sudo -u postgres createuser -P agentic
sudo -u postgres createdb -O agentic agentic_scent

# Initialize database
psql -U agentic -d agentic_scent -f docker/init-db.sql
```

### 4. Service Configuration
```bash
# Create systemd service
sudo tee /etc/systemd/system/agentic-scent.service << EOF
[Unit]
Description=Agentic Scent Analytics
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=agentic
WorkingDirectory=/home/agentic/agentic-scent-analytics
Environment=PATH=/home/agentic/agentic-scent-analytics/venv/bin
ExecStart=/home/agentic/agentic-scent-analytics/venv/bin/python -m agentic_scent.cli demo --duration 86400
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl daemon-reload
sudo systemctl enable agentic-scent
sudo systemctl start agentic-scent
```

### 5. NGINX Configuration
```bash
# Copy nginx configuration
sudo cp docker/nginx.conf /etc/nginx/sites-available/agentic-scent
sudo ln -s /etc/nginx/sites-available/agentic-scent /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Test and restart nginx
sudo nginx -t
sudo systemctl restart nginx
```

## Configuration Management

### Environment-Specific Configuration

#### Development (docker/.env.development)
```bash
AGENTIC_SCENT_ENVIRONMENT=development
AGENTIC_SCENT_LOG_LEVEL=DEBUG
AGENTIC_SCENT_ENABLE_ENCRYPTION=false
AGENTIC_SCENT_REQUIRE_AUTHENTICATION=false
```

#### Staging (docker/.env.staging)
```bash
AGENTIC_SCENT_ENVIRONMENT=staging
AGENTIC_SCENT_LOG_LEVEL=INFO
AGENTIC_SCENT_ENABLE_ENCRYPTION=true
AGENTIC_SCENT_REQUIRE_AUTHENTICATION=true
```

#### Production (docker/.env.production)
```bash
AGENTIC_SCENT_ENVIRONMENT=production
AGENTIC_SCENT_LOG_LEVEL=WARNING
AGENTIC_SCENT_ENABLE_ENCRYPTION=true
AGENTIC_SCENT_REQUIRE_AUTHENTICATION=true
AGENTIC_SCENT_MAX_CONCURRENT=200
AGENTIC_SCENT_CACHE_SIZE_MB=1024
```

### External System Integration

#### MES Integration
```bash
# Configure in environment file
AGENTIC_SCENT_MES_ENABLED=true
AGENTIC_SCENT_MES_ENDPOINT=https://mes.company.com/api
AGENTIC_SCENT_MES_API_KEY=your_mes_api_key
```

#### SCADA Integration
```bash
AGENTIC_SCENT_SCADA_ENABLED=true
AGENTIC_SCENT_SCADA_PROTOCOL=OPC_UA
AGENTIC_SCENT_SCADA_ENDPOINT=opc.tcp://scada.company.local:4840
```

## Monitoring and Alerting

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'agentic-scent'
    static_configs:
      - targets: ['agentic-scent:8090']
    scrape_interval: 10s
```

### Grafana Dashboards
```bash
# Import pre-built dashboards
curl -s https://raw.githubusercontent.com/terragonlabs/agentic-scent-analytics/main/docker/grafana/dashboards/main-dashboard.json \
  | curl -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $GRAFANA_API_KEY" \
    -d @- \
    http://localhost:3000/api/dashboards/db
```

### Alert Configuration
```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@company.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'admin@company.com'
    subject: 'Agentic Scent Analytics Alert'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
```

## Security Hardening

### SSL/TLS Configuration
```bash
# Generate strong SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
  -keyout docker/ssl/key.pem \
  -out docker/ssl/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Company/CN=agentic-scent.company.com"

# Set proper permissions
chmod 600 docker/ssl/key.pem
chmod 644 docker/ssl/cert.pem
```

### Database Security
```sql
-- Create restricted database user
CREATE USER agentic_readonly WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE agentic_scent TO agentic_readonly;
GRANT USAGE ON SCHEMA public TO agentic_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO agentic_readonly;

-- Enable row-level security
ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY audit_user_policy ON audit_events
  FOR SELECT USING (user_id = current_user);
```

### Network Security
```bash
# Configure firewall (UFW example)
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw deny 5432/tcp     # PostgreSQL (internal only)
sudo ufw deny 6379/tcp     # Redis (internal only)
sudo ufw enable
```

## Backup and Recovery

### Database Backup
```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/var/backups/agentic-scent"
DATE=$(date +%Y%m%d_%H%M%S)

# PostgreSQL backup
pg_dump -U agentic -h localhost agentic_scent | gzip > $BACKUP_DIR/postgres_$DATE.sql.gz

# Redis backup
redis-cli --rdb $BACKUP_DIR/redis_$DATE.rdb

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "postgres_*.sql.gz" -mtime +30 -delete
find $BACKUP_DIR -name "redis_*.rdb" -mtime +30 -delete
```

### Application Data Backup
```bash
#!/bin/bash
# backup-data.sh
tar -czf /var/backups/agentic-scent/data_$(date +%Y%m%d_%H%M%S).tar.gz \
  /app/data \
  /app/config \
  /app/logs
```

### Automated Backup (Cron)
```bash
# Add to crontab
0 2 * * * /opt/agentic-scent/backup.sh
0 3 * * 0 /opt/agentic-scent/backup-data.sh
```

### Recovery Procedures
```bash
# PostgreSQL restore
gunzip -c postgres_20240115_020000.sql.gz | psql -U agentic -d agentic_scent

# Redis restore
redis-cli shutdown
cp redis_20240115_020000.rdb /var/lib/redis/dump.rdb
systemctl start redis

# Application data restore
tar -xzf data_20240115_020000.tar.gz -C /
```

## Troubleshooting

### Common Issues

#### 1. Container Startup Failures
```bash
# Check container logs
docker-compose logs agentic-scent

# Check resource usage
docker stats

# Verify network connectivity
docker network ls
docker network inspect agentic-network
```

#### 2. Database Connection Issues
```bash
# Test database connectivity
docker-compose exec agentic-scent python -c "
import psycopg2
conn = psycopg2.connect(
    host='postgres',
    database='agentic_scent', 
    user='agentic',
    password='agentic_password'
)
print('Database connection successful')
"
```

#### 3. Performance Issues
```bash
# Check system resources
docker stats
kubectl top pods -n agentic-scent

# Analyze slow queries
docker-compose exec postgres psql -U agentic -d agentic_scent -c "
SELECT query, calls, mean_time, total_time 
FROM pg_stat_statements 
ORDER BY total_time DESC LIMIT 10;
"
```

#### 4. API Issues
```bash
# Test API endpoints
curl -H "Content-Type: application/json" \
     -X GET http://localhost/health

# Check API logs
docker-compose logs agentic-scent | grep ERROR
```

### Log Analysis
```bash
# Application logs
tail -f logs/agentic-scent.log

# Database logs
docker-compose logs postgres

# System logs
journalctl -u agentic-scent -f
```

### Performance Tuning
```bash
# PostgreSQL tuning
echo "shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB" >> postgresql.conf

# Redis tuning
echo "maxmemory 512mb
maxmemory-policy allkeys-lru" >> redis.conf
```

## Maintenance Procedures

### Regular Maintenance Tasks

#### Daily:
- Monitor system health and performance
- Check application logs for errors
- Verify backup completion

#### Weekly:
- Update system packages
- Review security logs
- Check disk space usage

#### Monthly:
- Update application dependencies
- Review and optimize database performance
- Test disaster recovery procedures

### Maintenance Windows
```bash
# Planned maintenance script
#!/bin/bash
echo "Starting maintenance window"

# Stop application
docker-compose stop agentic-scent

# Backup data
./backup.sh

# Update system
sudo apt update && sudo apt upgrade -y

# Update application
git pull origin main
docker-compose build --no-cache

# Restart services
docker-compose up -d

# Verify health
sleep 30
curl -f http://localhost/health || echo "Health check failed"

echo "Maintenance completed"
```

This deployment guide provides comprehensive instructions for deploying Agentic Scent Analytics in various environments with proper security, monitoring, and maintenance procedures.
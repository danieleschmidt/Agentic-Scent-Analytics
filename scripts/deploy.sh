#!/bin/bash
set -euo pipefail

# Production deployment script for Agentic Scent Analytics
# Usage: ./scripts/deploy.sh [docker|kubernetes] [environment] [version]

DEPLOYMENT_TYPE=${1:-docker}
ENVIRONMENT=${2:-production}
VERSION=${3:-latest}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_DIR="$ROOT_DIR/deploy"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        if ! command -v docker &> /dev/null; then
            error "Docker is not installed or not in PATH"
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            error "Docker Compose is not installed or not in PATH"
        fi
        
        # Check if Docker daemon is running
        if ! docker info &> /dev/null; then
            error "Docker daemon is not running"
        fi
        
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        if ! command -v kubectl &> /dev/null; then
            error "kubectl is not installed or not in PATH"
        fi
        
        if ! command -v helm &> /dev/null; then
            warning "Helm is not installed - some features may not work"
        fi
        
        # Check kubectl connectivity
        if ! kubectl version --client &> /dev/null; then
            error "kubectl is not properly configured"
        fi
        
    else
        error "Invalid deployment type: $DEPLOYMENT_TYPE. Use 'docker' or 'kubernetes'"
    fi
    
    success "Prerequisites check passed"
}

# Validate environment files
validate_environment() {
    log "Validating environment configuration..."
    
    local env_file="$DEPLOY_DIR/${ENVIRONMENT}.env"
    
    if [[ ! -f "$env_file" ]]; then
        error "Environment file not found: $env_file"
    fi
    
    # Check for required environment variables
    local required_vars=(
        "APP_NAME"
        "APP_VERSION"
        "DATABASE_URL"
        "REDIS_URL"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" "$env_file" && [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        error "Missing required environment variables: ${missing_vars[*]}"
    fi
    
    success "Environment validation passed"
}

# Build application
build_application() {
    log "Building application..."
    
    cd "$ROOT_DIR"
    
    # Build Docker image
    docker build \
        --target production \
        --tag "agentic-scent-analytics:${VERSION}" \
        --tag "agentic-scent-analytics:latest" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        --build-arg VERSION="$VERSION" \
        .
    
    success "Application build completed"
}

# Deploy with Docker Compose
deploy_docker() {
    log "Deploying with Docker Compose..."
    
    cd "$DEPLOY_DIR"
    
    # Set environment variables
    export APP_VERSION="$VERSION"
    export COMPOSE_PROJECT_NAME="agentic-scent"
    
    # Pull external images
    docker-compose -f docker-compose.prod.yml pull postgres redis nginx prometheus grafana
    
    # Deploy stack
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if docker-compose -f docker-compose.prod.yml exec -T app curl -f http://localhost:8000/health &>/dev/null; then
            break
        fi
        
        ((attempt++))
        log "Waiting for application to be ready... (attempt $attempt/$max_attempts)"
        sleep 10
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        error "Application failed to become ready within expected time"
    fi
    
    success "Docker deployment completed successfully"
    
    # Display service URLs
    log "Service URLs:"
    echo "  Application: http://localhost:8000"
    echo "  Grafana: http://localhost:3000"
    echo "  Prometheus: http://localhost:9090"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log "Deploying to Kubernetes..."
    
    cd "$DEPLOY_DIR"
    
    # Apply namespace first
    kubectl apply -f k8s-production.yaml
    
    # Wait for namespace to be ready
    kubectl wait --for=condition=Ready namespace/agentic-scent-prod --timeout=60s
    
    # Create or update image pull secret if needed
    if [[ -n "${DOCKER_REGISTRY_SECRET:-}" ]]; then
        kubectl create secret docker-registry regcred \
            --docker-server="${DOCKER_REGISTRY:-}" \
            --docker-username="${DOCKER_USERNAME:-}" \
            --docker-password="${DOCKER_PASSWORD:-}" \
            --namespace=agentic-scent-prod \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
    
    # Update image tag in deployment
    kubectl set image deployment/agentic-scent-app \
        app="agentic-scent-analytics:${VERSION}" \
        --namespace=agentic-scent-prod
    
    # Wait for rollout to complete
    kubectl rollout status deployment/agentic-scent-app \
        --namespace=agentic-scent-prod \
        --timeout=300s
    
    success "Kubernetes deployment completed successfully"
    
    # Display service information
    log "Service Information:"
    kubectl get services -n agentic-scent-prod
    
    log "Pod Status:"
    kubectl get pods -n agentic-scent-prod
    
    # Get ingress URL if available
    local ingress_url
    ingress_url=$(kubectl get ingress agentic-scent-ingress \
        -n agentic-scent-prod \
        -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "")
    
    if [[ -n "$ingress_url" ]]; then
        log "Application URL: https://$ingress_url"
    fi
}

# Run post-deployment tests
run_tests() {
    log "Running post-deployment tests..."
    
    cd "$ROOT_DIR"
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        # Test Docker deployment
        local health_url="http://localhost:8000/health"
        
        if curl -f "$health_url" &>/dev/null; then
            success "Health check passed"
        else
            error "Health check failed"
        fi
        
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        # Test Kubernetes deployment
        kubectl run curl-test \
            --image=curlimages/curl \
            --rm -i --tty \
            --namespace=agentic-scent-prod \
            --restart=Never \
            -- curl -f http://agentic-scent-service/health
        
        if [[ $? -eq 0 ]]; then
            success "Kubernetes health check passed"
        else
            error "Kubernetes health check failed"
        fi
    fi
    
    success "Post-deployment tests completed"
}

# Backup current deployment
backup_deployment() {
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        log "Creating backup of Docker deployment..."
        
        # Export database
        docker-compose -f "$DEPLOY_DIR/docker-compose.prod.yml" exec -T postgres \
            pg_dump -U agentic_user agentic_scent_prod > \
            "$DEPLOY_DIR/backup-$(date +%Y%m%d-%H%M%S).sql"
        
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        log "Creating backup of Kubernetes deployment..."
        
        # Export current configuration
        kubectl get all,configmap,secret,pvc,ingress \
            -n agentic-scent-prod \
            -o yaml > "$DEPLOY_DIR/k8s-backup-$(date +%Y%m%d-%H%M%S).yaml"
    fi
    
    success "Backup completed"
}

# Main deployment function
main() {
    log "Starting deployment of Agentic Scent Analytics"
    log "Deployment Type: $DEPLOYMENT_TYPE"
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    
    check_prerequisites
    validate_environment
    
    # Create backup before deployment
    if [[ "${SKIP_BACKUP:-}" != "true" ]]; then
        backup_deployment
    fi
    
    build_application
    
    case "$DEPLOYMENT_TYPE" in
        "docker")
            deploy_docker
            ;;
        "kubernetes")
            deploy_kubernetes
            ;;
    esac
    
    run_tests
    
    success "Deployment completed successfully!"
    log "Monitor the deployment with:"
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        echo "  docker-compose -f $DEPLOY_DIR/docker-compose.prod.yml logs -f"
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        echo "  kubectl logs -f deployment/agentic-scent-app -n agentic-scent-prod"
        echo "  kubectl get pods -n agentic-scent-prod -w"
    fi
}

# Handle script interruption
trap 'error "Deployment interrupted"' INT TERM

# Run main function
main "$@"
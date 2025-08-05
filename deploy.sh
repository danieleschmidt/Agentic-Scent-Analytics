#!/bin/bash

# Agentic Scent Analytics Deployment Script
# Supports Docker Compose and Kubernetes deployments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEPLOYMENT_TYPE="docker"
ENVIRONMENT="production"
NAMESPACE="agentic-scent"
IMAGE_TAG="latest"
REGISTRY=""
BUILD_IMAGE=true
SKIP_TESTS=false

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE         Deployment type: docker, k8s (default: docker)"
    echo "  -e, --environment ENV   Environment: development, staging, production (default: production)"
    echo "  -n, --namespace NS      Kubernetes namespace (default: agentic-scent)"
    echo "  -i, --image-tag TAG     Image tag (default: latest)"
    echo "  -r, --registry REG      Container registry URL"
    echo "  --no-build             Skip building the image"
    echo "  --skip-tests           Skip running tests"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -t docker -e development"
    echo "  $0 -t k8s -e production -i v1.0.0 -r registry.company.com"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -i|--image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        --no-build)
            BUILD_IMAGE=false
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker is required but not installed"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            log_error "Docker Compose is required but not installed"
            exit 1
        fi
    elif [[ "$DEPLOYMENT_TYPE" == "k8s" ]]; then
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl is required but not installed"
            exit 1
        fi
        
        if ! kubectl cluster-info &> /dev/null; then
            log_error "kubectl is not connected to a Kubernetes cluster"
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests as requested"
        return
    fi
    
    log_info "Running tests..."
    
    if [[ -f "run_basic_tests.py" ]]; then
        python3 run_basic_tests.py
        if [[ $? -eq 0 ]]; then
            log_success "Basic tests passed"
        else
            log_error "Basic tests failed"
            exit 1
        fi
    else
        log_warning "No test runner found, skipping tests"
    fi
}

# Build Docker image
build_image() {
    if [[ "$BUILD_IMAGE" == "false" ]]; then
        log_warning "Skipping image build as requested"
        return
    fi
    
    log_info "Building Docker image..."
    
    local image_name="agentic-scent"
    if [[ -n "$REGISTRY" ]]; then
        image_name="${REGISTRY}/agentic-scent"
    fi
    
    local full_image_name="${image_name}:${IMAGE_TAG}"
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        docker build --target development -t "${full_image_name}" .
        docker tag "${full_image_name}" "${image_name}:dev"
    else
        docker build --target production -t "${full_image_name}" .
    fi
    
    log_success "Docker image built: ${full_image_name}"
    
    # Push to registry if specified
    if [[ -n "$REGISTRY" ]]; then
        log_info "Pushing image to registry..."
        docker push "${full_image_name}"
        log_success "Image pushed to registry"
    fi
}

# Deploy using Docker Compose
deploy_docker() {
    log_info "Deploying using Docker Compose..."
    
    # Create necessary directories
    mkdir -p data logs config
    
    # Set environment variables
    export AGENTIC_SCENT_ENVIRONMENT="$ENVIRONMENT"
    export COMPOSE_PROJECT_NAME="agentic-scent"
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        # Development deployment
        docker-compose --profile dev up -d
        log_info "Development environment is starting..."
        log_info "Application will be available at: http://localhost:8001"
        log_info "Grafana dashboard: http://localhost:3000 (admin/admin)"
    else
        # Production deployment
        docker-compose up -d
        log_info "Production environment is starting..."
        log_info "Application will be available at: http://localhost:80"
        log_info "Grafana dashboard: http://localhost:3000 (admin/admin)"
    fi
    
    # Wait for services to be healthy
    log_info "Waiting for services to become healthy..."
    sleep 30
    
    # Check service health
    if docker-compose ps | grep -q "Up (healthy)"; then
        log_success "Services are healthy and running"
    else
        log_warning "Some services may not be fully healthy yet"
        docker-compose ps
    fi
    
    log_success "Docker deployment completed"
}

# Deploy to Kubernetes
deploy_k8s() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl apply -f k8s/namespace.yaml
    
    # Apply configurations
    log_info "Applying configurations..."
    kubectl apply -f k8s/configmap.yaml
    
    # Deploy PostgreSQL
    log_info "Deploying PostgreSQL..."
    kubectl apply -f k8s/postgres.yaml
    
    # Deploy Redis
    log_info "Deploying Redis..."
    kubectl apply -f k8s/redis.yaml
    
    # Deploy application
    log_info "Deploying application..."
    
    # Update image in deployment if registry is specified
    if [[ -n "$REGISTRY" ]]; then
        sed "s|image: agentic-scent:latest|image: ${REGISTRY}/agentic-scent:${IMAGE_TAG}|g" k8s/deployment.yaml | kubectl apply -f -
    else
        kubectl apply -f k8s/deployment.yaml
    fi
    
    # Apply services and ingress
    kubectl apply -f k8s/service.yaml
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/agentic-scent-app -n "$NAMESPACE"
    
    # Get service information
    log_info "Getting service information..."
    kubectl get services -n "$NAMESPACE"
    kubectl get ingress -n "$NAMESPACE"
    
    # Get application URL
    if kubectl get ingress agentic-scent-ingress -n "$NAMESPACE" &> /dev/null; then
        INGRESS_HOST=$(kubectl get ingress agentic-scent-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}')
        if [[ -n "$INGRESS_HOST" ]]; then
            log_info "Application will be available at: https://$INGRESS_HOST"
        fi
    fi
    
    log_success "Kubernetes deployment completed"
}

# Cleanup function
cleanup() {
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        log_info "To stop the deployment, run: docker-compose down"
        log_info "To remove all data, run: docker-compose down -v"
    elif [[ "$DEPLOYMENT_TYPE" == "k8s" ]]; then
        log_info "To delete the deployment, run: kubectl delete namespace $NAMESPACE"
    fi
}

# Main deployment function
main() {
    log_info "Starting Agentic Scent Analytics deployment"
    log_info "Deployment type: $DEPLOYMENT_TYPE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Image tag: $IMAGE_TAG"
    
    check_prerequisites
    run_tests
    build_image
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        deploy_docker
    elif [[ "$DEPLOYMENT_TYPE" == "k8s" ]]; then
        deploy_k8s
    else
        log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
        exit 1
    fi
    
    cleanup
    log_success "Deployment completed successfully!"
}

# Trap signals for cleanup
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main
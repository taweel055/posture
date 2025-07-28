#!/bin/bash
# Docker helper script for FitLifePosture

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Image name
IMAGE_NAME="fitlifeposture"
CONTAINER_NAME="fitlifeposture_app"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[FitLifePosture]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_status "Docker is installed ✓"
}

# Build the Docker image
build_image() {
    print_status "Building FitLifePosture Docker image..."
    docker build -t ${IMAGE_NAME}:latest .
    print_status "Image built successfully ✓"
}

# Build GPU-enabled image
build_gpu_image() {
    print_status "Building FitLifePosture GPU Docker image..."
    docker build -f Dockerfile.gpu -t ${IMAGE_NAME}:gpu .
    print_status "GPU image built successfully ✓"
}

# Run the container
run_container() {
    print_status "Starting FitLifePosture container..."
    
    # Check if container already exists
    if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
        print_warning "Container already exists. Removing old container..."
        docker rm -f ${CONTAINER_NAME}
    fi
    
    # Platform-specific settings
    case "$(uname -s)" in
        Linux*)
            print_status "Running on Linux..."
            xhost +local:docker 2>/dev/null || true
            docker run -d \
                --name ${CONTAINER_NAME} \
                -e DISPLAY=$DISPLAY \
                -e QT_X11_NO_MITSHM=1 \
                -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                -v /dev/video0:/dev/video0 \
                --device /dev/video0 \
                -v $(pwd)/reports:/app/reports \
                -v $(pwd)/data:/app/data \
                -v $(pwd)/logs:/app/logs \
                --network host \
                --cap-add SYS_ADMIN \
                ${IMAGE_NAME}:latest
            ;;
        Darwin*)
            print_status "Running on macOS..."
            print_warning "Camera access in Docker on macOS is limited."
            print_warning "Consider running the app natively for full camera support."
            
            # Check if XQuartz is installed
            if ! command -v xquartz &> /dev/null; then
                print_error "XQuartz is required for GUI support on macOS."
                print_error "Install it from: https://www.xquartz.org/"
                exit 1
            fi
            
            # Allow connections from localhost
            xhost +localhost 2>/dev/null || true
            
            docker run -d \
                --name ${CONTAINER_NAME} \
                -e DISPLAY=host.docker.internal:0 \
                -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                -v $(pwd)/reports:/app/reports \
                -v $(pwd)/data:/app/data \
                -v $(pwd)/logs:/app/logs \
                ${IMAGE_NAME}:latest
            ;;
        *)
            print_error "Unsupported platform: $(uname -s)"
            exit 1
            ;;
    esac
    
    print_status "Container started successfully ✓"
    print_status "View logs with: docker logs -f ${CONTAINER_NAME}"
}

# Run interactive shell in container
run_shell() {
    print_status "Starting interactive shell..."
    
    case "$(uname -s)" in
        Linux*)
            xhost +local:docker 2>/dev/null || true
            docker run -it --rm \
                -e DISPLAY=$DISPLAY \
                -e QT_X11_NO_MITSHM=1 \
                -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                -v /dev/video0:/dev/video0 \
                --device /dev/video0 \
                -v $(pwd)/reports:/app/reports \
                -v $(pwd)/data:/app/data \
                -v $(pwd)/logs:/app/logs \
                --network host \
                --cap-add SYS_ADMIN \
                ${IMAGE_NAME}:latest \
                /bin/bash
            ;;
        Darwin*)
            xhost +localhost 2>/dev/null || true
            docker run -it --rm \
                -e DISPLAY=host.docker.internal:0 \
                -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                -v $(pwd)/reports:/app/reports \
                -v $(pwd)/data:/app/data \
                -v $(pwd)/logs:/app/logs \
                ${IMAGE_NAME}:latest \
                /bin/bash
            ;;
    esac
}

# Stop the container
stop_container() {
    print_status "Stopping FitLifePosture container..."
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
    print_status "Container stopped ✓"
}

# View logs
view_logs() {
    print_status "Viewing container logs..."
    docker logs -f ${CONTAINER_NAME}
}

# Clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
    docker rmi ${IMAGE_NAME}:latest 2>/dev/null || true
    docker rmi ${IMAGE_NAME}:gpu 2>/dev/null || true
    print_status "Cleanup complete ✓"
}

# Test camera access
test_camera() {
    print_status "Testing camera access in Docker..."
    
    case "$(uname -s)" in
        Linux*)
            docker run --rm \
                --device /dev/video0 \
                ${IMAGE_NAME}:latest \
                python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera available:', cap.isOpened()); cap.release()"
            ;;
        Darwin*)
            print_warning "Camera test not available on macOS Docker"
            ;;
    esac
}

# Main menu
show_help() {
    echo "FitLifePosture Docker Manager"
    echo "============================="
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  build-gpu   Build the GPU-enabled Docker image"
    echo "  run         Run the container"
    echo "  shell       Run interactive shell"
    echo "  stop        Stop the container"
    echo "  logs        View container logs"
    echo "  test        Test camera access"
    echo "  cleanup     Remove containers and images"
    echo "  help        Show this help message"
}

# Main script
check_docker

case "${1:-help}" in
    build)
        build_image
        ;;
    build-gpu)
        build_gpu_image
        ;;
    run)
        run_container
        ;;
    shell)
        run_shell
        ;;
    stop)
        stop_container
        ;;
    logs)
        view_logs
        ;;
    test)
        test_camera
        ;;
    cleanup)
        cleanup
        ;;
    help|*)
        show_help
        ;;
esac

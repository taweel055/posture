# FitLifePosture Docker Guide

## 🐳 Overview

FitLifePosture is containerized for easy deployment and consistent environments across different systems. The Docker image includes all necessary dependencies for posture analysis.

## 📋 Prerequisites

1. **Docker**: Install Docker from [docker.com](https://www.docker.com/get-started)
2. **Docker Compose**: Usually included with Docker Desktop
3. **Platform-specific requirements**:
   - **Linux**: X11 server running
   - **macOS**: XQuartz installed from [xquartz.org](https://www.xquartz.org/)
   - **Windows**: WSL2 with X server (VcXsrv or similar)

### GPU Support (Optional)
- NVIDIA GPU with CUDA 11.8+ support
- NVIDIA Container Toolkit installed

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Using Docker Manager Script

```bash
# Make script executable
chmod +x docker-manager.sh

# Build image
./docker-manager.sh build

# Run container
./docker-manager.sh run

# View logs
./docker-manager.sh logs

# Stop container
./docker-manager.sh stop
```

## 🔧 Manual Docker Commands

### Build Image

```bash
# Standard build
docker build -t fitlifeposture:latest .

# GPU-enabled build
docker build -f Dockerfile.gpu -t fitlifeposture:gpu .
```

### Run Container

#### Linux
```bash
# Allow X11 connections
xhost +local:docker

# Run with camera and display
docker run -d \
  --name fitlifeposture_app \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --device /dev/video0 \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/data:/app/data \
  --network host \
  fitlifeposture:latest
```

#### macOS
```bash
# Start XQuartz and allow connections
open -a XQuartz
xhost +localhost

# Run container (limited camera support)
docker run -d \
  --name fitlifeposture_app \
  -e DISPLAY=host.docker.internal:0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/data:/app/data \
  fitlifeposture:latest
```

#### Windows (WSL2)
```bash
# Set DISPLAY variable
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0

# Run container
docker run -d \
  --name fitlifeposture_app \
  -e DISPLAY=$DISPLAY \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/data:/app/data \
  fitlifeposture:latest
```

### GPU-Enabled Container

```bash
# Run with NVIDIA GPU
docker run -d \
  --name fitlifeposture_gpu \
  --runtime=nvidia \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --device /dev/video0 \
  -v $(pwd)/reports:/app/reports \
  --network host \
  fitlifeposture:gpu
```

## 📁 Volume Mounts

The container uses several volumes for persistence:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./reports` | `/app/reports` | Generated analysis reports |
| `./data` | `/app/data` | Application data |
| `./logs` | `/app/logs` | Application logs |
| `/tmp/.X11-unix` | `/tmp/.X11-unix` | X11 socket for GUI |
| `/dev/video0` | `/dev/video0` | Camera device |

## 🎥 Camera Access

### Linux
- Camera devices are directly passed to the container
- Multiple cameras can be added in docker-compose.yml

### macOS
- ⚠️ **Limited Support**: Docker on macOS doesn't have direct camera access
- Consider running the application natively on macOS for full camera functionality
- Alternative: Use a USB camera that can be passed through

### Windows
- Use USB camera passthrough with WSL2
- Or run the container with a network camera stream

## 🖥️ Display Configuration

### Troubleshooting Display Issues

1. **Linux - "Cannot connect to X server"**
   ```bash
   xhost +local:docker
   ```

2. **macOS - No display**
   - Ensure XQuartz is running
   - In XQuartz preferences, enable "Allow connections from network clients"
   - Restart XQuartz after changes

3. **Windows - Display not working**
   - Ensure X server (VcXsrv) is running
   - Disable Windows Firewall for X server
   - Use `-ac` flag when starting VcXsrv

## 🔍 Debugging

### Interactive Shell
```bash
# Enter running container
docker exec -it fitlifeposture_app bash

# Run with shell instead of app
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/video0 \
  fitlifeposture:latest \
  bash
```

### Test Camera Access
```bash
# Inside container
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera available:', cap.isOpened())"
```

### View Logs
```bash
# Container logs
docker logs -f fitlifeposture_app

# Application logs
docker exec fitlifeposture_app cat /app/logs/app.log
```

## 🚨 Common Issues

### 1. Permission Denied
```bash
# Fix permissions
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Camera Not Found
- Check camera device exists: `ls -la /dev/video*`
- Try different device: `--device /dev/video1`
- Check permissions: `sudo chmod 666 /dev/video0`

### 3. GUI Not Displaying
- Verify DISPLAY variable: `echo $DISPLAY`
- Check X11 permissions: `xhost`
- Test with simple X11 app: `docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix fedora xeyes`

### 4. Out of Memory
- Increase Docker memory limit in Docker Desktop settings
- Reduce camera resolution in app settings

## 🔒 Security Considerations

1. **X11 Security**: Using `xhost +` reduces security. For production:
   ```bash
   # More secure: allow only specific container
   xhost +local:$(docker inspect -f '{{ .Config.Hostname }}' fitlifeposture_app)
   ```

2. **User Permissions**: Container runs as non-root user `fitlife`

3. **Camera Access**: Only required cameras should be mounted

## 🎯 Performance Optimization

1. **Use GPU acceleration** when available:
   ```bash
   docker-compose --profile gpu up
   ```

2. **Adjust camera resolution** in settings.json:
   ```json
   {
     "camera_resolution": "640x480"
   }
   ```

3. **Limit FPS** for lower-end systems:
   ```json
   {
     "processing_fps": 15
   }
   ```

## 📊 Monitoring

```bash
# Container resource usage
docker stats fitlifeposture_app

# Health check status
docker inspect fitlifeposture_app | grep -A 5 "Health"
```

## 🔄 Updates

```bash
# Pull latest changes
git pull

# Rebuild image
docker-compose build --no-cache

# Restart with new image
docker-compose up -d
```

## 📝 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DISPLAY` | `:0` | X11 display |
| `CAMERA_INDEX` | `0` | Camera device index |
| `PYTHONUNBUFFERED` | `1` | Immediate output |
| `QT_X11_NO_MITSHM` | `1` | Fix Qt rendering |

## 🆘 Support

For issues specific to Docker:
1. Check container logs: `docker logs fitlifeposture_app`
2. Verify system requirements
3. Try running outside Docker to isolate issues
4. Open an issue with Docker version and error logs

---

Happy posture analysis! 🏃‍♂️✨

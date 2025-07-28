# FitLifePosture Docker Quick Start

## 🚀 Quick Commands

### Standard GUI Mode
```bash
# Build and run with GUI
docker-compose up -d

# Or using the helper script
./docker-manager.sh build
./docker-manager.sh run
```

### Web Interface Mode
```bash
# Run web-only version (no X11 needed)
docker-compose --profile web up -d

# Access at: http://localhost:8080
```

### GPU Accelerated Mode
```bash
# Run with NVIDIA GPU
docker-compose --profile gpu up -d
```

## 🌐 Running Modes

### 1. GUI Mode (Default)
- Full desktop interface
- Requires X11 display
- Best for local development

### 2. Web Mode
- Browser-based interface
- No X11 required
- Perfect for remote access
- Access at `http://localhost:8080`

### 3. Headless Mode
- API-only operation
- For integration with other systems

## 🎯 Common Tasks

### Check if running
```bash
docker ps | grep fitlifeposture
```

### View logs
```bash
docker logs -f fitlifeposture_app
```

### Switch to web mode
```bash
# Stop current container
docker-compose down

# Start in web mode
docker-compose run -e INTERFACE_MODE=web fitlifeposture
```

### Access container shell
```bash
docker exec -it fitlifeposture_app bash
```

## 🐧 Linux Quick Start
```bash
# Allow X11 connections
xhost +local:docker

# Run
docker-compose up -d
```

## 🍎 macOS Quick Start
```bash
# Install XQuartz
brew install --cask xquartz

# Start XQuartz and enable network connections
open -a XQuartz

# In XQuartz preferences: Security > Allow connections from network clients

# Run web mode (recommended for macOS)
docker-compose --profile web up -d
```

## 🪟 Windows Quick Start
```bash
# In WSL2 terminal
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0

# Run web mode (recommended)
docker-compose --profile web up -d
```

## 📱 Remote Access

To access from other devices on your network:

1. Find your IP address:
   ```bash
   # Linux/macOS
   hostname -I | awk '{print $1}'
   
   # or
   ip addr show | grep "inet " | grep -v 127.0.0.1
   ```

2. Access from any device:
   ```
   http://YOUR_IP:8080
   ```

## 🔧 Troubleshooting

### Camera not found
```bash
# Check camera devices
ls -la /dev/video*

# Test camera in container
docker exec fitlifeposture_app python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8081:8080"  # Use 8081 instead
```

### Permission denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

## 💡 Pro Tips

1. **Performance**: Use GPU mode for 3x faster analysis
2. **Storage**: Mount additional volumes for large datasets
3. **Security**: Use HTTPS proxy for production deployment
4. **Scaling**: Use Docker Swarm for multi-node deployment

---

Need help? Check the full [DOCKER_GUIDE.md](./DOCKER_GUIDE.md)

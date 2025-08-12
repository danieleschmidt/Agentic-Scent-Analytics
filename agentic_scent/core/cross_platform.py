"""
Cross-platform compatibility layer for multi-environment deployment.
"""

import os
import sys
import logging
import platform
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import tempfile
import shutil


class Platform(Enum):
    """Supported platforms."""
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "darwin"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"


class Architecture(Enum):
    """Supported architectures."""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    ARM32 = "arm32"
    UNKNOWN = "unknown"


@dataclass
class SystemInfo:
    """System information."""
    platform: Platform
    architecture: Architecture
    os_version: str
    python_version: str
    has_docker: bool = False
    has_kubernetes: bool = False
    container_runtime: Optional[str] = None
    memory_gb: float = 0.0
    cpu_cores: int = 0


class PlatformManager:
    """
    Manages cross-platform compatibility and system detection.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system_info = self._detect_system()
        
        # Platform-specific configurations
        self.platform_configs = self._load_platform_configs()
        
        # Path configurations
        self.paths = self._setup_platform_paths()
        
        self.logger.info(f"Platform detected: {self.system_info.platform.value} ({self.system_info.architecture.value})")
    
    def _detect_system(self) -> SystemInfo:
        """Detect current system information."""
        # Platform detection
        system = platform.system().lower()
        if system == "linux":
            if self._is_docker_container():
                detected_platform = Platform.DOCKER
            else:
                detected_platform = Platform.LINUX
        elif system == "darwin":
            detected_platform = Platform.MACOS
        elif system == "windows":
            detected_platform = Platform.WINDOWS
        else:
            detected_platform = Platform.LINUX  # Default fallback
        
        # Architecture detection
        machine = platform.machine().lower()
        if machine in ["x86_64", "amd64"]:
            arch = Architecture.X86_64
        elif machine in ["arm64", "aarch64"]:
            arch = Architecture.ARM64
        elif machine in ["arm", "armv7l"]:
            arch = Architecture.ARM32
        else:
            arch = Architecture.UNKNOWN
        
        # System resources
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_cores = psutil.cpu_count()
        except ImportError:
            memory_gb = 0.0
            cpu_cores = 0
        
        # Container runtime detection
        container_runtime = None
        has_docker = self._check_command_available("docker")
        has_kubernetes = self._check_command_available("kubectl")
        
        if has_docker:
            try:
                result = subprocess.run(["docker", "version", "--format", "json"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    container_runtime = "docker"
            except:
                pass
        
        return SystemInfo(
            platform=detected_platform,
            architecture=arch,
            os_version=platform.platform(),
            python_version=platform.python_version(),
            has_docker=has_docker,
            has_kubernetes=has_kubernetes,
            container_runtime=container_runtime,
            memory_gb=memory_gb,
            cpu_cores=cpu_cores
        )
    
    def _is_docker_container(self) -> bool:
        """Check if running inside Docker container."""
        # Check for Docker-specific files
        docker_indicators = [
            Path("/.dockerenv"),
            Path("/proc/1/cgroup")
        ]
        
        for indicator in docker_indicators:
            if indicator.exists():
                if indicator.name == "cgroup":
                    try:
                        with open(indicator) as f:
                            content = f.read()
                            if "docker" in content or "containerd" in content:
                                return True
                    except:
                        pass
                else:
                    return True
        
        return False
    
    def _check_command_available(self, command: str) -> bool:
        """Check if a command is available on the system."""
        try:
            subprocess.run([command, "--version"], 
                          capture_output=True, timeout=5)
            return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _load_platform_configs(self) -> Dict[Platform, Dict[str, Any]]:
        """Load platform-specific configurations."""
        return {
            Platform.LINUX: {
                "service_manager": "systemd",
                "config_dir": "/etc/agentic-scent",
                "data_dir": "/var/lib/agentic-scent",
                "log_dir": "/var/log/agentic-scent",
                "pid_file": "/var/run/agentic-scent.pid",
                "user": "agentic-scent",
                "group": "agentic-scent",
                "file_permissions": 0o644,
                "dir_permissions": 0o755,
                "executable_permissions": 0o755
            },
            
            Platform.WINDOWS: {
                "service_manager": "windows_service",
                "config_dir": os.path.expandvars(r"%PROGRAMDATA%\AgenticScent"),
                "data_dir": os.path.expandvars(r"%PROGRAMDATA%\AgenticScent\Data"),
                "log_dir": os.path.expandvars(r"%PROGRAMDATA%\AgenticScent\Logs"),
                "pid_file": None,
                "user": None,
                "group": None,
                "file_permissions": None,
                "dir_permissions": None,
                "executable_permissions": None
            },
            
            Platform.MACOS: {
                "service_manager": "launchd",
                "config_dir": "/usr/local/etc/agentic-scent",
                "data_dir": "/usr/local/var/agentic-scent",
                "log_dir": "/usr/local/var/log/agentic-scent",
                "pid_file": "/usr/local/var/run/agentic-scent.pid",
                "user": "_agentic-scent",
                "group": "_agentic-scent",
                "file_permissions": 0o644,
                "dir_permissions": 0o755,
                "executable_permissions": 0o755
            },
            
            Platform.DOCKER: {
                "service_manager": "process",
                "config_dir": "/app/config",
                "data_dir": "/app/data",
                "log_dir": "/app/logs",
                "pid_file": "/app/run/agentic-scent.pid",
                "user": "app",
                "group": "app",
                "file_permissions": 0o644,
                "dir_permissions": 0o755,
                "executable_permissions": 0o755
            }
        }
    
    def _setup_platform_paths(self) -> Dict[str, Path]:
        """Setup platform-specific paths."""
        config = self.platform_configs.get(self.system_info.platform, 
                                          self.platform_configs[Platform.LINUX])
        
        # Create base paths
        paths = {
            "config": Path(config["config_dir"]),
            "data": Path(config["data_dir"]),
            "logs": Path(config["log_dir"]),
            "temp": Path(tempfile.gettempdir()) / "agentic-scent",
            "home": Path.home() / ".agentic-scent"
        }
        
        # Create directories if they don't exist
        for path_type, path in paths.items():
            if path_type != "temp":  # Don't auto-create temp directory
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    if config.get("dir_permissions"):
                        path.chmod(config["dir_permissions"])
                except PermissionError:
                    # Fall back to user directory if system directories not accessible
                    fallback_path = Path.home() / f".agentic-scent/{path_type}"
                    fallback_path.mkdir(parents=True, exist_ok=True)
                    paths[path_type] = fallback_path
                    self.logger.info(f"Using fallback path for {path_type}: {fallback_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to create {path_type} directory: {e}")
        
        return paths
    
    def get_config_path(self, filename: str) -> Path:
        """Get platform-appropriate config file path."""
        return self.paths["config"] / filename
    
    def get_data_path(self, filename: str) -> Path:
        """Get platform-appropriate data file path."""
        return self.paths["data"] / filename
    
    def get_log_path(self, filename: str) -> Path:
        """Get platform-appropriate log file path."""
        return self.paths["logs"] / filename
    
    def get_temp_path(self, filename: str) -> Path:
        """Get platform-appropriate temporary file path."""
        temp_dir = self.paths["temp"]
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir / filename
    
    def is_admin(self) -> bool:
        """Check if running with administrator/root privileges."""
        if self.system_info.platform == Platform.WINDOWS:
            try:
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            except:
                return False
        else:
            return os.geteuid() == 0
    
    def get_system_service_config(self) -> Dict[str, Any]:
        """Get system service configuration for the current platform."""
        platform_config = self.platform_configs.get(self.system_info.platform)
        if not platform_config:
            return {}
        
        service_config = {
            "service_manager": platform_config.get("service_manager"),
            "user": platform_config.get("user"),
            "group": platform_config.get("group"),
            "working_directory": str(self.paths["data"]),
            "pid_file": platform_config.get("pid_file"),
            "log_file": str(self.get_log_path("service.log")),
            "config_file": str(self.get_config_path("service.conf"))
        }
        
        # Platform-specific service configurations
        if self.system_info.platform == Platform.LINUX:
            service_config.update({
                "systemd_unit_file": "/etc/systemd/system/agentic-scent.service",
                "systemd_user_unit_file": f"{Path.home()}/.config/systemd/user/agentic-scent.service"
            })
        elif self.system_info.platform == Platform.WINDOWS:
            service_config.update({
                "service_name": "AgenticScentService",
                "service_display_name": "Agentic Scent Analytics Service",
                "service_description": "Industrial AI platform for smart factory e-nose deployments"
            })
        elif self.system_info.platform == Platform.MACOS:
            service_config.update({
                "launchd_plist": "/Library/LaunchDaemons/com.terragonlabs.agentic-scent.plist",
                "launchd_user_plist": f"{Path.home()}/Library/LaunchAgents/com.terragonlabs.agentic-scent.plist"
            })
        
        return service_config
    
    def optimize_for_platform(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize configuration for the current platform."""
        optimized = config.copy()
        
        # Platform-specific optimizations
        if self.system_info.platform == Platform.WINDOWS:
            # Windows-specific optimizations
            optimized.setdefault("async_io", "asyncio")  # Use asyncio on Windows
            optimized.setdefault("multiprocessing_context", "spawn")
            optimized.setdefault("file_locking", "msvcrt")
            
        elif self.system_info.platform in [Platform.LINUX, Platform.DOCKER]:
            # Linux-specific optimizations
            optimized.setdefault("async_io", "uvloop")  # Use uvloop for better performance
            optimized.setdefault("multiprocessing_context", "fork")
            optimized.setdefault("file_locking", "fcntl")
            
        elif self.system_info.platform == Platform.MACOS:
            # macOS-specific optimizations
            optimized.setdefault("async_io", "asyncio")  # uvloop has issues on some macOS versions
            optimized.setdefault("multiprocessing_context", "fork")
            optimized.setdefault("file_locking", "fcntl")
        
        # Memory-based optimizations
        if self.system_info.memory_gb > 8.0:
            optimized.setdefault("cache_size_mb", 512)
            optimized.setdefault("max_workers", min(32, self.system_info.cpu_cores * 2))
        elif self.system_info.memory_gb > 4.0:
            optimized.setdefault("cache_size_mb", 256)
            optimized.setdefault("max_workers", min(16, self.system_info.cpu_cores))
        else:
            optimized.setdefault("cache_size_mb", 128)
            optimized.setdefault("max_workers", max(2, self.system_info.cpu_cores // 2))
        
        # Container-specific optimizations
        if self.system_info.platform == Platform.DOCKER:
            optimized.setdefault("log_to_stdout", True)
            optimized.setdefault("config_hot_reload", False)
            optimized.setdefault("health_check_interval", 30)
        
        return optimized
    
    def get_recommended_deployment(self) -> Dict[str, Any]:
        """Get recommended deployment strategy for the current platform."""
        recommendations = {
            "platform": self.system_info.platform.value,
            "architecture": self.system_info.architecture.value,
            "deployment_type": "standalone",
            "scaling_strategy": "vertical",
            "service_management": "manual",
            "monitoring_strategy": "basic"
        }
        
        # Platform-specific recommendations
        if self.system_info.platform == Platform.DOCKER:
            recommendations.update({
                "deployment_type": "containerized",
                "scaling_strategy": "horizontal",
                "orchestration": "docker-compose",
                "service_management": "container",
                "monitoring_strategy": "prometheus"
            })
            
            if self.system_info.has_kubernetes:
                recommendations.update({
                    "orchestration": "kubernetes",
                    "scaling_strategy": "auto",
                    "service_management": "kubernetes",
                    "monitoring_strategy": "prometheus_operator"
                })
        
        elif self.system_info.platform == Platform.LINUX and self.system_info.memory_gb > 4.0:
            recommendations.update({
                "deployment_type": "native",
                "service_management": "systemd",
                "monitoring_strategy": "systemd_journal"
            })
        
        elif self.system_info.platform == Platform.WINDOWS:
            recommendations.update({
                "deployment_type": "windows_service",
                "service_management": "windows_service_manager",
                "monitoring_strategy": "windows_event_log"
            })
        
        elif self.system_info.platform == Platform.MACOS:
            recommendations.update({
                "deployment_type": "launchd_service",
                "service_management": "launchd",
                "monitoring_strategy": "console_log"
            })
        
        return recommendations
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check for platform-specific dependencies."""
        dependencies = {}
        
        # Python dependencies
        python_deps = ["asyncio", "logging", "pathlib", "json"]
        for dep in python_deps:
            try:
                __import__(dep)
                dependencies[f"python_{dep}"] = True
            except ImportError:
                dependencies[f"python_{dep}"] = False
        
        # Optional performance dependencies
        optional_deps = ["uvloop", "psutil", "numpy", "pandas"]
        for dep in optional_deps:
            try:
                __import__(dep)
                dependencies[f"optional_{dep}"] = True
            except ImportError:
                dependencies[f"optional_{dep}"] = False
        
        # System commands
        system_commands = ["docker", "kubectl", "systemctl", "curl", "wget"]
        for cmd in system_commands:
            dependencies[f"command_{cmd}"] = self._check_command_available(cmd)
        
        return dependencies
    
    def generate_install_script(self) -> str:
        """Generate platform-specific installation script."""
        if self.system_info.platform == Platform.LINUX:
            return self._generate_linux_install_script()
        elif self.system_info.platform == Platform.WINDOWS:
            return self._generate_windows_install_script()
        elif self.system_info.platform == Platform.MACOS:
            return self._generate_macos_install_script()
        elif self.system_info.platform == Platform.DOCKER:
            return self._generate_docker_install_script()
        else:
            return "# Platform not supported for automated installation"
    
    def _generate_linux_install_script(self) -> str:
        """Generate Linux installation script."""
        return """#!/bin/bash
# Agentic Scent Analytics - Linux Installation Script

set -e

echo "Installing Agentic Scent Analytics on Linux..."

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Installing system-wide..."
    INSTALL_TYPE="system"
    CONFIG_DIR="/etc/agentic-scent"
    DATA_DIR="/var/lib/agentic-scent"
    LOG_DIR="/var/log/agentic-scent"
else
    echo "Installing for current user..."
    INSTALL_TYPE="user"
    CONFIG_DIR="$HOME/.agentic-scent/config"
    DATA_DIR="$HOME/.agentic-scent/data"
    LOG_DIR="$HOME/.agentic-scent/logs"
fi

# Create directories
mkdir -p "$CONFIG_DIR" "$DATA_DIR" "$LOG_DIR"

# Install Python dependencies
python3 -m pip install --user agentic-scent-analytics[all]

# Create systemd service (if system install)
if [ "$INSTALL_TYPE" = "system" ]; then
    cat > /etc/systemd/system/agentic-scent.service << EOF
[Unit]
Description=Agentic Scent Analytics Service
After=network.target

[Service]
Type=simple
User=agentic-scent
Group=agentic-scent
WorkingDirectory=$DATA_DIR
ExecStart=$(which python3) -m agentic_scent.cli start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable agentic-scent
fi

echo "Installation completed successfully!"
"""
    
    def _generate_windows_install_script(self) -> str:
        """Generate Windows installation script."""
        return """@echo off
REM Agentic Scent Analytics - Windows Installation Script

echo Installing Agentic Scent Analytics on Windows...

REM Check for admin privileges
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running with administrator privileges...
    set INSTALL_TYPE=system
    set CONFIG_DIR=%PROGRAMDATA%\\AgenticScent
    set DATA_DIR=%PROGRAMDATA%\\AgenticScent\\Data
    set LOG_DIR=%PROGRAMDATA%\\AgenticScent\\Logs
) else (
    echo Running as regular user...
    set INSTALL_TYPE=user
    set CONFIG_DIR=%USERPROFILE%\\.agentic-scent\\config
    set DATA_DIR=%USERPROFILE%\\.agentic-scent\\data
    set LOG_DIR=%USERPROFILE%\\.agentic-scent\\logs
)

REM Create directories
mkdir "%CONFIG_DIR%" 2>nul
mkdir "%DATA_DIR%" 2>nul
mkdir "%LOG_DIR%" 2>nul

REM Install Python dependencies
python -m pip install --user agentic-scent-analytics[all]

REM Install as Windows service (if admin)
if "%INSTALL_TYPE%"=="system" (
    python -m agentic_scent.cli install-service
)

echo Installation completed successfully!
pause
"""
    
    def _generate_macos_install_script(self) -> str:
        """Generate macOS installation script."""
        return """#!/bin/bash
# Agentic Scent Analytics - macOS Installation Script

set -e

echo "Installing Agentic Scent Analytics on macOS..."

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Installing system-wide..."
    INSTALL_TYPE="system"
    CONFIG_DIR="/usr/local/etc/agentic-scent"
    DATA_DIR="/usr/local/var/agentic-scent"
    LOG_DIR="/usr/local/var/log/agentic-scent"
else
    echo "Installing for current user..."
    INSTALL_TYPE="user"
    CONFIG_DIR="$HOME/.agentic-scent/config"
    DATA_DIR="$HOME/.agentic-scent/data"
    LOG_DIR="$HOME/.agentic-scent/logs"
fi

# Create directories
mkdir -p "$CONFIG_DIR" "$DATA_DIR" "$LOG_DIR"

# Install Python dependencies
python3 -m pip install --user agentic-scent-analytics[all]

# Create launchd service
if [ "$INSTALL_TYPE" = "system" ]; then
    PLIST_PATH="/Library/LaunchDaemons/com.terragonlabs.agentic-scent.plist"
else
    PLIST_PATH="$HOME/Library/LaunchAgents/com.terragonlabs.agentic-scent.plist"
fi

cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.terragonlabs.agentic-scent</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(which python3)</string>
        <string>-m</string>
        <string>agentic_scent.cli</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>$DATA_DIR</string>
</dict>
</plist>
EOF

launchctl load "$PLIST_PATH"

echo "Installation completed successfully!"
"""
    
    def _generate_docker_install_script(self) -> str:
        """Generate Docker installation script."""
        return """#!/bin/bash
# Agentic Scent Analytics - Docker Installation Script

set -e

echo "Setting up Agentic Scent Analytics with Docker..."

# Create docker-compose.yml
cat > docker-compose.yml << EOF
version: '3.8'

services:
  agentic-scent:
    image: terragonlabs/agentic-scent-analytics:latest
    ports:
      - "8080:8080"
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-m", "agentic_scent.cli", "health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=agentic_scent
      - POSTGRES_USER=agentic_scent
      - POSTGRES_PASSWORD=changeme
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
EOF

# Create directory structure
mkdir -p config data logs

# Start services
docker-compose up -d

echo "Docker installation completed successfully!"
echo "Access the dashboard at: http://localhost:8080"
"""
    
    def get_system_info(self) -> SystemInfo:
        """Get detected system information."""
        return self.system_info
    
    def get_platform_paths(self) -> Dict[str, Path]:
        """Get platform-specific paths."""
        return self.paths.copy()


# Global platform manager
_platform_manager = None


def get_platform_manager() -> PlatformManager:
    """Get global platform manager instance."""
    global _platform_manager
    if _platform_manager is None:
        _platform_manager = PlatformManager()
    return _platform_manager


def get_system_info() -> SystemInfo:
    """Get system information."""
    return get_platform_manager().get_system_info()


def is_admin() -> bool:
    """Check if running with admin privileges."""
    return get_platform_manager().is_admin()


def get_config_path(filename: str) -> Path:
    """Get platform-appropriate config path."""
    return get_platform_manager().get_config_path(filename)


def get_data_path(filename: str) -> Path:
    """Get platform-appropriate data path."""
    return get_platform_manager().get_data_path(filename)


def get_log_path(filename: str) -> Path:
    """Get platform-appropriate log path."""
    return get_platform_manager().get_log_path(filename)
#!/usr/bin/env python3
"""
Cosim Gym Setup Script (Option 3)

A Python-based setup tool that handles:
  - Environment validation
  - Conda environment creation
  - Docker setup
  - Validation and health checks

Usage:
    python setup.py                    # Interactive mode (guided setup)
    python setup.py --auto            # Non-interactive (full setup)
    python setup.py --env-only        # Setup Python environment only
    python setup.py --docker-only     # Setup Docker only
    python setup.py --validate        # Validate existing setup
    python setup.py --cleanup         # Remove environment and containers
"""

import subprocess
import sys
import os
import argparse
import time
from pathlib import Path
from typing import Tuple, Optional


class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

    @staticmethod
    def print_header(text: str) -> None:
        print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}{Colors.END}\n")

    @staticmethod
    def print_success(text: str) -> None:
        print(f"{Colors.GREEN}✓ {text}{Colors.END}")

    @staticmethod
    def print_error(text: str) -> None:
        print(f"{Colors.RED}✗ {text}{Colors.END}")

    @staticmethod
    def print_warning(text: str) -> None:
        print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

    @staticmethod
    def print_info(text: str) -> None:
        print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


class SetupManager:
    """Manages Cosim Gym setup process"""

    def __init__(self, workspace_root: Optional[str] = None):
        if workspace_root:
            self.workspace_root = Path(workspace_root)
        else:
            self.workspace_root = Path(__file__).parent
        
        self.env_name = "cosim_gym"
        self.env_file = self.workspace_root / "environment.yml"
        self.docker_compose_file = self.workspace_root / "src" / "docker-compose.yaml"

    def run_command(self, cmd: list, description: str = "") -> Tuple[bool, str]:
        """Run command and capture output"""
        try:
            if description:
                print(f"{Colors.BLUE}→ {description}...{Colors.END}", end=" ", flush=True)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                if description:
                    Colors.print_success(description)
                return True, result.stdout
            else:
                if description:
                    Colors.print_error(description)
                return False, result.stderr
        except subprocess.TimeoutExpired:
            if description:
                Colors.print_error(f"{description} (timeout)")
            return False, "Command timeout"
        except Exception as e:
            if description:
                Colors.print_error(f"{description} ({str(e)})")
            return False, str(e)

    def check_tool(self, tool: str) -> bool:
        """Check if tool is available"""
        success, _ = self.run_command(["which", tool])
        return success

    def validate_prerequisites(self) -> bool:
        """Check if required tools are installed"""
        Colors.print_header("Validating Prerequisites")
        
        required_tools = {
            "conda": "Conda/Miniconda - https://docs.conda.io/projects/miniconda/",
            "docker": "Docker Desktop - https://www.docker.com/products/docker-desktop",
            "python": "Python (usually included with Conda)"
        }
        
        missing = []
        for tool, url in required_tools.items():
            if self.check_tool(tool):
                Colors.print_success(tool.capitalize())
            else:
                Colors.print_error(tool.capitalize())
                missing.append(f"  • {tool}: {url}")
        
        if missing:
            Colors.print_error("\nMissing tools:")
            for msg in missing:
                print(msg)
            return False
        
        Colors.print_success("All prerequisites met")
        return True

    def check_files(self) -> bool:
        """Verify required project files exist"""
        Colors.print_info(f"Checking project files in {self.workspace_root}")
        
        required_files = [
            self.env_file,
            self.docker_compose_file,
        ]
        
        missing = []
        for file in required_files:
            if file.exists():
                Colors.print_success(f"Found {file.name}")
            else:
                Colors.print_error(f"Missing {file.name}")
                missing.append(str(file))
        
        return len(missing) == 0

    def setup_conda_env(self) -> bool:
        """Create conda environment from environment.yml"""
        Colors.print_header("Setting Up Python Environment")
        
        # Check if env already exists
        success, _ = self.run_command(
            ["conda", "env", "list"],
            "Checking existing environments"
        )
        
        if not success:
            Colors.print_error("Could not list conda environments")
            return False
        
        # Create environment
        colors = Colors()
        print(f"\n{colors.BLUE}Creating environment '{self.env_name}'...{colors.END}")
        success, output = self.run_command(
            ["conda", "env", "create", "-f", str(self.env_file), "--yes"],
            "Installing dependencies (this may take several minutes)"
        )
        
        if not success:
            Colors.print_error(f"Failed to create environment:\n{output}")
            return False
        
        Colors.print_success(f"Environment '{self.env_name}' created")
        Colors.print_info(f"To activate: conda activate {self.env_name}")
        return True

    def setup_docker(self) -> bool:
        """Start Docker containers"""
        Colors.print_header("Setting Up Docker")
        
        # Check Docker daemon
        success, _ = self.run_command(["docker", "ps"], "Checking Docker daemon")
        if not success:
            Colors.print_error("Docker daemon not running. Start Docker Desktop and retry.")
            return False
        
        # Start containers
        print(f"\n{Colors.BLUE}Starting containers...{Colors.END}")
        success, output = self.run_command(
            ["docker", "compose", "-f", str(self.docker_compose_file), "up", "-d"],
            "Starting services"
        )
        
        if not success:
            Colors.print_error(f"Failed to start containers:\n{output}")
            return False
        
        # Wait for Redis to be healthy
        Colors.print_info("Waiting for Redis to become healthy...")
        max_retries = 10
        for i in range(max_retries):
            success, _ = self.run_command(
                ["docker", "compose", "-f", str(self.docker_compose_file), 
                 "exec", "-T", "redis", "redis-cli", "ping"]
            )
            if success:
                Colors.print_success("Redis is healthy")
                return True
            print(f"  Attempt {i+1}/{max_retries}...", end="\r", flush=True)
            time.sleep(1)
        
        Colors.print_warning("Redis health check timed out (may resolve shortly)")
        return True

    def validate_setup(self) -> bool:
        """Validate complete setup"""
        Colors.print_header("Validating Setup")
        
        all_good = True
        
        # Check conda env
        success, _ = self.run_command(
            ["conda", "env", "list"],
            "Checking Python environment"
        )
        
        if success:
            Colors.print_success("Python environment configured")
        else:
            Colors.print_warning("Python environment not found")
            all_good = False
        
        # Check Docker
        success, _ = self.run_command(
            ["docker", "ps", "--format", "table {{.Names}}"],
            "Checking Docker services"
        )
        
        if success:
            Colors.print_success("Docker services running")
        else:
            Colors.print_warning("Docker services not available")
            all_good = False
        
        return all_good

    def cleanup(self, remove_env: bool = False, remove_docker: bool = False) -> bool:
        """Clean up environment and/or containers"""
        if remove_env:
            Colors.print_header("Removing Python Environment")
            self.run_command(
                ["conda", "env", "remove", "-n", self.env_name, "--yes"],
                f"Removing {self.env_name}"
            )
        
        if remove_docker:
            Colors.print_header("Stopping Docker Services")
            self.run_command(
                ["docker", "compose", "-f", str(self.docker_compose_file), "down"],
                "Stopping containers"
            )
        
        Colors.print_success("Cleanup complete")
        return True

    def interactive_setup(self) -> bool:
        """Interactive setup mode"""
        Colors.print_header("Cosim Gym - Interactive Setup")
        
        if not self.validate_prerequisites():
            Colors.print_error("Please install missing tools and retry.")
            return False
        
        if not self.check_files():
            Colors.print_error("Required project files not found.")
            return False
        
        print("\n" + Colors.BOLD + "Setup Options:" + Colors.END)
        print("1. Full setup (Python + Docker)")
        print("2. Python environment only")
        print("3. Docker only")
        print("4. Validate existing setup")
        print("5. Exit")
        
        choice = input(f"\n{Colors.BLUE}Select option (1-5): {Colors.END}").strip()
        
        if choice == "1":
            if not self.setup_conda_env():
                return False
            if not self.setup_docker():
                return False
        elif choice == "2":
            if not self.setup_conda_env():
                return False
        elif choice == "3":
            if not self.setup_docker():
                return False
        elif choice == "4":
            return self.validate_setup()
        elif choice == "5":
            Colors.print_info("Setup cancelled")
            return True
        else:
            Colors.print_error("Invalid option")
            return False
        
        self.validate_setup()
        Colors.print_header("Setup Complete!")
        print(f"Next steps:")
        print(f"  1. Activate environment: {Colors.BOLD}conda activate {self.env_name}{Colors.END}")
        print(f"  2. Run simulation: {Colors.BOLD}python src/test_script.py{Colors.END}")
        print(f"  3. Launch dashboard: {Colors.BOLD}streamlit run src/dashboard/streamlit_dashboard.py{Colors.END}")
        
        return True

    def auto_setup(self) -> bool:
        """Non-interactive full setup"""
        Colors.print_header("Cosim Gym - Automatic Setup")
        
        if not self.validate_prerequisites():
            return False
        
        if not self.check_files():
            return False
        
        if not self.setup_conda_env():
            return False
        
        if not self.setup_docker():
            return False
        
        self.validate_setup()
        Colors.print_header("Setup Complete!")
        print(f"Run: {Colors.BOLD}conda activate {self.env_name}{Colors.END}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Cosim Gym Setup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py                    # Interactive mode
  python setup.py --auto            # Full automatic setup
  python setup.py --env-only        # Python environment only
  python setup.py --docker-only     # Docker setup only
  python setup.py --validate        # Validate setup
  python setup.py --cleanup         # Remove environment + containers
        """
    )
    
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Non-interactive full setup"
    )
    parser.add_argument(
        "--env-only",
        action="store_true",
        help="Setup Python environment only"
    )
    parser.add_argument(
        "--docker-only",
        action="store_true",
        help="Setup Docker only"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing setup"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove environment and containers"
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Project root directory (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    manager = SetupManager(args.root)
    
    try:
        if args.auto:
            success = manager.auto_setup()
        elif args.env_only:
            success = (manager.validate_prerequisites() and 
                      manager.check_files() and 
                      manager.setup_conda_env())
        elif args.docker_only:
            success = (manager.validate_prerequisites() and 
                      manager.check_files() and 
                      manager.setup_docker())
        elif args.validate:
            success = manager.validate_setup()
        elif args.cleanup:
            success = manager.cleanup(remove_env=True, remove_docker=True)
        else:
            success = manager.interactive_setup()
        
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Setup interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        Colors.print_error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

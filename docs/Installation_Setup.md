# Quick start & Installation

To use this repository there are 4 option you could follows. The first 3 allows to setup locally the repository and are mutually exclusive, while the 4th option allow to use the repo directly without local setup.

Before following any of the **local setup** options below, you need to clone the repository and move into the project folder:

```bash
git clone <repository-url>
cd Cosim_gym
```


##  Manual Setup Recomended for the tutorial

#### pre-requisites
- python 3.12 installed
- docker installed

1.  **Create Conda Environment**:
    ```bash
    conda env create -f environment.yml
    conda activate cosim_gym
    ```

2.  **Start Infrastructure**:
    ```bash
    docker compose -f src/docker-compose.yaml up -d
    ```

3.  **Run Simulation**:
    ```bash
    python src/test_script.py
    ```

4.  **Run Dashboard**:
    ```bash
    streamlit run src/dashboard/streamlit_dashboard.py
    ```
    Access at http://localhost:8501


##  Dev Container + VS Code

#### pre-requisites
- VSCode
- docker
- dev container plugin

1. **Open repo folder in VSCode**
2. **click on open inside container**



##  Docker

#### pre-requisites
- docker

```bash
# Start all services (Redis + Python environment)
docker compose -f docker-compose.setup.yml up -d

# Verify services are running
docker compose -f docker-compose.setup.yml ps

# Wait until conda environment installation is finished
docker compose -f docker-compose.setup.yml logs -f cosim-env
# Stop following logs when you see:
# "Installation complete. Keep container running..."

# Access the Python environment
docker compose -f docker-compose.setup.yml exec cosim-env bash

# Inside the container:
source /opt/conda/etc/profile.d/conda.sh
conda activate cosim_gym
python src/test_script.py

# Or run simulation directly
docker compose -f docker-compose.setup.yml exec cosim-env \
  /opt/conda/bin/conda run -n cosim_gym python src/test_script.py
```



<!-- 

All commands shown in the local setup sections, such as `make setup`, `python setup.py`, or `docker compose -f docker-compose.setup.yml ...`, assume you are already inside the cloned repository root.

- [Local setup]()
- [No setup]()



## 📋 Setup Options Overview

Choose one of the following setup options to get started. Each option is **independent** and allows you to set up the repository with Python environment and dependencies from a single entry point.

| Option | Best For | Setup Time | Requirements |
|--------|----------|-----------|--------------|
| [**Option 1: Makefile**]() | Teams, CI/CD, multiple runs | ~5-10 min | Conda, Docker, GNU Make |
| [**Option 2: Python Script**]() | Cross-platform, flexibility | ~5-10 min | Conda, Docker, Python 3.8+ |
| [**Option 3: Docker-Only**]() | Full isolation, reproducibility | ~10-15 min | Docker only |

> **📌 Note:** These options are **independent and mutually exclusive**. Choose *one* for your workflow.
> They do not conflict with DevContainer development (see [Dev Container Setup](#dev-container-setup-optional---vs-code) below).

---

## Option 1: Makefile Setup (Recommended for Linux/macOS) 🔨

The simplest approach using **Makefile** with color-coded commands.

### Prerequisites
- Conda (Miniconda/Anaconda): [Install](https://docs.conda.io/projects/miniconda/)
- Docker Desktop: [Install](https://www.docker.com/products/docker-desktop)
- GNU Make: Usually pre-installed on macOS/Linux

### Quick Setup

```bash
# Show available commands
make help

# Full setup (Python environment + Docker)
make setup

# Run simulation
make run

# Launch dashboard
make run-dashboard

# Check status
make status
```

### Available Commands

```bash
make setup              # Complete setup (environment + Docker)
make setup-env         # Python environment only
make setup-docker      # Docker services only
make run               # Start simulation
make run-dashboard     # Launch Streamlit dashboard
make validate          # Validate setup
make clean             # Stop Docker containers
make teardown          # Full cleanup (remove env + containers)
make logs              # View Redis logs
```

### Example Workflow

```bash
# First time setup
make setup

# Check everything is ready
make validate

# Run simulation
make run

# In another terminal, launch dashboard
make run-dashboard

# When done, stop services
make clean
```

---

## Option 2: Python Setup Script (Best for Cross-Platform) 🐍

An interactive Python-based setup tool with detailed status reporting.

### Prerequisites
- Conda (Miniconda/Anaconda): [Install](https://docs.conda.io/projects/miniconda/)
- Docker Desktop: [Install](https://www.docker.com/products/docker-desktop)
- Python 3.8+

### Quick Setup

```bash
# Interactive mode (guided)
python setup.py

# Or: Full non-interactive setup
python setup.py --auto

# Environment only
python setup.py --env-only

# Docker only
python setup.py --docker-only

# Validate existing setup
python setup.py --validate

# Full cleanup
python setup.py --cleanup
```

### Detailed Options

```bash
python setup.py                    # Interactive mode - choose what to setup
python setup.py --auto            # Non-interactive full setup
python setup.py --env-only        # Python environment only (no Docker)
python setup.py --docker-only     # Docker services only (no Conda env)
python setup.py --validate        # Check current setup status
python setup.py --cleanup         # Remove environment and containers
python setup.py --root /path/to/repo  # Specify project root (auto-detected by default)
```

### Example Workflow

```bash
# First time: interactive setup
python setup.py

# Or for automation
python setup.py --auto

# In future: validate only
python setup.py --validate

# Run simulation (after activation)
conda activate cosim_gym
python src/test_script.py

# Launch dashboard
streamlit run src/dashboard/streamlit_dashboard.py
```

---

## Option 3: Docker-Only Setup (Most Reproducible) 🐳

Everything runs in Docker containers - minimal host dependencies, maximum reproducibility.

### Prerequisites
- Docker Engine / Docker Desktop: [Install](https://docs.docker.com/engine/install/)
- That's it!

For Ubuntu servers, make sure Docker Compose is available in one of these forms:

```bash
docker compose version
# or (legacy binary)
docker-compose --version
```

If `docker compose` is not available on your server, use `docker-compose` in all commands below.

### Quick Setup

```bash
# Start all services (Redis + Python environment)
docker compose -f docker-compose.setup.yml up -d

# Verify services are running
docker compose -f docker-compose.setup.yml ps

# Wait until conda environment installation is finished
docker compose -f docker-compose.setup.yml logs -f cosim-env
# Stop following logs when you see:
# "Installation complete. Keep container running..."

# Access the Python environment
docker compose -f docker-compose.setup.yml exec cosim-env bash

# Inside the container:
source /opt/conda/etc/profile.d/conda.sh
conda activate cosim_gym
python src/test_script.py

# Or run simulation directly
docker compose -f docker-compose.setup.yml exec cosim-env \
  /opt/conda/bin/conda run -n cosim_gym python src/test_script.py
```

### Ubuntu Troubleshooting: `unknown shorthand flag: 'f' in -f`

If you see:

```text
unknown shorthand flag: 'f' in -f
```

usually one of these happened:

1. The command was run with the wrong order (`up -f ...` instead of `-f ... up`).
2. Your host uses `docker-compose` (legacy) instead of `docker compose`.
3. The command was accidentally run as `docker -f ...` (missing `compose`).

Use one of these exact forms:

```bash
# Compose plugin (preferred)
docker compose -f docker-compose.setup.yml up -d

# Legacy binary
docker-compose -f docker-compose.setup.yml up -d
```

Important:

- `-f docker-compose.setup.yml` must come right after `docker compose` (or `docker-compose`)
- do not place `-f` after `up`

If you instead see `conda: command not found`, usually one of these happened:

1. `cosim-env` is still installing Miniconda/environment (wait for completion logs).
2. You ran inside a shell that did not source Conda init scripts.
3. You used `bash -c "conda activate ..."` (non-interactive shell), which is not reliable.

Use one of these reliable patterns:

```bash
# One-shot command (recommended in docs/scripts)
docker compose -f docker-compose.setup.yml exec cosim-env \
  /opt/conda/bin/conda run -n cosim_gym python src/test_script.py

# Interactive shell
docker compose -f docker-compose.setup.yml exec cosim-env bash
source /opt/conda/etc/profile.d/conda.sh
conda activate cosim_gym
```

If `conda` exists but `cosim_gym` is missing from `conda env list`, the environment creation likely failed during container startup.

Use this diagnostic flow:

```bash
# 1) Check startup logs (look for conda/pip errors)
docker compose -f docker-compose.setup.yml logs cosim-env

# 2) Enter container
docker compose -f docker-compose.setup.yml exec cosim-env bash

# 3) Check available environments
/opt/conda/bin/conda env list

# 4) Recreate env manually with full output
/opt/conda/bin/conda env create -f /app/environment.yml

# 5) Verify and run
/opt/conda/bin/conda run -n cosim_gym python -V
/opt/conda/bin/conda run -n cosim_gym python src/test_script.py
```

If logs stop at:

```text
Creating environment from environment.yml...
Do you accept the Terms of Service (ToS) ...
```

the setup is blocked by interactive Conda ToS confirmation. The Docker setup now auto-accepts ToS, but you need to recreate the container so the updated command is applied:

```bash
docker compose -f docker-compose.setup.yml down
docker compose -f docker-compose.setup.yml up -d --force-recreate cosim-env
docker compose -f docker-compose.setup.yml logs -f cosim-env
```

Wait until:

```text
Installation complete. Keep container running...
```

Tip:

- This repository pins the Conda environment to `python=3.11` in `environment.yml` for better package compatibility across Linux servers.

### Typical Workflow

```bash
# First time setup
docker compose -f docker-compose.setup.yml up -d

# Check status
docker compose -f docker-compose.setup.yml ps

# Run simulation inside container
docker compose -f docker-compose.setup.yml exec cosim-env \
  /opt/conda/bin/conda run -n cosim_gym python src/test_script.py

# View logs
docker compose -f docker-compose.setup.yml logs -f cosim-env

# Stop everything
docker compose -f docker-compose.setup.yml down

# Full cleanup (including volumes)
docker compose -f docker-compose.setup.yml down -v
```

### Advanced: Interactive Development

```bash
# Start services
docker compose -f docker-compose.setup.yml up -d

# Enter interactive shell
docker compose -f docker-compose.setup.yml exec cosim-env bash

# Inside container, activate environment
source /opt/conda/etc/profile.d/conda.sh
conda activate cosim_gym

# Now you can run any command
python src/test_script.py
streamlit run src/dashboard/streamlit_dashboard.py
```

---

## Accessing the Dashboard on a Remote Ubuntu Server

If you install and run the repository on a remote Ubuntu server, the dashboard still runs on that server, but you usually open it from your local machine browser.

### Recommended: SSH Port Forwarding

The safest and simplest option is to keep Streamlit bound to localhost on the server and forward the port through SSH.

On the remote server:

```bash
streamlit run src/dashboard/streamlit_dashboard.py --server.port 8501 --server.address 127.0.0.1
```

On your local machine:

```bash
ssh -L 8501:127.0.0.1:8501 your_user@your_server
```

Then open in your local browser:

```text
http://localhost:8501
```

### Alternative: Expose the Dashboard on the Server Network

If you explicitly want the dashboard reachable from outside the server, bind Streamlit to all interfaces:

```bash
streamlit run src/dashboard/streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0
```

Then open:

```text
http://<server-ip>:8501
```

This approach requires:

- the server firewall to allow port `8501`
- the network security rules to allow inbound access
- extra care, because the dashboard becomes reachable from outside the server

### Practical Recommendation

For development, testing, and most research workflows, prefer **SSH port forwarding**. It is usually the easiest and safest way to use the dashboard on a remote server without exposing it publicly.

---

## Manual Setup (Original Workflow)

If you prefer manual setup without automation tools:

1.  **Create Conda Environment**:
    ```bash
    conda env create -f environment.yml
    conda activate cosim_gym
    ```

2.  **Start Infrastructure**:
    ```bash
    docker compose -f src/docker-compose.yaml up -d
    ```

3.  **Run Simulation**:
    ```bash
    python src/test_script.py
    ```

4.  **Run Dashboard**:
    ```bash
    streamlit run src/dashboard/streamlit_dashboard.py
    ```
    Access at http://localhost:8501

---

## NO setup - DevContainer approach (Optional - VS Code)

For development inside a containerized VS Code environment:
### About Dev Containers

Dev Containers are a **separate development workflow** managed by VS Code. They use `.devcontainer/devcontainer.json` and are **independent** of the three setup options above.

**Note:** Dev Containers use different Docker resources than Options 1-3:
- Container names: `devcontainer`, `cosim_redis` (no "_setup" suffix)
- Volume names: `redis_data` (no "_setup" suffix)  
- Docker Compose files: `.devcontainer/docker-compose.yml` + `src/docker-compose.yaml`

**These do NOT conflict** with the setup options above. You can use either:
- **Setup Options 1-3**: For standard local or Docker-based workflows
- **Dev Containers**: Exclusively for VS Code development workflow

### Setup Steps
1.  **Prerequisites**:
    - Docker Desktop
    - VS Code + [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2.  **Open in Container**:
    - Open this folder in VS Code
    - Click **Reopen in Container** (or: `Dev Containers: Reopen in Container` in Command Palette)

3.  **Development**:
    - **Redis** is available at `redis:6379`
    - Run: `python src/test_script.py`
    - Dashboard: `streamlit run src/dashboard/streamlit_dashboard.py` → http://localhost:8501

 -->

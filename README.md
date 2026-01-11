# NovoMD - Molecular Dynamics API

[![CI](https://github.com/quantnexusai/NovoMD/actions/workflows/ci.yml/badge.svg)](https://github.com/quantnexusai/NovoMD/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com)

Open-source REST API for molecular dynamics simulations, protein-ligand docking, and conformational analysis.

![NovoMD Demo](demo.gif)

## Features

- **SMILES to OpenMD Conversion**: Convert SMILES strings to OpenMD format files with real 3D optimization
- **Comprehensive Property Calculation**: 32+ molecular descriptors calculated from real 3D coordinates
  - **Geometry** (7): radius of gyration, asphericity, eccentricity, inertia tensors, PMI
  - **Energy** (6): conformer energy, VDW, electrostatic, torsion/angle strain
  - **Electrostatics** (6): dipole moment, partial charges, electrostatic potential
  - **Surface/Volume** (4): SASA, molecular volume, globularity, surface-to-volume ratio
  - **Atom Counts** (2): total atoms, heavy atoms
  - **3D Visualization** (5+): full atomic coordinates, bond connectivity, PDB format
- **Multiple Force Fields**: Support for AMBER, CHARMM, OPLS, and GROMOS force fields
- **Real Calculations Only**: No mock data - all properties derived from actual 3D structures

> **Note**: This API focuses on real molecular property calculations. For full MD simulations (trajectories, binding affinity), integrate with GROMACS, AMBER, or similar MD engines.

## Tech Stack

- **Framework**: FastAPI
- **Molecular Processing**: RDKit, OpenBabel (optional)
- **Chemistry**: BioPython
- **Deployment**: Docker, Docker Compose

## Quick Start

### Using Pre-built Docker Image (Fastest)

```bash
# Pull and run the latest image
docker run -d \
  -p 8010:8010 \
  -e NOVOMD_API_KEY="your-secure-api-key" \
  --name novomd \
  ghcr.io/quantnexusai/novomd:latest

# Test the service
curl http://localhost:8010/health
```

Available tags: `latest`, `main`, `v1.1.0`, `v1.1`, `v1`

### Using Docker Compose

1. **Clone the repository**
   ```bash
   git clone https://github.com/quantnexusai/NovoMD.git
   cd NovoMD
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and set NOVOMD_API_KEY to a secure random string
   ```

3. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Test the service**
   ```bash
   curl http://localhost:8010/health
   ```

### Local Development

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install RDKit (optional but recommended)**
   ```bash
   # Using pip
   pip install rdkit-pypi

   # Or using conda (recommended)
   conda install -c conda-forge rdkit
   ```

3. **Set environment variables**
   ```bash
   export NOVOMD_API_KEY="your-secure-api-key"
   export PORT=8010
   ```

4. **Run the server**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8010
   ```

5. **Access API documentation**
   - Swagger UI: http://localhost:8010/docs
   - ReDoc: http://localhost:8010/redoc

## MCP Server (AI Assistant Integration)

NovoMD is available as an MCP (Model Context Protocol) server, allowing AI assistants like Claude to directly query molecular properties.

**MCP Endpoint:** `https://quantnexusai-novomd.hf.space/gradio_api/mcp/sse`

### Adding to Claude Desktop

1. Open Claude Desktop settings
2. Go to **Integrations** > **Add Custom Connector**
3. Enter the MCP URL:
   ```
   https://quantnexusai-novomd.hf.space/gradio_api/mcp/sse
   ```
4. Save and restart Claude

Once connected, you can ask Claude questions like:
- "What are the 3D coordinates for Acetaminophen?"
- "Calculate the molecular properties of aspirin (CC(=O)OC1=CC=CC=C1C(=O)O)"
- "What is the dipole moment of caffeine?"

### Response Data

The MCP server returns comprehensive molecular data including:
- **Properties**: 20+ calculated descriptors (geometry, energy, electrostatics, surface/volume)
- **3D Structure**: Full atomic coordinates (`coords_x`, `coords_y`, `coords_z`), atom types, and bond connectivity

### Other MCP-Compatible AI Assistants

NovoMD's MCP server works with any AI assistant that supports the Model Context Protocol:
- **Claude Desktop** - Native MCP support
- **Cursor** - Via MCP configuration
- **Continue.dev** - VS Code extension with MCP support
- **Custom agents** - Any application using the [MCP specification](https://modelcontextprotocol.io/)

For programmatic access, use the SSE endpoint directly or integrate via the [MCP SDK](https://github.com/modelcontextprotocol/sdk).

## API Usage

### Authentication

All endpoints (except `/health`) require an API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8010/status
```

### Examples

#### Convert SMILES to OpenMD Format with Full Property Calculation

```bash
curl -X POST http://localhost:8010/smiles-to-omd \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "smiles": "CCO",
    "force_field": "AMBER",
    "optimize_3d": true,
    "add_hydrogens": true,
    "box_size": 30.0
  }'
```

**Returns 32+ Molecular Properties:**
- **Geometry** (7): radius_of_gyration, asphericity, eccentricity, inertia_shape_factor, span_r, pmi1, pmi2
- **Energy** (6): conformer_energy, vdw_energy, electrostatic_energy, torsion_strain, angle_strain, optimization_delta
- **Electrostatics** (6): dipole_moment, total_charge, max/min_partial_charge, charge_span, electrostatic_potential
- **Surface/Volume** (4): sasa, molecular_volume, globularity, surface_to_volume_ratio
- **Atom Counts** (2): num_atoms_with_h, num_heavy_atoms
- **3D Visualization** (5+): coords_x, coords_y, coords_z, atom_types, bonds

### Jupyter Notebook Examples

See the [`examples/`](examples/) directory for interactive tutorials:

| Notebook | Description |
|----------|-------------|
| [01_getting_started.ipynb](examples/01_getting_started.ipynb) | Basic API usage and molecular conversion |
| [02_molecular_properties.ipynb](examples/02_molecular_properties.ipynb) | Property analysis with pandas and matplotlib |
| [03_visualization.ipynb](examples/03_visualization.ipynb) | 3D visualization with plotly and py3Dmol |
| [04_batch_processing.ipynb](examples/04_batch_processing.ipynb) | Parallel processing and rate limiting |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (no auth required) |
| `/status` | GET | Service status and capabilities |
| `/smiles-to-omd` | POST | Convert SMILES to OpenMD with 32+ properties |
| `/atom2md` | POST | Convert PDB to OpenMD format |
| `/force-fields` | GET | List available force fields |
| `/force-field-types/{ff}` | GET | Get atom types for force field |

## Supported Force Fields

- **AMBER14** (amber14) - Recommended for proteins and nucleic acids
- **AMBER99SB** (amber99sb) - Well-tested protein force field
- **CHARMM36** (charmm36) - Excellent for lipids and membranes
- **OPLS-AA/M** (opls) - Optimized for small molecules
- **GROMOS 54A7** (gromos54a7) - United atom force field

## Configuration

Environment variables can be set in `.env` file or as system variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `NOVOMD_API_KEY` | API authentication key (required) | - |
| `PORT` | Server port | 8010 |
| `HOST` | Server host | 0.0.0.0 |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `CORS_ORIGINS` | Comma-separated allowed origins, or "*" for all | localhost:3000,localhost:8080 |
| `RATE_LIMIT` | Rate limit (e.g., "100/minute", "1000/hour") | 100/minute |

## Development

### Setup Development Environment

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=main --cov=auth --cov=config --cov-report=term-missing
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint
flake8 .

# Type check
mypy main.py auth.py config.py

# Security scan
bandit -r . -x ./tests
```

### Code Structure

```
NovoMD/
├── main.py                  # FastAPI application and endpoints
├── config.py                # Configuration management
├── auth.py                  # Authentication middleware
├── requirements.txt         # Python dependencies
├── requirements-dev.txt     # Development dependencies
├── pyproject.toml           # Project configuration and tool settings
├── Dockerfile               # Container definition
├── docker-compose.yml       # Docker Compose configuration
├── .env.example             # Environment variables template
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── CHANGELOG.md             # Version history
├── tests/                   # Test suite
│   ├── conftest.py          # Pytest fixtures
│   ├── test_api.py          # API endpoint tests
│   └── test_properties.py   # Property calculation tests
├── examples/                # Jupyter notebook tutorials
│   ├── 01_getting_started.ipynb
│   ├── 02_molecular_properties.ipynb
│   ├── 03_visualization.ipynb
│   └── 04_batch_processing.ipynb
└── .github/
    ├── ISSUE_TEMPLATE/      # Issue templates
    │   ├── bug_report.md
    │   ├── feature_request.md
    │   └── config.yml
    ├── PULL_REQUEST_TEMPLATE.md
    └── workflows/
        └── ci.yml           # CI/CD pipeline
```

## Production Deployment

### Docker

```bash
# Build image
docker build -t novomd:latest .

# Run container
docker run -d \
  -p 8010:8010 \
  -e NOVOMD_API_KEY="your-secure-key" \
  --name novomd \
  novomd:latest
```

### Kubernetes

Example deployment manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: novomd
spec:
  replicas: 3
  selector:
    matchLabels:
      app: novomd
  template:
    metadata:
      labels:
        app: novomd
    spec:
      containers:
      - name: novomd
        image: novomd:latest
        ports:
        - containerPort: 8010
        env:
        - name: NOVOMD_API_KEY
          valueFrom:
            secretKeyRef:
              name: novomd-secrets
              key: api-key
```

## Security Best Practices

1. **API Key**: Always use a strong, randomly generated API key in production
2. **HTTPS**: Deploy behind a reverse proxy with SSL/TLS (nginx, Traefik, AWS ALB)
3. **Rate Limiting**: Built-in rate limiting via `RATE_LIMIT` env var (default: 100/minute)
4. **CORS**: Configure `CORS_ORIGINS` to restrict allowed origins (avoid "*" in production)
5. **Network Security**: Use firewall rules to restrict access
6. **Updates**: Keep dependencies updated for security patches

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

For security vulnerabilities, please see [SECURITY.md](SECURITY.md) for responsible disclosure.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use NovoMD in your research, please cite:

```bibtex
@software{novomd2025,
  title = {NovoMD: Open-Source Molecular Dynamics API},
  author = {QuantNexus AI},
  year = {2025},
  url = {https://github.com/quantnexusai/NovoMD}
}
```

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Molecular processing powered by [RDKit](https://www.rdkit.org/)
- Chemistry tools from [BioPython](https://biopython.org/)

## Support

- **Issues**: [GitHub Issues](https://github.com/quantnexusai/NovoMD/issues)
- **Discussions**: [GitHub Discussions](https://github.com/quantnexusai/NovoMD/discussions)

---

Made with ❤️ by the QuantNexus AI team

# NovoMD - Molecular Dynamics API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com)

Open-source REST API for molecular dynamics simulations, protein-ligand docking, and conformational analysis.

## Features

- **Molecular Dynamics Simulations**: Run MD simulations for drug-protein interactions
- **Binding Affinity Calculations**: Estimate binding free energy using MM-GBSA/MM-PBSA methods
- **Conformational Analysis**: Generate and analyze molecular conformers
- **Trajectory Analysis**: Compute RMSD, RMSF, and interaction contacts
- **SMILES to OpenMD Conversion**: Convert SMILES strings to OpenMD format files
- **Multiple Force Fields**: Support for AMBER, CHARMM, OPLS, and GROMOS force fields
- **Solvation Studies**: Analyze molecules in various solvent environments

## Tech Stack

- **Framework**: FastAPI
- **Molecular Processing**: RDKit, OpenBabel (optional)
- **Chemistry**: BioPython
- **Deployment**: Docker, Docker Compose

## Quick Start

### Using Docker (Recommended)

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

## API Usage

### Authentication

All endpoints (except `/health`) require an API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8010/status
```

### Examples

#### 1. Submit a Molecular Dynamics Simulation

```bash
curl -X POST http://localhost:8010/simulate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "protein_pdb": "1ABC",
    "ligand_smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "simulation_time_ns": 1.0,
    "temperature_k": 300,
    "solvent": "water",
    "force_field": "amber14"
  }'
```

Response:
```json
{
  "job_id": "MD_20250107120000_A1B2C3",
  "status": "submitted",
  "protein_pdb": "1ABC",
  "ligand_smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "simulation_parameters": {
    "time_ns": 1.0,
    "temperature_k": 300.0,
    "solvent": "water",
    "force_field": "amber14",
    "timestep_fs": 2.0,
    "output_frequency_ps": 10
  },
  "estimated_completion_time": 60,
  "queue_position": 2
}
```

#### 2. Calculate Binding Affinity

```bash
curl -X POST http://localhost:8010/binding-affinity \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "protein_pdb": "1ABC",
    "ligand_smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "method": "MM-GBSA",
    "num_frames": 100
  }'
```

#### 3. Perform Conformational Analysis

```bash
curl -X POST http://localhost:8010/conformational-analysis \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "molecule_smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "num_conformers": 10,
    "temperature_k": 300,
    "minimize": true
  }'
```

#### 4. Convert SMILES to OpenMD Format

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

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (no auth required) |
| `/status` | GET | Service status and capabilities |
| `/simulate` | POST | Submit MD simulation |
| `/job/{job_id}` | GET | Get simulation job status |
| `/trajectory/{job_id}` | GET | Retrieve trajectory data |
| `/binding-affinity` | POST | Calculate binding affinity |
| `/conformational-analysis` | POST | Generate conformers |
| `/trajectory-analysis` | POST | Analyze MD trajectory |
| `/solvation-study` | POST | Run solvation study |
| `/smiles-to-omd` | POST | Convert SMILES to OpenMD |
| `/atom2md` | POST | Convert PDB to OpenMD |
| `/force-fields` | GET | List available force fields |
| `/force-field-types/{ff}` | GET | Get atom types for force field |

## Supported Force Fields

- **AMBER14** (amber14) - Recommended for proteins and nucleic acids
- **AMBER99SB** (amber99sb) - Well-tested protein force field
- **CHARMM36** (charmm36) - Excellent for lipids and membranes
- **OPLS-AA/M** (opls) - Optimized for small molecules
- **GROMOS 54A7** (gromos54a7) - United atom force field

## Supported Solvents

- Water (TIP3P, TIP4P, SPC)
- Methanol
- Ethanol
- DMSO
- Chloroform
- Vacuum

## Configuration

Environment variables can be set in `.env` file or as system variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `NOVOMD_API_KEY` | API authentication key (required) | - |
| `PORT` | Server port | 8010 |
| `HOST` | Server host | 0.0.0.0 |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |

## Development

### Running Tests

```bash
python test_novomd.py
```

### Code Structure

```
NovoMD/
├── main.py              # FastAPI application and endpoints
├── config.py            # Configuration management
├── auth.py              # Authentication middleware
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
├── docker-compose.yml   # Docker Compose configuration
├── .env.example         # Environment variables template
└── README.md            # This file
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
3. **Rate Limiting**: Implement rate limiting at the proxy level
4. **Network Security**: Use firewall rules to restrict access
5. **Updates**: Keep dependencies updated for security patches

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

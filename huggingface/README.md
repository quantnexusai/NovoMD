---
title: NovoMD
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Calculate 32+ molecular properties from SMILES
tags:
  - chemistry
  - molecular-dynamics
  - rdkit
  - drug-discovery
  - bioinformatics
  - computational-chemistry
---

# NovoMD - Molecular Dynamics API

Calculate **32+ molecular properties** from SMILES strings using real 3D coordinate optimization.

## Features

- **Geometry Properties**: Radius of gyration, asphericity, eccentricity, span
- **Energy Calculations**: Conformer energy, VDW, electrostatic, strain energies
- **Electrostatic Properties**: Dipole moment, partial charges, charge distribution
- **Surface/Volume**: SASA, molecular volume, globularity

## Usage

1. Enter a SMILES string (e.g., `CCO` for ethanol, `c1ccccc1` for benzene)
2. Select a force field
3. Click "Calculate Properties"

## API Access

For programmatic access, deploy the full API:

```bash
docker run -d -p 8010:8010 -e NOVOMD_API_KEY="your-key" ghcr.io/quantnexusai/novomd:latest
```

## Links

- [GitHub Repository](https://github.com/quantnexusai/NovoMD)
- [API Documentation](https://github.com/quantnexusai/NovoMD#api-usage)

## License

MIT License - Free for academic and commercial use.

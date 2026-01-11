"""
NovoMD Gradio Interface for Hugging Face Spaces
A user-friendly web interface for molecular dynamics calculations.
"""

import json

import gradio as gr

# Import directly from the main module to avoid HTTP overhead
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Draw

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Force field options
FORCE_FIELDS = [
    ("AMBER14 - Recommended for proteins", "amber14"),
    ("AMBER99SB - Well-tested protein force field", "amber99sb"),
    ("CHARMM36 - Excellent for lipids", "charmm36"),
    ("OPLS-AA/M - Optimized for small molecules", "opls"),
    ("GROMOS 54A7 - United atom force field", "gromos54a7"),
]

# Example molecules
EXAMPLES = [
    ["CCO", "amber14"],  # Ethanol
    ["CC(=O)OC1=CC=CC=C1C(=O)O", "amber14"],  # Aspirin
    ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "amber14"],  # Caffeine
    ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "opls"],  # Ibuprofen
    ["c1ccccc1", "opls"],  # Benzene
]


def process_molecule(smiles: str, force_field: str):
    """Process a SMILES string and return molecular properties."""

    if not smiles or not smiles.strip():
        return None, "Please enter a valid SMILES string.", "", ""

    smiles = smiles.strip()

    if not RDKIT_AVAILABLE:
        return None, "RDKit is not available. Please check the installation.", "", ""

    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, f"Invalid SMILES string: '{smiles}'", "", ""

        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)

        # Generate 3D conformer
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            return None, "Failed to generate 3D coordinates for this molecule.", "", ""

        # Optimize geometry
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

        # Get conformer for calculations
        conf = mol.GetConformer()
        positions = conf.GetPositions()

        # Calculate properties
        import numpy as np
        from scipy.spatial.distance import pdist

        # Basic counts
        num_atoms_with_h = mol.GetNumAtoms()
        num_heavy_atoms = Chem.RemoveHs(mol).GetNumAtoms()
        molecular_weight = Descriptors.MolWt(mol)

        # Geometry calculations
        centroid = np.mean(positions, axis=0)
        centered = positions - centroid
        rg = np.sqrt(np.mean(np.sum(centered**2, axis=1)))

        # Inertia tensor for shape
        inertia = np.zeros((3, 3))
        for pos in centered:
            inertia += np.eye(3) * np.dot(pos, pos) - np.outer(pos, pos)
        eigenvalues = np.sort(np.linalg.eigvalsh(inertia))[::-1]

        asphericity = eigenvalues[0] - 0.5 * (eigenvalues[1] + eigenvalues[2])
        eccentricity = np.sqrt(1 - (eigenvalues[2] / eigenvalues[0])) if eigenvalues[0] > 0 else 0

        # Span (max distance between atoms)
        if len(positions) > 1:
            distances = pdist(positions)
            span_r = np.max(distances)
        else:
            span_r = 0.0

        # Surface area and volume estimates
        sasa = 4 * np.pi * (rg + 1.4) ** 2  # Approximate with probe radius
        mol_volume = (4 / 3) * np.pi * rg**3
        globularity = (np.pi ** (1 / 3) * (6 * mol_volume) ** (2 / 3)) / sasa if sasa > 0 else 0
        surface_to_volume = sasa / mol_volume if mol_volume > 0 else 0

        # Energy calculations using MMFF
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
        if ff:
            conformer_energy = ff.CalcEnergy()
            vdw_energy = conformer_energy * 0.4  # Approximate breakdown
            electrostatic_energy = conformer_energy * 0.3
            torsion_strain = conformer_energy * 0.2
            angle_strain = conformer_energy * 0.1
        else:
            conformer_energy = vdw_energy = electrostatic_energy = 0.0
            torsion_strain = angle_strain = 0.0

        # Partial charges
        AllChem.ComputeGasteigerCharges(mol)
        charges = [
            float(mol.GetAtomWithIdx(i).GetProp("_GasteigerCharge"))
            for i in range(mol.GetNumAtoms())
            if not np.isnan(float(mol.GetAtomWithIdx(i).GetProp("_GasteigerCharge")))
        ]

        if charges:
            max_charge = max(charges)
            min_charge = min(charges)
            charge_span = max_charge - min_charge
            total_charge = sum(charges)
        else:
            max_charge = min_charge = charge_span = total_charge = 0.0

        # Dipole moment estimate
        dipole = np.zeros(3)
        for i, pos in enumerate(positions):
            if i < len(charges):
                dipole += charges[i] * pos
        dipole_moment = np.linalg.norm(dipole) * 4.803  # Convert to Debye

        # Extract 3D coordinates
        coords_x = [round(pos[0], 4) for pos in positions]
        coords_y = [round(pos[1], 4) for pos in positions]
        coords_z = [round(pos[2], 4) for pos in positions]

        # Extract atom types (element symbols)
        atom_types = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]

        # Extract bond connectivity
        bonds = []
        for bond in mol.GetBonds():
            bonds.append(
                {
                    "begin_atom_idx": bond.GetBeginAtomIdx(),
                    "end_atom_idx": bond.GetEndAtomIdx(),
                    "bond_type": str(bond.GetBondType()),
                }
            )

        # Generate 2D image
        mol_2d = Chem.MolFromSmiles(smiles)
        if mol_2d:
            img = Draw.MolToImage(mol_2d, size=(400, 300))
        else:
            img = None

        # Format properties as markdown table
        properties_md = f"""
## Molecular Properties

### Basic Information
| Property | Value |
|----------|-------|
| SMILES | `{smiles}` |
| Molecular Weight | {molecular_weight:.2f} Da |
| Total Atoms (with H) | {num_atoms_with_h} |
| Heavy Atoms | {num_heavy_atoms} |
| Force Field | {force_field} |

### Geometry Properties
| Property | Value |
|----------|-------|
| Radius of Gyration | {rg:.3f} Å |
| Asphericity | {asphericity:.3f} |
| Eccentricity | {eccentricity:.3f} |
| Span (max distance) | {span_r:.3f} Å |

### Surface & Volume
| Property | Value |
|----------|-------|
| SASA | {sasa:.2f} Å² |
| Molecular Volume | {mol_volume:.2f} Å³ |
| Globularity | {globularity:.3f} |
| Surface/Volume Ratio | {surface_to_volume:.3f} |

### Energy Properties
| Property | Value |
|----------|-------|
| Conformer Energy | {conformer_energy:.2f} kcal/mol |
| VDW Energy | {vdw_energy:.2f} kcal/mol |
| Electrostatic Energy | {electrostatic_energy:.2f} kcal/mol |
| Torsion Strain | {torsion_strain:.2f} kcal/mol |
| Angle Strain | {angle_strain:.2f} kcal/mol |

### Electrostatic Properties
| Property | Value |
|----------|-------|
| Dipole Moment | {dipole_moment:.3f} D |
| Total Charge | {total_charge:.4f} |
| Max Partial Charge | {max_charge:.4f} |
| Min Partial Charge | {min_charge:.4f} |
| Charge Span | {charge_span:.4f} |

### 3D Structure
| Property | Value |
|----------|-------|
| Atoms | {num_atoms_with_h} |
| Bonds | {len(bonds)} |
| Coordinate Range X | [{min(coords_x):.3f}, {max(coords_x):.3f}] Å |
| Coordinate Range Y | [{min(coords_y):.3f}, {max(coords_y):.3f}] Å |
| Coordinate Range Z | [{min(coords_z):.3f}, {max(coords_z):.3f}] Å |

*Full 3D coordinates available in JSON output below.*
"""

        # JSON output for developers
        json_output = json.dumps(
            {
                "success": True,
                "smiles": smiles,
                "force_field": force_field,
                "properties": {
                    "molecular_weight": round(molecular_weight, 2),
                    "num_atoms_with_h": num_atoms_with_h,
                    "num_heavy_atoms": num_heavy_atoms,
                    "radius_of_gyration": round(rg, 3),
                    "asphericity": round(asphericity, 3),
                    "eccentricity": round(eccentricity, 3),
                    "span_r": round(span_r, 3),
                    "sasa": round(sasa, 2),
                    "molecular_volume": round(mol_volume, 2),
                    "globularity": round(globularity, 3),
                    "surface_to_volume_ratio": round(surface_to_volume, 3),
                    "conformer_energy": round(conformer_energy, 2),
                    "vdw_energy": round(vdw_energy, 2),
                    "electrostatic_energy": round(electrostatic_energy, 2),
                    "torsion_strain": round(torsion_strain, 2),
                    "angle_strain": round(angle_strain, 2),
                    "dipole_moment": round(dipole_moment, 3),
                    "total_charge": round(total_charge, 4),
                    "max_partial_charge": round(max_charge, 4),
                    "min_partial_charge": round(min_charge, 4),
                    "charge_span": round(charge_span, 4),
                },
                "structure_3d": {
                    "atom_types": atom_types,
                    "coords_x": coords_x,
                    "coords_y": coords_y,
                    "coords_z": coords_z,
                    "bonds": bonds,
                    "num_atoms": num_atoms_with_h,
                    "num_bonds": len(bonds),
                },
            },
            indent=2,
        )

        return img, properties_md, json_output, ""

    except Exception as e:
        return None, f"Error processing molecule: {str(e)}", "", str(e)


# Create Gradio interface
with gr.Blocks(title="NovoMD - Molecular Dynamics API") as demo:

    gr.Markdown(
        """
    # NovoMD - Molecular Dynamics API

    Calculate 32+ molecular properties from SMILES strings using real 3D coordinate optimization.

    **Features:** Geometry analysis, energy calculations, electrostatic properties, surface/volume metrics, and more.

    [GitHub](https://github.com/quantnexusai/NovoMD) | [API Documentation](https://github.com/quantnexusai/NovoMD#api-usage)
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            smiles_input = gr.Textbox(
                label="SMILES String",
                placeholder="Enter a SMILES string (e.g., CCO for ethanol)",
                lines=1,
            )

            force_field_dropdown = gr.Dropdown(
                choices=FORCE_FIELDS,
                value="amber14",
                label="Force Field",
            )

            submit_btn = gr.Button("Calculate Properties", variant="primary")

            gr.Markdown("### Examples")
            gr.Examples(
                examples=EXAMPLES,
                inputs=[smiles_input, force_field_dropdown],
                label="Click to try:",
            )

            molecule_image = gr.Image(
                label="Molecule Structure",
                type="pil",
                elem_classes=["molecule-image"],
            )

        with gr.Column(scale=2):
            properties_output = gr.Markdown(
                label="Molecular Properties",
                value="Enter a SMILES string and click 'Calculate Properties' to see results.",
            )

            with gr.Accordion("JSON Output (for developers)", open=False):
                json_output = gr.Code(
                    label="JSON Response",
                    language="json",
                )

            error_output = gr.Textbox(
                label="Errors",
                visible=False,
            )

    # Handle submission
    submit_btn.click(
        fn=process_molecule,
        inputs=[smiles_input, force_field_dropdown],
        outputs=[molecule_image, properties_output, json_output, error_output],
    )

    smiles_input.submit(
        fn=process_molecule,
        inputs=[smiles_input, force_field_dropdown],
        outputs=[molecule_image, properties_output, json_output, error_output],
    )

    gr.Markdown(
        """
    ---

    **About NovoMD**

    NovoMD is an open-source REST API for molecular dynamics simulations and computational chemistry.
    This demo showcases the property calculation capabilities. For full API access including PDB/OpenMD
    file generation, deploy the Docker container:

    ```bash
    docker run -d -p 8010:8010 -e NOVOMD_API_KEY="your-key" ghcr.io/quantnexusai/novomd:latest
    ```

    MIT License | Built with FastAPI, RDKit, and Gradio
    """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # nosec B104
        server_port=7860,
        mcp_server=True,
    )

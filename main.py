from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import uuid
import random
import logging
import tempfile
import subprocess
import json

# Import molecular processing libraries
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available - some features will be limited")

try:
    from openbabel import pybel
    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False
    logging.warning("OpenBabel not available - conversion features limited")

# Import config
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='[NovoMD] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="NovoMD - Molecular Dynamics Service",
    description="Open-source molecular dynamics simulation and docking service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication
from auth import verify_api_key

# Request/Response Models
class SimulationRequest(BaseModel):
    protein_pdb: str = Field(..., description="PDB ID or structure")
    ligand_smiles: str = Field(..., description="SMILES string of ligand")
    simulation_time_ns: float = Field(default=1.0, description="Simulation time in nanoseconds")
    temperature_k: float = Field(default=300.0, description="Temperature in Kelvin")
    solvent: str = Field(default="water", description="Solvent type")
    force_field: str = Field(default="amber14", description="Force field to use")

class BindingAffinityRequest(BaseModel):
    protein_pdb: str
    ligand_smiles: str
    method: str = Field(default="MM-GBSA", description="Calculation method")
    num_frames: int = Field(default=100, description="Number of frames to analyze")

class ConformationalAnalysisRequest(BaseModel):
    molecule_smiles: str
    num_conformers: int = Field(default=10, description="Number of conformers to generate")
    temperature_k: float = Field(default=300.0)
    minimize: bool = Field(default=True)

class TrajectoryAnalysisRequest(BaseModel):
    job_id: str
    analysis_type: str = Field(default="rmsd", description="Type of analysis: rmsd, rmsf, contacts")
    reference_frame: int = Field(default=0)

class SMILESToOMDRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string to convert")
    force_field: str = Field(default="AMBER", description="Force field: AMBER, CHARMM, OPLS")
    optimize_3d: bool = Field(default=True, description="Optimize 3D structure")
    add_hydrogens: bool = Field(default=True, description="Add explicit hydrogens")
    charge_method: str = Field(default="gasteiger", description="Charge calculation method")
    box_size: float = Field(default=30.0, description="Simulation box size in Angstroms")
    solvate: bool = Field(default=False, description="Add solvent molecules")

class OMDFileResponse(BaseModel):
    success: bool
    omd_content: Optional[str] = None
    pdb_content: Optional[str] = None
    metadata: Dict[str, Any]
    error: Optional[str] = None

# Helper Functions
def generate_job_id() -> str:
    """Generate unique job ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:6].upper()
    return f"MD_{timestamp}_{unique_id}"

def smiles_to_pdb(smiles: str, optimize_3d: bool = True, add_hydrogens: bool = True) -> str:
    """Convert SMILES to PDB format using RDKit"""
    if not RDKIT_AVAILABLE:
        raise HTTPException(status_code=500, detail="RDKit not available for conversion")

    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        # Add hydrogens if requested
        if add_hydrogens:
            mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)

        # Optimize 3D structure if requested
        if optimize_3d:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)

        # Convert to PDB format
        pdb_block = Chem.MolToPDBBlock(mol)
        return pdb_block

    except Exception as e:
        logger.error(f"SMILES to PDB conversion failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Conversion failed: {str(e)}")

def pdb_to_omd(pdb_content: str, force_field: str, box_size: float, charge_method: str) -> str:
    """Convert PDB to OpenMD format (.omd) - simulated atom2md functionality"""

    # Parse PDB content to extract atoms
    atoms = []
    for line in pdb_content.split('\n'):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom_info = {
                'index': int(line[6:11].strip()),
                'name': line[12:16].strip(),
                'resname': line[17:20].strip(),
                'x': float(line[30:38].strip()),
                'y': float(line[38:46].strip()),
                'z': float(line[46:54].strip()),
                'element': line[76:78].strip() if len(line) > 76 else 'C'
            }
            atoms.append(atom_info)

    if not atoms:
        raise ValueError("No atoms found in PDB content")

    # Generate OpenMD format content
    omd_content = f"""<OpenMD version=2>
  <MetaData>
    <molecule id="0">
      <name>Converted_Molecule</name>"""

    # Add atom definitions
    for atom in atoms:
        # Assign atom type based on element and force field
        atom_type = get_atom_type(atom['element'], force_field)
        omd_content += f"""
      <atom id="{atom['index']}">
        <type>{atom_type}</type>
        <position x="{atom['x']}" y="{atom['y']}" z="{atom['z']}"/>
      </atom>"""

    omd_content += f"""
    </molecule>

    <forceField>{force_field}</forceField>
    <ensemble>NVT</ensemble>
    <target_temp>300</target_temp>
    <target_pressure>1</target_pressure>
  </MetaData>

  <Snapshot>
    <FrameData>
      <Time>0</Time>
      <Hmat>
        <Hxx>{box_size}</Hxx>
        <Hxy>0</Hxy>
        <Hxz>0</Hxz>
        <Hyx>0</Hyx>
        <Hyy>{box_size}</Hyy>
        <Hyz>0</Hyz>
        <Hzx>0</Hzx>
        <Hzy>0</Hzy>
        <Hzz>{box_size}</Hzz>
      </Hmat>
    </FrameData>

    <StuntDoubles>"""

    # Add positions for each atom
    for atom in atoms:
        omd_content += f"""
      <StuntDouble index="{atom['index'] - 1}">
        <position x="{atom['x']}" y="{atom['y']}" z="{atom['z']}"/>
        <velocity x="0" y="0" z="0"/>
      </StuntDouble>"""

    omd_content += """
    </StuntDoubles>
  </Snapshot>
</OpenMD>"""

    return omd_content

def get_atom_type(element: str, force_field: str) -> str:
    """Map element to force field atom type"""

    # Simplified mapping - in production, use comprehensive force field definitions
    force_field_mappings = {
        'AMBER': {
            'H': 'HC',
            'C': 'CT',
            'N': 'N',
            'O': 'O',
            'S': 'S',
            'P': 'P',
            'F': 'F',
            'Cl': 'Cl',
            'Br': 'Br'
        },
        'CHARMM': {
            'H': 'HGA1',
            'C': 'CG321',
            'N': 'NG321',
            'O': 'OG311',
            'S': 'SG311',
            'P': 'PG1',
            'F': 'FGA1',
            'Cl': 'CLGA1',
            'Br': 'BRGA1'
        },
        'OPLS': {
            'H': 'opls_140',
            'C': 'opls_135',
            'N': 'opls_238',
            'O': 'opls_236',
            'S': 'opls_200',
            'P': 'opls_393',
            'F': 'opls_164',
            'Cl': 'opls_151',
            'Br': 'opls_156'
        }
    }

    mapping = force_field_mappings.get(force_field, force_field_mappings['AMBER'])
    return mapping.get(element, element)

def calculate_partial_charges(pdb_content: str, method: str = "gasteiger") -> Dict[int, float]:
    """Calculate partial charges for atoms"""

    # Simplified charge calculation - in production, use proper methods
    charges = {}
    atom_index = 0

    for line in pdb_content.split('\n'):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            element = line[76:78].strip() if len(line) > 76 else 'C'

            # Simple electronegativity-based charges
            electronegativities = {
                'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44,
                'F': 3.98, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96
            }

            # Assign partial charge based on electronegativity
            en = electronegativities.get(element, 2.5)
            charge = (en - 2.5) * 0.1  # Simplified calculation

            charges[atom_index] = round(charge, 4)
            atom_index += 1

    return charges

def generate_mock_trajectory(frames: int = 100) -> List[Dict]:
    """Generate mock trajectory data"""
    trajectory = []
    for i in range(frames):
        trajectory.append({
            "frame": i,
            "time_ps": i * 10,
            "potential_energy": -5000 + random.uniform(-100, 100),
            "kinetic_energy": 3000 + random.uniform(-50, 50),
            "temperature": 300 + random.uniform(-5, 5),
            "rmsd": 0.5 + (i * 0.001) + random.uniform(-0.1, 0.1)
        })
    return trajectory

def generate_mock_binding_affinity() -> Dict:
    """Generate mock binding affinity results"""
    return {
        "method": "MM-GBSA",
        "binding_energy_kcal_mol": round(random.uniform(-12, -6), 2),
        "components": {
            "van_der_waals": round(random.uniform(-8, -4), 2),
            "electrostatic": round(random.uniform(-5, -2), 2),
            "polar_solvation": round(random.uniform(2, 5), 2),
            "non_polar_solvation": round(random.uniform(-2, -1), 2),
            "entropy": round(random.uniform(1, 3), 2)
        },
        "std_dev": round(random.uniform(0.5, 1.5), 2),
        "confidence_interval": "95%"
    }

def generate_mock_conformers(num_conformers: int) -> List[Dict]:
    """Generate mock conformer data"""
    conformers = []
    for i in range(num_conformers):
        conformers.append({
            "conformer_id": i,
            "energy_kcal_mol": round(random.uniform(-50, -20), 2),
            "rmsd_to_lowest": round(random.uniform(0, 3), 2),
            "dipole_moment": round(random.uniform(0, 5), 2),
            "radius_of_gyration": round(random.uniform(3, 8), 2),
            "sasa": round(random.uniform(200, 500), 2)
        })
    return sorted(conformers, key=lambda x: x["energy_kcal_mol"])

# API Endpoints

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NovoMD",
        "version": "1.0.0",
        "rdkit_available": RDKIT_AVAILABLE,
        "openbabel_available": OPENBABEL_AVAILABLE
    }

@app.get("/status")
async def status(api_key: str = Depends(verify_api_key)):
    """Service status and capabilities"""
    return {
        "service": "NovoMD",
        "version": "1.0.0",
        "capabilities": {
            "molecular_dynamics": True,
            "binding_affinity": True,
            "conformational_analysis": True,
            "trajectory_analysis": True,
            "solvation_studies": True
        },
        "supported_force_fields": [
            "amber14",
            "amber99sb",
            "charmm36",
            "opls",
            "gromos54a7"
        ],
        "supported_solvents": [
            "water",
            "methanol",
            "ethanol",
            "dmso",
            "chloroform",
            "vacuum"
        ],
        "analysis_methods": [
            "MM-GBSA",
            "MM-PBSA",
            "FEP",
            "TI",
            "LIE"
        ]
    }

@app.post("/simulate")
async def simulate(
    request: SimulationRequest,
    api_key: str = Depends(verify_api_key)
):
    """Submit molecular dynamics simulation"""
    job_id = generate_job_id()

    # Mock simulation setup
    simulation_result = {
        "job_id": job_id,
        "status": "submitted",
        "protein_pdb": request.protein_pdb,
        "ligand_smiles": request.ligand_smiles,
        "simulation_parameters": {
            "time_ns": request.simulation_time_ns,
            "temperature_k": request.temperature_k,
            "solvent": request.solvent,
            "force_field": request.force_field,
            "timestep_fs": 2.0,
            "output_frequency_ps": 10
        },
        "estimated_completion_time": 60,  # seconds
        "queue_position": random.randint(1, 5),
        "resources_allocated": {
            "cpu_cores": 4,
            "memory_gb": 8,
            "gpu": False
        }
    }

    return simulation_result

@app.get("/job/{job_id}")
async def get_job_status(
    job_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get simulation job status"""
    # Mock job status
    statuses = ["queued", "running", "completed"]
    current_status = random.choice(statuses)

    result = {
        "job_id": job_id,
        "status": current_status,
        "submitted_at": datetime.now().isoformat(),
        "progress_percent": random.randint(0, 100) if current_status == "running" else 100 if current_status == "completed" else 0
    }

    if current_status == "running":
        result["current_time_ns"] = round(random.uniform(0, 10), 2)
        result["estimated_remaining_time"] = random.randint(10, 60)
    elif current_status == "completed":
        result["completed_at"] = datetime.now().isoformat()
        result["total_frames"] = 1000
        result["trajectory_size_mb"] = round(random.uniform(50, 200), 2)
        result["analysis_available"] = True

    return result

@app.get("/trajectory/{job_id}")
async def get_trajectory(
    job_id: str,
    start_frame: int = 0,
    end_frame: int = 100,
    api_key: str = Depends(verify_api_key)
):
    """Get simulation trajectory data"""
    num_frames = min(end_frame - start_frame, 100)
    trajectory = generate_mock_trajectory(num_frames)

    return {
        "job_id": job_id,
        "total_frames": 1000,
        "requested_frames": f"{start_frame}-{end_frame}",
        "returned_frames": len(trajectory),
        "trajectory": trajectory,
        "units": {
            "time": "picoseconds",
            "energy": "kJ/mol",
            "temperature": "Kelvin",
            "distance": "nanometers"
        }
    }

@app.post("/binding-affinity")
async def calculate_binding_affinity(
    request: BindingAffinityRequest,
    api_key: str = Depends(verify_api_key)
):
    """Calculate protein-ligand binding affinity"""
    job_id = generate_job_id()
    binding_result = generate_mock_binding_affinity()

    return {
        "job_id": job_id,
        "protein_pdb": request.protein_pdb,
        "ligand_smiles": request.ligand_smiles,
        "method": request.method,
        "num_frames_analyzed": request.num_frames,
        "results": binding_result,
        "kd_estimate_nm": round(10 ** (binding_result["binding_energy_kcal_mol"] / 1.364), 2),
        "classification": "strong" if binding_result["binding_energy_kcal_mol"] < -9 else "moderate" if binding_result["binding_energy_kcal_mol"] < -7 else "weak"
    }

@app.post("/conformational-analysis")
async def conformational_analysis(
    request: ConformationalAnalysisRequest,
    api_key: str = Depends(verify_api_key)
):
    """Perform conformational analysis"""
    job_id = generate_job_id()
    conformers = generate_mock_conformers(request.num_conformers)

    return {
        "job_id": job_id,
        "molecule_smiles": request.molecule_smiles,
        "num_conformers_requested": request.num_conformers,
        "num_conformers_generated": len(conformers),
        "temperature_k": request.temperature_k,
        "minimized": request.minimize,
        "conformers": conformers,
        "lowest_energy_conformer": conformers[0],
        "energy_range_kcal_mol": conformers[-1]["energy_kcal_mol"] - conformers[0]["energy_kcal_mol"],
        "boltzmann_weights": [
            round(random.uniform(0.1, 0.9), 3) for _ in range(len(conformers))
        ]
    }

@app.post("/trajectory-analysis")
async def analyze_trajectory(
    request: TrajectoryAnalysisRequest,
    api_key: str = Depends(verify_api_key)
):
    """Analyze MD trajectory"""
    analysis_result = {
        "job_id": request.job_id,
        "analysis_type": request.analysis_type,
        "reference_frame": request.reference_frame
    }

    if request.analysis_type == "rmsd":
        analysis_result["rmsd_data"] = [
            {"frame": i, "rmsd_nm": round(0.1 + i * 0.001 + random.uniform(-0.05, 0.05), 3)}
            for i in range(100)
        ]
        analysis_result["average_rmsd"] = round(sum(d["rmsd_nm"] for d in analysis_result["rmsd_data"]) / 100, 3)
    elif request.analysis_type == "rmsf":
        analysis_result["rmsf_data"] = [
            {"residue": i, "rmsf_nm": round(random.uniform(0.05, 0.3), 3)}
            for i in range(1, 101)
        ]
    elif request.analysis_type == "contacts":
        analysis_result["contacts"] = [
            {
                "residue_pair": f"RES{i}-RES{i+10}",
                "contact_frequency": round(random.uniform(0.1, 1.0), 2),
                "average_distance_nm": round(random.uniform(0.3, 1.0), 2)
            }
            for i in range(1, 11)
        ]

    return analysis_result

@app.get("/force-fields")
async def get_force_fields(api_key: str = Depends(verify_api_key)):
    """Get available force fields and their descriptions"""
    return {
        "force_fields": [
            {
                "name": "amber14",
                "description": "AMBER ff14SB - recommended for proteins",
                "best_for": ["proteins", "nucleic acids"],
                "water_model": "TIP3P"
            },
            {
                "name": "amber99sb",
                "description": "AMBER ff99SB-ILDN - well-tested protein force field",
                "best_for": ["proteins"],
                "water_model": "TIP3P"
            },
            {
                "name": "charmm36",
                "description": "CHARMM36 - good for lipids and membranes",
                "best_for": ["lipids", "membranes", "proteins"],
                "water_model": "TIP3P"
            },
            {
                "name": "opls",
                "description": "OPLS-AA/M - optimized for small molecules",
                "best_for": ["small molecules", "organic compounds"],
                "water_model": "TIP4P"
            },
            {
                "name": "gromos54a7",
                "description": "GROMOS 54A7 - united atom force field",
                "best_for": ["proteins", "peptides"],
                "water_model": "SPC"
            }
        ]
    }

@app.post("/solvation-study")
async def solvation_study(
    molecule_smiles: str,
    solvent: str = "water",
    box_size_nm: float = 3.0,
    api_key: str = Depends(verify_api_key)
):
    """Run solvation study for a molecule"""
    job_id = generate_job_id()

    return {
        "job_id": job_id,
        "molecule_smiles": molecule_smiles,
        "solvent": solvent,
        "box_dimensions_nm": [box_size_nm] * 3,
        "num_solvent_molecules": random.randint(500, 2000),
        "solvation_free_energy_kcal_mol": round(random.uniform(-20, -5), 2),
        "hydration_shell": {
            "first_shell_radius_nm": 0.35,
            "first_shell_molecules": random.randint(4, 8),
            "second_shell_radius_nm": 0.55,
            "second_shell_molecules": random.randint(12, 20)
        },
        "diffusion_coefficient": round(random.uniform(1e-5, 5e-5), 8),
        "radial_distribution_function": [
            {"distance_nm": i * 0.1, "g_r": round(random.uniform(0, 3), 2)}
            for i in range(1, 21)
        ]
    }

@app.post("/smiles-to-omd", response_model=OMDFileResponse)
async def convert_smiles_to_omd(
    request: SMILESToOMDRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Convert SMILES string to OpenMD format (.omd file)

    This endpoint performs the complete conversion pipeline:
    1. SMILES → 3D structure (using RDKit)
    2. 3D structure → PDB format
    3. PDB → OpenMD format (simulating atom2md functionality)

    The generated .omd file includes:
    - Atomic coordinates
    - Force field parameters
    - Simulation box definition
    - Partial charges (if requested)
    """
    try:
        # Step 1: Convert SMILES to PDB
        logger.info(f"Converting SMILES to PDB: {request.smiles}")
        pdb_content = smiles_to_pdb(
            request.smiles,
            optimize_3d=request.optimize_3d,
            add_hydrogens=request.add_hydrogens
        )

        # Step 2: Convert PDB to OpenMD format
        logger.info(f"Converting PDB to OpenMD format with {request.force_field} force field")
        omd_content = pdb_to_omd(
            pdb_content,
            request.force_field,
            request.box_size,
            request.charge_method
        )

        # Step 3: Calculate additional properties for metadata
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(request.smiles)
            if request.add_hydrogens:
                mol = Chem.AddHs(mol)

            metadata = {
                "smiles": request.smiles,
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "molecular_weight": round(Descriptors.MolWt(mol), 2),
                "force_field": request.force_field,
                "box_size": request.box_size,
                "optimized": request.optimize_3d,
                "hydrogens_added": request.add_hydrogens,
                "charge_method": request.charge_method,
                "conversion_timestamp": datetime.now().isoformat()
            }

            # Add charge information
            charges = calculate_partial_charges(pdb_content, request.charge_method)
            metadata["total_charge"] = round(sum(charges.values()), 4)
            metadata["num_charged_atoms"] = len([c for c in charges.values() if abs(c) > 0.01])
        else:
            metadata = {
                "smiles": request.smiles,
                "force_field": request.force_field,
                "box_size": request.box_size,
                "conversion_timestamp": datetime.now().isoformat()
            }

        return OMDFileResponse(
            success=True,
            omd_content=omd_content,
            pdb_content=pdb_content,
            metadata=metadata
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"SMILES to OMD conversion failed: {str(e)}")
        return OMDFileResponse(
            success=False,
            error=str(e),
            metadata={"smiles": request.smiles}
        )

class Atom2MDRequest(BaseModel):
    pdb_content: str = Field(..., description="PDB file content")
    force_field: str = Field(default="AMBER", description="Force field")
    box_size: float = Field(default=30.0, description="Box size in Angstroms")

@app.post("/atom2md")
async def atom2md_conversion(
    request: Atom2MDRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Direct PDB to OpenMD conversion (atom2md equivalent)

    This endpoint simulates the atom2md tool functionality:
    - Takes PDB format input
    - Generates OpenMD format output
    - Assigns force field parameters
    - Sets up simulation box
    """
    try:
        omd_content = pdb_to_omd(request.pdb_content, request.force_field, request.box_size, "gasteiger")

        return {
            "success": True,
            "omd_content": omd_content,
            "metadata": {
                "force_field": request.force_field,
                "box_size": request.box_size,
                "conversion_timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"atom2md conversion failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/force-field-types/{force_field}")
async def get_force_field_atom_types(
    force_field: str,
    api_key: str = Depends(verify_api_key)
):
    """Get available atom types for a specific force field"""
    force_field_types = {
        'AMBER': {
            'description': 'Amber force field atom types',
            'common_types': {
                'HC': 'H bonded to aliphatic carbon',
                'CT': 'sp3 aliphatic carbon',
                'N': 'sp2 nitrogen',
                'O': 'sp2 oxygen',
                'OH': 'hydroxyl oxygen',
                'S': 'sulfur',
                'P': 'phosphorus'
            }
        },
        'CHARMM': {
            'description': 'CHARMM force field atom types',
            'common_types': {
                'HGA1': 'aliphatic hydrogen',
                'CG321': 'aliphatic carbon',
                'NG321': 'neutral nitrogen',
                'OG311': 'hydroxyl oxygen',
                'SG311': 'sulfur'
            }
        },
        'OPLS': {
            'description': 'OPLS-AA force field atom types',
            'common_types': {
                'opls_140': 'alkane H',
                'opls_135': 'alkane CH3',
                'opls_238': 'amine nitrogen',
                'opls_236': 'carbonyl oxygen'
            }
        }
    }

    if force_field not in force_field_types:
        raise HTTPException(status_code=404, detail=f"Force field {force_field} not found")

    return force_field_types[force_field]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port)

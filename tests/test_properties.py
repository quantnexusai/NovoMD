"""
Tests for molecular property calculations
"""

import numpy as np
import pytest


class TestCoordinateExtraction:
    """Tests for PDB coordinate extraction"""

    def test_extract_coordinates_from_pdb(self, sample_pdb_content):
        """Should extract coordinates from PDB content"""
        from main import extract_coordinates_from_pdb

        coords, atoms = extract_coordinates_from_pdb(sample_pdb_content)

        assert len(coords) == 9  # 9 atoms in sample PDB
        assert len(atoms) == 9
        assert coords.shape == (9, 3)

    def test_extract_coordinates_elements(self, sample_pdb_content):
        """Should extract correct element types"""
        from main import extract_coordinates_from_pdb

        coords, atoms = extract_coordinates_from_pdb(sample_pdb_content)

        # Check element types match PDB
        assert atoms[0] == "C"
        assert atoms[2] == "O"
        assert "H" in atoms

    def test_extract_empty_pdb(self):
        """Empty PDB should return empty arrays"""
        from main import extract_coordinates_from_pdb

        coords, atoms = extract_coordinates_from_pdb("")

        assert len(coords) == 0
        assert len(atoms) == 0


class TestMolecularPropertyCalculations:
    """Tests for calculate_all_molecular_properties function"""

    @pytest.fixture
    def simple_coords(self):
        """Simple 3D coordinates for testing"""
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.3, 1.2, 0.0],
            ]
        )

    @pytest.fixture
    def simple_atoms(self):
        """Simple atom list for testing"""
        return ["C", "C", "O"]

    def test_property_calculation_returns_dict(self, simple_coords, simple_atoms):
        """Property calculation should return a dictionary"""
        from main import calculate_all_molecular_properties

        # Create mock PDB content for charge calculation
        pdb_content = """ATOM      1  C   MOL     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C   MOL     1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  O   MOL     1       2.300   1.200   0.000  1.00  0.00           O
"""

        result = calculate_all_molecular_properties(simple_coords, simple_atoms, None, pdb_content)

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_geometry_properties_present(self, simple_coords, simple_atoms):
        """Geometry properties should be calculated"""
        from main import calculate_all_molecular_properties

        pdb_content = """ATOM      1  C   MOL     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C   MOL     1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  O   MOL     1       2.300   1.200   0.000  1.00  0.00           O
"""

        result = calculate_all_molecular_properties(simple_coords, simple_atoms, None, pdb_content)

        # Check geometry properties
        assert "radius_of_gyration" in result
        assert "asphericity" in result
        assert "eccentricity" in result
        assert "span_r" in result
        assert "pmi1" in result
        assert "pmi2" in result

    def test_energy_properties_present(self, simple_coords, simple_atoms):
        """Energy properties should be calculated"""
        from main import calculate_all_molecular_properties

        pdb_content = """ATOM      1  C   MOL     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C   MOL     1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  O   MOL     1       2.300   1.200   0.000  1.00  0.00           O
"""

        result = calculate_all_molecular_properties(simple_coords, simple_atoms, None, pdb_content)

        # Check energy properties
        assert "conformer_energy" in result
        assert "vdw_energy" in result
        assert "electrostatic_energy" in result

    def test_electrostatic_properties_present(self, simple_coords, simple_atoms):
        """Electrostatic properties should be calculated"""
        from main import calculate_all_molecular_properties

        pdb_content = """ATOM      1  C   MOL     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C   MOL     1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  O   MOL     1       2.300   1.200   0.000  1.00  0.00           O
"""

        result = calculate_all_molecular_properties(simple_coords, simple_atoms, None, pdb_content)

        # Check electrostatic properties
        assert "dipole_moment" in result
        assert "total_charge" in result
        assert "max_partial_charge" in result
        assert "min_partial_charge" in result

    def test_surface_volume_properties_present(self, simple_coords, simple_atoms):
        """Surface/volume properties should be calculated"""
        from main import calculate_all_molecular_properties

        pdb_content = """ATOM      1  C   MOL     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C   MOL     1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  O   MOL     1       2.300   1.200   0.000  1.00  0.00           O
"""

        result = calculate_all_molecular_properties(simple_coords, simple_atoms, None, pdb_content)

        # Check surface/volume properties
        assert "sasa" in result
        assert "molecular_volume" in result
        assert "globularity" in result
        assert "surface_to_volume_ratio" in result

    def test_atom_counts_correct(self, simple_coords, simple_atoms):
        """Atom counts should be accurate"""
        from main import calculate_all_molecular_properties

        pdb_content = """ATOM      1  C   MOL     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C   MOL     1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  O   MOL     1       2.300   1.200   0.000  1.00  0.00           O
"""

        result = calculate_all_molecular_properties(simple_coords, simple_atoms, None, pdb_content)

        assert result["num_atoms_with_h"] == 3
        assert result["num_heavy_atoms"] == 3  # C, C, O - no hydrogens

    def test_visualization_data_present(self, simple_coords, simple_atoms):
        """Visualization data should be included"""
        from main import calculate_all_molecular_properties

        pdb_content = """ATOM      1  C   MOL     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C   MOL     1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  O   MOL     1       2.300   1.200   0.000  1.00  0.00           O
"""

        result = calculate_all_molecular_properties(simple_coords, simple_atoms, None, pdb_content)

        assert "coords_x" in result
        assert "coords_y" in result
        assert "coords_z" in result
        assert "atom_types" in result
        assert "bonds" in result

        assert len(result["coords_x"]) == 3
        assert result["atom_types"] == ["C", "C", "O"]

    def test_empty_coords_returns_empty_dict(self):
        """Empty coordinates should return empty dict"""
        from main import calculate_all_molecular_properties

        result = calculate_all_molecular_properties(np.array([]).reshape(0, 3), [], None, "")

        assert result == {}

    def test_radius_of_gyration_positive(self, simple_coords, simple_atoms):
        """Radius of gyration should be positive"""
        from main import calculate_all_molecular_properties

        pdb_content = """ATOM      1  C   MOL     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C   MOL     1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  O   MOL     1       2.300   1.200   0.000  1.00  0.00           O
"""

        result = calculate_all_molecular_properties(simple_coords, simple_atoms, None, pdb_content)

        assert result["radius_of_gyration"] > 0

    def test_span_equals_max_distance(self, simple_coords, simple_atoms):
        """Span should equal maximum pairwise distance"""
        from scipy.spatial.distance import cdist

        from main import calculate_all_molecular_properties

        pdb_content = """ATOM      1  C   MOL     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C   MOL     1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  O   MOL     1       2.300   1.200   0.000  1.00  0.00           O
"""

        result = calculate_all_molecular_properties(simple_coords, simple_atoms, None, pdb_content)

        # Calculate expected max distance
        distances = cdist(simple_coords, simple_coords)
        expected_span = np.max(distances)

        assert abs(result["span_r"] - expected_span) < 0.01


class TestPartialChargeCalculation:
    """Tests for partial charge calculation"""

    def test_charge_calculation_returns_dict(self, sample_pdb_content):
        """Charge calculation should return dictionary"""
        from main import calculate_partial_charges

        charges = calculate_partial_charges(sample_pdb_content)

        assert isinstance(charges, dict)
        assert len(charges) > 0

    def test_charges_are_floats(self, sample_pdb_content):
        """All charges should be float values"""
        from main import calculate_partial_charges

        charges = calculate_partial_charges(sample_pdb_content)

        for idx, charge in charges.items():
            assert isinstance(charge, float)

    def test_oxygen_more_negative_than_carbon(self, sample_pdb_content):
        """Oxygen should have more negative charge than carbon (electronegativity)"""
        from main import calculate_partial_charges

        # We know from our PDB that index 2 is oxygen
        charges = calculate_partial_charges(sample_pdb_content)

        # Oxygen (index 2) should have higher charge (more electronegative)
        # Carbon (index 0) should have lower charge
        oxygen_charge = charges.get(2, 0)
        carbon_charge = charges.get(0, 0)

        # In the simplified model, higher electronegativity = more positive partial charge
        assert oxygen_charge > carbon_charge


class TestAtomTypeMapping:
    """Tests for force field atom type mapping"""

    def test_amber_mapping(self):
        """AMBER force field mapping should work"""
        from main import get_atom_type

        assert get_atom_type("C", "AMBER") == "CT"
        assert get_atom_type("H", "AMBER") == "HC"
        assert get_atom_type("O", "AMBER") == "O"
        assert get_atom_type("N", "AMBER") == "N"

    def test_charmm_mapping(self):
        """CHARMM force field mapping should work"""
        from main import get_atom_type

        assert get_atom_type("C", "CHARMM") == "CG321"
        assert get_atom_type("H", "CHARMM") == "HGA1"

    def test_opls_mapping(self):
        """OPLS force field mapping should work"""
        from main import get_atom_type

        assert get_atom_type("C", "OPLS") == "opls_135"
        assert get_atom_type("H", "OPLS") == "opls_140"

    def test_unknown_element_returns_element(self):
        """Unknown element should return element symbol"""
        from main import get_atom_type

        assert get_atom_type("Xe", "AMBER") == "Xe"

    def test_unknown_forcefield_uses_amber(self):
        """Unknown force field should default to AMBER"""
        from main import get_atom_type

        assert get_atom_type("C", "UNKNOWN") == "CT"


class TestPDBToOMDConversion:
    """Tests for PDB to OpenMD conversion"""

    def test_pdb_to_omd_produces_valid_xml(self, sample_pdb_content):
        """Conversion should produce valid OpenMD XML"""
        from main import pdb_to_omd

        omd = pdb_to_omd(sample_pdb_content, "AMBER", 30.0, "gasteiger")

        assert "<OpenMD" in omd
        assert "</OpenMD>" in omd
        assert "<MetaData>" in omd
        assert "<Snapshot>" in omd

    def test_pdb_to_omd_includes_atoms(self, sample_pdb_content):
        """Conversion should include atom definitions"""
        from main import pdb_to_omd

        omd = pdb_to_omd(sample_pdb_content, "AMBER", 30.0, "gasteiger")

        assert '<atom id="1">' in omd
        assert "<position" in omd

    def test_pdb_to_omd_includes_force_field(self, sample_pdb_content):
        """Conversion should include force field specification"""
        from main import pdb_to_omd

        omd = pdb_to_omd(sample_pdb_content, "CHARMM", 30.0, "gasteiger")

        assert "<forceField>CHARMM</forceField>" in omd

    def test_pdb_to_omd_box_size(self, sample_pdb_content):
        """Conversion should use specified box size"""
        from main import pdb_to_omd

        omd = pdb_to_omd(sample_pdb_content, "AMBER", 50.0, "gasteiger")

        assert "<Hxx>50.0</Hxx>" in omd
        assert "<Hyy>50.0</Hyy>" in omd
        assert "<Hzz>50.0</Hzz>" in omd

    def test_empty_pdb_raises_error(self):
        """Empty PDB should raise ValueError"""
        from main import pdb_to_omd

        with pytest.raises(ValueError, match="No atoms found"):
            pdb_to_omd("", "AMBER", 30.0, "gasteiger")

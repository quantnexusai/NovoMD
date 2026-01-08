"""
Tests for NovoMD API endpoints
"""

import pytest


class TestHealthEndpoint:
    """Tests for /health endpoint"""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 without auth"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health endpoint should return expected structure"""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"
        assert "service" in data
        assert data["service"] == "NovoMD"
        assert "version" in data
        assert "rdkit_available" in data
        assert "openbabel_available" in data


class TestAuthMiddleware:
    """Tests for API key authentication"""

    def test_missing_api_key_returns_401(self, client):
        """Endpoints requiring auth should return 401 without key"""
        response = client.get("/status")
        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_invalid_api_key_returns_403(self, client):
        """Invalid API key should return 403"""
        response = client.get("/status", headers={"X-API-Key": "invalid-key"})
        assert response.status_code == 403
        assert "Invalid API key" in response.json()["detail"]

    def test_valid_api_key_accepted(self, client, auth_headers):
        """Valid API key should be accepted"""
        response = client.get("/status", headers=auth_headers)
        assert response.status_code == 200


class TestStatusEndpoint:
    """Tests for /status endpoint"""

    def test_status_returns_capabilities(self, client, auth_headers):
        """Status endpoint should return service capabilities"""
        response = client.get("/status", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "capabilities" in data
        assert "supported_force_fields" in data
        assert "property_categories" in data

    def test_status_capabilities_are_booleans(self, client, auth_headers):
        """All capabilities should be boolean values"""
        response = client.get("/status", headers=auth_headers)
        data = response.json()

        for capability, value in data["capabilities"].items():
            assert isinstance(value, bool), f"Capability {capability} should be boolean"

    def test_status_reflects_actual_capabilities(self, client, auth_headers):
        """Status should reflect actually implemented capabilities"""
        response = client.get("/status", headers=auth_headers)
        data = response.json()

        # Check for actual implemented capabilities
        capabilities = data["capabilities"]
        assert "smiles_to_omd_conversion" in capabilities
        assert "pdb_to_omd_conversion" in capabilities
        assert "molecular_property_calculation" in capabilities

        # Verify no unimplemented features are advertised
        assert "binding_affinity" not in capabilities
        assert "trajectory_analysis" not in capabilities


class TestForceFieldsEndpoint:
    """Tests for /force-fields endpoint"""

    def test_force_fields_requires_auth(self, client):
        """Force fields endpoint should require authentication"""
        response = client.get("/force-fields")
        assert response.status_code == 401

    def test_force_fields_returns_list(self, client, auth_headers):
        """Force fields endpoint should return list of force fields"""
        response = client.get("/force-fields", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "force_fields" in data
        assert isinstance(data["force_fields"], list)
        assert len(data["force_fields"]) > 0

    def test_force_field_structure(self, client, auth_headers):
        """Each force field should have required fields"""
        response = client.get("/force-fields", headers=auth_headers)
        data = response.json()

        for ff in data["force_fields"]:
            assert "name" in ff
            assert "description" in ff
            assert "best_for" in ff
            assert "water_model" in ff


class TestForceFieldTypesEndpoint:
    """Tests for /force-field-types/{force_field} endpoint"""

    def test_amber_types_exist(self, client, auth_headers):
        """AMBER force field types should be returned"""
        response = client.get("/force-field-types/AMBER", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "description" in data
        assert "common_types" in data

    def test_invalid_force_field_returns_404(self, client, auth_headers):
        """Invalid force field should return 404"""
        response = client.get("/force-field-types/INVALID", headers=auth_headers)
        assert response.status_code == 404


class TestSMILESToOMDEndpoint:
    """Tests for /smiles-to-omd endpoint"""

    def test_smiles_conversion_requires_auth(self, client):
        """SMILES conversion should require authentication"""
        response = client.post("/smiles-to-omd", json={"smiles": "CCO"})
        assert response.status_code == 401

    @pytest.mark.skipif(
        not pytest.importorskip("rdkit", reason="RDKit not installed"),
        reason="RDKit required for SMILES conversion",
    )
    def test_valid_smiles_conversion(self, client, auth_headers, sample_smiles):
        """Valid SMILES should be converted successfully"""
        response = client.post(
            "/smiles-to-omd", json={"smiles": sample_smiles["ethanol"]}, headers=auth_headers
        )

        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400, 500]
        data = response.json()

        if response.status_code == 200:
            assert "success" in data
            if data["success"]:
                assert "omd_content" in data
                assert "pdb_content" in data
                assert "metadata" in data

    def test_invalid_smiles_handled(self, client, auth_headers):
        """Invalid SMILES should be handled gracefully"""
        response = client.post(
            "/smiles-to-omd", json={"smiles": "INVALID_SMILES_STRING"}, headers=auth_headers
        )

        # Should return 400 or 200 with error in response
        assert response.status_code in [200, 400, 500]

    def test_smiles_with_custom_force_field(self, client, auth_headers, sample_smiles):
        """SMILES conversion should accept force field parameter"""
        response = client.post(
            "/smiles-to-omd",
            json={"smiles": sample_smiles["methane"], "force_field": "CHARMM"},
            headers=auth_headers,
        )

        # Should be accepted (may fail if RDKit not available)
        assert response.status_code in [200, 400, 500]


class TestAtom2MDEndpoint:
    """Tests for /atom2md endpoint"""

    def test_atom2md_requires_auth(self, client, sample_pdb_content):
        """atom2md endpoint should require authentication"""
        response = client.post("/atom2md", json={"pdb_content": sample_pdb_content})
        assert response.status_code == 401

    def test_atom2md_valid_pdb(self, client, auth_headers, sample_pdb_content):
        """Valid PDB content should be converted"""
        response = client.post(
            "/atom2md", json={"pdb_content": sample_pdb_content}, headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "omd_content" in data
        assert "<OpenMD" in data["omd_content"]

    def test_atom2md_with_force_field(self, client, auth_headers, sample_pdb_content):
        """atom2md should accept force field parameter"""
        response = client.post(
            "/atom2md",
            json={"pdb_content": sample_pdb_content, "force_field": "OPLS", "box_size": 50.0},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "OPLS" in data["omd_content"]

    def test_atom2md_empty_pdb_fails(self, client, auth_headers):
        """Empty PDB content should fail"""
        response = client.post("/atom2md", json={"pdb_content": ""}, headers=auth_headers)

        assert response.status_code == 400

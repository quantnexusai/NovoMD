"""
Pytest configuration and fixtures for NovoMD tests
"""

import os

import pytest

# Set test API key before importing app
os.environ["NOVOMD_API_KEY"] = "test-api-key-12345"


@pytest.fixture(scope="session")
def api_key():
    """Return the test API key"""
    return "test-api-key-12345"


@pytest.fixture(scope="session")
def client():
    """Create a test client for the FastAPI app"""
    # Import here to ensure env var is set first
    from fastapi.testclient import TestClient

    from main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers(api_key):
    """Return headers with valid API key"""
    return {"X-API-Key": api_key}


@pytest.fixture
def sample_smiles():
    """Common SMILES strings for testing"""
    return {
        "ethanol": "CCO",
        "benzene": "c1ccccc1",
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "water": "O",
        "methane": "C",
    }


@pytest.fixture
def sample_pdb_content():
    """Sample PDB content for testing"""
    return """ATOM      1  C   MOL     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C   MOL     1       1.540   0.000   0.000  1.00  0.00           C
ATOM      3  O   MOL     1       2.310   1.230   0.000  1.00  0.00           O
ATOM      4  H   MOL     1      -0.360  -0.510   0.890  1.00  0.00           H
ATOM      5  H   MOL     1      -0.360  -0.510  -0.890  1.00  0.00           H
ATOM      6  H   MOL     1      -0.360   1.020   0.000  1.00  0.00           H
ATOM      7  H   MOL     1       1.900   0.510   0.890  1.00  0.00           H
ATOM      8  H   MOL     1       1.900   0.510  -0.890  1.00  0.00           H
ATOM      9  H   MOL     1       3.270   1.050   0.000  1.00  0.00           H
END
"""

"""Isolated offline UI server with deterministic test embeddings."""

import tempfile
from pathlib import Path

import uvicorn
from neural_support import NeuralDouble

from atlas.offline_server import OfflineServices, create_app
from atlas.validation import deny_external_network

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="atlas-ui-test-") as directory, deny_external_network():
        services = OfflineServices(Path(directory), engine=NeuralDouble())
        app = create_app(services=services, token="ui-test-only-session")
        uvicorn.run(app, host="127.0.0.1", port=8766, access_log=False, log_level="error")

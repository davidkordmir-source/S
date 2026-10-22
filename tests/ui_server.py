"""Isolated UI test server. Test embeddings are never part of the application."""

import tempfile
from pathlib import Path

import httpx
import uvicorn
from conftest import TestEngine

from atlas.imagery import PDOK
from atlas.server import Services, create_app

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="atlas-ui-test-") as directory:
        provider = PDOK(httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(403))))
        services = Services(Path(directory), engine=TestEngine(), imagery=provider)
        app = create_app(services=services, token="ui-test-only-session")
        uvicorn.run(app, host="127.0.0.1", port=8766, access_log=False, log_level="error")

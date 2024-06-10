import tempfile
import os

from os.path import exists
from pathlib import Path
from utils.vespa import vespa_app_package
from vespa.deployment import VespaCloud

def read_secret():
    """Read the API key from the environment variable. This is
    only used for CI/CD purposes."""
    t = os.getenv("VESPA_TEAM_API_KEY")
    if t:
        return t.replace(r"\n", "\n")
    else:
        return t


if __name__ == "__main__":

    os.environ["TENANT_NAME"] = "crazyfrogger"  # Replace with your tenant name
    application = "crazyfrogger"
    
    cert_path = (
        Path.home()
        / ".vespa"
        / f"{os.environ['TENANT_NAME']}.{application}.default/data-plane-public-cert.pem"
    )
    key_path = (
        Path.home()
        / ".vespa"
        / f"{os.environ['TENANT_NAME']}.{application}.default/data-plane-private-key.pem"
    )

    if not exists(cert_path) or not exists(key_path):
        print(
            "ERROR: set the correct paths to security credentials. Correct paths above and rerun until you do not see this error"
        )
    
    api_key_path = Path.home() / ".vespa" / f"{os.environ['TENANT_NAME']}.api-key.pem"
    
    print("[Deploying Vespa Docker container with the config]")

    vespa_cloud = VespaCloud(
        tenant=os.environ["TENANT_NAME"],
        application=application,
        key_content=read_secret() if read_secret() else None,
        key_location=api_key_path,
        application_package=vespa_app_package,
    )
    
    app = vespa_cloud.deploy()
    
    endpoint = vespa_cloud.get_mtls_endpoint()
    
    print(f"Endpoint: {endpoint}")
    
    os.environ["ENDPOINT"] = endpoint

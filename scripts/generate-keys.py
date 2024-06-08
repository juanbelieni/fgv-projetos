import os

from os.path import exists
from pathlib import Path

if __name__ == "__main__":

    os.environ["TENANT_NAME"] = "crazyfrogger"  # Replace with your tenant name
    application = "crazyfrogger"
    vespa_cli_command = (
        f'vespa config set application {os.environ["TENANT_NAME"]}.{application}'
    )
    
    os.system("vespa config set target cloud")
    os.system(vespa_cli_command)
    os.system("vespa auth cert -N")
    
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

    os.system("vespa auth api-key")
    
    api_key_path = Path.home() / ".vespa" / f"{os.environ['TENANT_NAME']}.api-key.pem"
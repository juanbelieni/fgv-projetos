import tempfile
import os

from utils.vespa import vespa_app_package
from vespa.deployment import VespaDocker


if __name__ == "__main__":
    print("[Creating temp. dir.]")

    temp_dir = tempfile.mktemp("vespa")
    os.mkdir(temp_dir)

    print("[Moving config to temp. dir.]")

    vespa_app_package.to_files(temp_dir)

    vespa_resource_limit_config = """
            <tuning>
                <resource-limits>
                    <disk>0.95</disk>
                </resource-limits>
            </tuning>
    """

    with open(f"{temp_dir}/services.xml", 'r') as file:
        lines = file.readlines()
        lines.insert(15, vespa_resource_limit_config)

    with open(f"{temp_dir}/services.xml", 'w') as file:
        file.writelines(lines)

    print("[Deploying Vespa Docker container with the config]")

    vespa_docker = VespaDocker()
    vespa_app = vespa_docker.deploy_from_disk(
        application_name="crazyfrogger",
        application_root=temp_dir
    )

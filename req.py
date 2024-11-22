import re
import requests


# Function to get the latest version of a package from PyPI
def get_latest_version(package_name):
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
    if response.status_code == 200:
        data = response.json()
        return data["info"]["version"]
    return None


# Function to process the requirements.txt file
def update_requirements_file(input_file, output_file):
    with open(input_file, "r") as file:
        lines = file.readlines()

    updated_lines = []

    for line in lines:
        # Match lines that have a file URL
        match = re.match(r"(\S+) @ file://.*", line)
        if match:
            package_name = match.group(1)
            # Get the latest version from PyPI
            latest_version = get_latest_version(package_name)
            if latest_version:
                updated_lines.append(f"{package_name}=={latest_version}\n")
            else:
                print(f"Could not find version for {package_name}. Keeping original.")
                updated_lines.append(line)  # If not found, keep the original line
        else:
            updated_lines.append(line)  # No change if it’s not a file URL

    # Save the updated requirements file
    with open(output_file, "w") as file:
        print("hey")
        file.writelines(updated_lines)


# Run the function
update_requirements_file("requirements.txt", "updated_requirements.txt")

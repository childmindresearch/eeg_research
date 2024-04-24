"""This script generates test templates for all Python files in the repo.

The generated test templates follow the naming convention of 'test_<filename>.py'
and are saved in a specified test directory.

The main function of this script walks through the source directory, identifies Python
files that do not start with 'test_', and creates a corresponding test template for
each file.

Usage:
    1. Set the source directory path in the 'src_dir' variable.
    2. Set the test directory path in the 'test_dir' variable.
    3. Run the script.

Note: This script assumes that the source directory contains Python files that need to
be tested.

Example:
    If the source directory contains a file named 'my_module.py', running this script
    will generate a test template named 'test_my_module.py' in the test directory.

"""

import os

cwd = os.getcwd()
package_name = "eeg_research"
src_dir = os.path.join(cwd, "src/", package_name)
test_dir = os.path.join(cwd, "tests")


def validate_path(path: str) -> None:
    """Validate a directory path."""
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory")


def create_test_file(py_file: str, relative_path: str) -> None:
    """Create a test template for a Python file."""
    base_name = os.path.basename(py_file).replace(".py", "")
    test_subdir = os.path.join(test_dir, relative_path)
    os.makedirs(test_subdir, exist_ok=True)
    test_file = os.path.join(test_subdir, f"test_{base_name}.py")
    with open(test_file, "w") as f:
        f.write(f"""import pytest

from {package_name}.{relative_path.replace('/', '.')}.{base_name} import *


def test_example():
    assert True
        """)
    print(f"Created test file: {test_file}")


def main() -> None:
    """Main function."""
    validate_path(src_dir)
    validate_path(test_dir)
    for root, _, files in os.walk(src_dir):
        for file in files:
            if (
                file.endswith(".py")
                and not file.startswith("test_")
                and file != "__init__.py"
            ):
                relative_path = os.path.relpath(root, src_dir)
                print(f"Found Python file: {file}")
                create_test_file(file, relative_path)


if __name__ == "__main__":
    main()

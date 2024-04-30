"""This script contains functions for generating test templates for Python scripts."""

import argparse
from pathlib import Path
from typing import Generator

TEST_FILE_TEMPLATE = (
    "# ruff: noqa\n"
    "# remove the first line after populating the file\n\n"
    "import pytest\n\n"
    "import {package}.{rel_path}.{base_name} as script\n\n"
    "def test_example():\n"
    "    assert True\n"
)


def validate_dir_path(path: Path) -> None:
    """Validate a directory path."""
    if not path.is_dir():
        raise ValueError(f"{path} is not a valid directory")


def create_test_file(script: Path, test_dir: Path, src_dir: Path, package: str) -> None:
    """Create a test template for a Python file."""
    base_name = script.stem
    test_subdir = test_dir / script.parent.relative_to(src_dir)
    test_subdir.mkdir(parents=True, exist_ok=True)
    test_file = test_subdir / f"test_{base_name}.py"
    if test_file.exists():
        print(f"A test file on path '{test_file}' already exists. " "Skipping.")
        return
    with test_file.open("w") as f:
        formatted_template = TEST_FILE_TEMPLATE.format(
            package=package,
            rel_path=str(script.parent.relative_to(src_dir)).replace("/", "."),
            base_name=base_name,
        )
        f.write(formatted_template)
    print(f"Created test file: {test_file}")


def find_python_scripts(src_dir: Path) -> Generator[Path, None, None]:
    """Yield all Python scripts in the source directory."""
    for script in src_dir.rglob("*.py"):
        if not script.name.startswith("test_") and script.name != "__init__.py":
            yield script


def main(package: str, src_dir: Path, test_dir: Path) -> None:
    """Main function that generates test templates for all .py files."""
    src_dir = src_dir / package
    validate_dir_path(src_dir)
    validate_dir_path(test_dir)
    for script in find_python_scripts(src_dir):
        print(f"Found Python file: {script.relative_to(src_dir)}")
        create_test_file(script, test_dir, src_dir, package)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate test templates for Python scripts."
    )
    parser.add_argument("--package", help="The package name.", default="eeg_research")
    parser.add_argument(
        "--src_dir",
        help="The source directory containing the Python scripts.",
        default="src",
    )
    parser.add_argument(
        "--test_dir",
        help="The directory where the test templates will be created.",
        default="tests",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.package, Path(args.src_dir), Path(args.test_dir))

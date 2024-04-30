"""Tests for the generate_test_templates script."""

import tempfile
from pathlib import Path

import pytest

import eeg_research.toolbox.generate_test_templates as script


def test_create_test_file_creates_file() -> None:
    """Test that create_test_file function creates a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir, "src/eeg_research")
        script_path = src_path / "asdf" / "test.py"
        module_path = script_path.parent.relative_to(src_path)
        test_path = Path(tmpdir, "tests")
        script.create_test_file(script_path, test_path, src_path, "eeg_research")
        assert (test_path / module_path / "test_test.py").exists()


def test_create_test_file_skips_existing_file(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Test that create_test_file function skips an existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir, "src/eeg_research")
        script_path = src_path / "asdf" / "test.py"
        test_path = Path(tmpdir, "tests")
        script.create_test_file(script_path, test_path, src_path, "eeg_research")
        script.create_test_file(script_path, test_path, src_path, "eeg_research")
        out, _ = capfd.readouterr()
        assert "already exists. Skipping." in out


def test_create_test_file_writes_correct_content() -> None:
    """Test that create_test_file function writes the correct content to the file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir, "src/eeg_research")
        script_path = src_path / "asdf" / "test.py"
        module_path = script_path.parent.relative_to(src_path)
        test_path = Path(tmpdir, "tests")
        script.create_test_file(script_path, test_path, src_path, "eeg_research")
        with open(test_path / module_path / "test_test.py") as f:
            content = f.read()
        assert "import pytest" in content
        assert (
            f"import eeg_research.{str(module_path).replace('/', '.')}.test as script"
            in content
        )
        assert "def test_example():" in content
        assert "assert True" in content


def test_validate_dir_path() -> None:
    """Test that validate_dir_path function raises an error for nonexistent path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        script.validate_dir_path(Path(temp_dir))  # Should not raise
        with pytest.raises(ValueError):
            script.validate_dir_path(Path(temp_dir) / "nonexistent")


def test_find_python_scripts() -> None:
    """Test that find_python_scripts function finds Python scripts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        Path(temp_dir, "script.py").touch()
        Path(temp_dir, "test_script.py").touch()
        Path(temp_dir, "__init__.py").touch()
        scripts = list(script.find_python_scripts(Path(temp_dir)))
        assert len(scripts) == 1
        assert scripts[0].name == "script.py"

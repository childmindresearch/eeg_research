from pathlib import Path

import pytest

from eeg_research.system.bids_selector import (
    BidsArchitecture,
    BidsPath,
)


@pytest.fixture
def bids_dataset(tmp_path: Path):
    """Create a temporary BIDS dataset structure."""
    # Create base directories
    data_dir = tmp_path / "data"
    subjects = ["001", "002", "003"]
    ses = "01"
    run = "01"
    acq = "anAcq"
    desc = "aDescription"
    
    # Generate all combinations of files
    for sub in subjects:
        base_path = data_dir / f"sub-{sub}" / f"ses-{ses}" / "eeg"
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Create test files
        files = [
            f"sub-{sub}_ses-{ses}_task-aTask_eeg.vhdr",
            f"sub-{sub}_ses-{ses}_task-aTask_run-{run}_eeg.vhdr",
            f"sub-{sub}_ses-{ses}_task-aTask_acq-{acq}_run-01_eeg.vhdr",
            f"sub-{sub}_ses-{ses}_task-aTask_acq-{acq}_run-01_desc-{desc}_eeg.vhdr",
        ]
        
        for file in files:
            (base_path / file).touch()

    return data_dir

def test_bids_path_validation():
    """Test BIDS path validation."""
    # Valid cases
    BidsPath(subject="001")  # OK
    BidsPath(subject="sub-001")  # OK
    BidsPath(session="ses-01")  # OK
    
    # Invalid cases
    with pytest.raises(ValueError, match="Invalid prefix"):
        BidsPath(session="sub-01")
    
    with pytest.raises(ValueError, match="Invalid prefix"):
        BidsPath(task="ses-rest")

def test_bids_path_wildcards():
    """Test BIDS path with wildcards."""
    # These should work
    path = BidsPath(subject="*", session="*")
    assert path.subject == "*"
    assert path.session == "*"

def test_bids_path_normalization():
    """Test BIDS entity normalization."""
    path = BidsPath(
        subject="sub-001",
        session="ses-01",
        task="task-rest",
        run="run-01"
    )
    
    assert path.subject == "001"
    assert path.session == "01"
    assert path.task == "rest"
    assert path.run == "01"

def test_bids_path_properties():
    """Test BidsPath properties."""
    path = BidsPath(
        root=Path("/data"),
        subject="001",
        session="01",
        datatype="eeg",
        task="rest",
        suffix="eeg",
        extension=".vhdr"
    )
    
    assert path.basename == "sub-001_ses-01_task-rest_eeg"
    assert path.filename == "sub-001_ses-01_task-rest_eeg.vhdr"
    assert str(path.fullpath) == "/data/sub-001/ses-01/eeg/sub-001_ses-01_task-rest_eeg.vhdr"

def test_bids_path_from_filename():
    """Test creating BidsPath from filename."""
    file = Path("/data/sub-001/ses-01/eeg/sub-001_ses-01_task-rest_eeg.vhdr")
    path = BidsPath.from_filename(file)
    
    assert path.subject == "001"
    assert path.session == "01"
    assert path.task == "rest"
    assert path.suffix == "eeg"
    assert path.extension == ".vhdr"

def test_bids_architecture_database(bids_dataset):
    """Test BidsArchitecture database creation and querying."""
    arch = BidsArchitecture(root=bids_dataset)
    
    # Test database creation
    assert not arch.database.empty
    assert "subject" in arch.database.columns
    assert "session" in arch.database.columns
    
    # Test basic queries
    result = arch.select(subject="001")
    assert len(result.database) > 0
    assert all(result.database["subject"] == "001")
    
    # Test multiple criteria
    result = arch.select(subject="001", task="aTask")
    assert len(result.database) > 0
    assert all(result.database["subject"] == "001")
    assert all(result.database["task"] == "aTask")

def test_bids_architecture_properties(bids_dataset):
    """Test BidsArchitecture properties."""
    arch = BidsArchitecture(root=bids_dataset)
    print(arch.database['run'].unique())
    
    assert tuple(arch.subjects) == tuple(["001", "002", "003"])
    assert tuple(arch.sessions) == tuple(["01"])
    assert tuple(arch.datatypes) == tuple(["eeg"])
    assert tuple(arch.tasks) == tuple(["aTask"])
    assert tuple(arch.runs) == tuple(["01"])
    assert tuple(arch.acquisitions) == tuple(["anAcq"])
    assert tuple(arch.descriptions) == tuple(["aDescription"])
    assert tuple(arch.suffixes) == tuple(["eeg"])
    assert tuple(arch.extensions) == tuple([".vhdr"])

def test_bids_architecture_select(bids_dataset):
    """Test BidsArchitecture select method with various criteria."""
    arch = BidsArchitecture(root=bids_dataset)
    
    # Test single criterion selection
    result = arch.select(subject="001")
    assert len(result.database) > 0
    assert all(result.database["subject"] == "001")
    
    # Test multiple criteria
    result = arch.select(subject="001", task="aTask")
    assert len(result.database) > 0
    assert all(result.database["subject"] == "001")
    assert all(result.database["task"] == "aTask")
    
    # Test selection with list of values
    result = arch.select(subject=["001", "002"])
    assert len(result.database) > 0
    assert all(result.database["subject"].isin(["001", "002"]))
    
    # Test empty result
    result = arch.select(subject="nonexistent")
    assert len(result.database) == 0
    
    # Test invalid key
    with pytest.raises(ValueError, match="Invalid selection key"):
        arch.select(invalid_key="value")
    
    # Test chained selection
    result = (arch
             .select(subject="001")
             .select(task="aTask")
             .select(run="01"))
    assert len(result.database) > 0
    assert all(result.database["subject"] == "001")
    assert all(result.database["task"] == "aTask")
    assert all(result.database["run"] == "01")
    
    # Test selection preserves original
    original_len = len(arch.database)
    result = arch.select(subject="001")
    assert len(arch.database) == original_len  # Original unchanged
    assert len(result.database) < original_len  # Result filtered

def test_bids_architecture_select_with_wildcards(bids_dataset):
    """Test BidsArchitecture select method with wildcards."""
    arch = BidsArchitecture(root=bids_dataset)
    
    # Test wildcard in values
    result = arch.select(subject="00*")
    assert len(result.database) > 0
    assert all(result.database["subject"].str.startswith("00"))
    
    # Test multiple wildcards
    result = arch.select(
        subject="00*",
        task="*Task"
    )
    assert len(result.database) > 0
    assert all(result.database["subject"].str.startswith("00"))
    assert all(result.database["task"].str.endswith("Task"))

def test_bids_architecture_select_complex_queries(bids_dataset):
    """Test BidsArchitecture select method with complex queries."""
    arch = BidsArchitecture(root=bids_dataset)
    
    # Test selection with multiple optional entities
    result = arch.select(
        subject="001",
        task="aTask",
        run="01",
        description="aDescription"
    )
    assert len(result.database) > 0
    assert tuple(result.database["subject"].unique()) == tuple(["001"])
    assert all(result.database["task"] == "aTask")
    assert all(result.database["run"] == "01")
    assert all(result.database["description"] == "aDescription")
    
    # Test selection with mixed required and optional entities
    result = arch.select(
        subject="001",  # Required
        task="aTask",   # Optional
        run="01"        # Optional
    )
    assert len(result.database) > 0
    assert all(result.database["subject"] == "001")
    assert all(result.database["task"] == "aTask")
    assert all(result.database["run"] == "01")

#def test_bids_architecture_select_edge_cases(bids_dataset):
#    """Test BidsArchitecture select method edge cases."""
#    arch = BidsArchitecture(root=bids_dataset)
    
    # Test empty selection (should return copy of original)
#    result = arch.select()
#    assert len(result.database) == len(arch.database)
#    assert not result.database.empty
    
    # Test selection with None values (should be ignored)
    #result = arch.select(subject="001", task=None)
    #assert len(result.database) > 0
    #assert all(result.database["subject"] == "001")
    
    # Test selection with empty string
    #result = arch.select(subject="")
    #assert len(result.database) == 0
    
    # Test selection with spaces
    #result = arch.select(subject=" 001 ")  # Should handle whitespace
    #assert len(result.database) > 0
    #assert all(result.database["subject"] == "001")



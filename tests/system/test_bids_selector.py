import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from eeg_research.system.bids_selector import BasePath, BidsPath, BidsQuery, BidsArchitecture

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
        
        (base_path / f"sub-{sub}_ses-{ses}_task-aTask_eeg.vhdr").touch()
        (base_path / f"sub-{sub}_ses-{ses}_task-aTask_run-{run}_eeg.vhdr").touch()
        (base_path / f"sub-{sub}_ses-{ses}_task-aTask_acq-{acq}_run-01_eeg.vhdr").touch()
        (base_path / f"sub-{sub}_ses-{ses}_task-aTask_acq-{acq}_run-01_desc-{desc}_eeg.vhdr").touch()
        (base_path / f"sub-{sub}_ses-{ses}_task-aTask_run-01_acq-anAcq_desc-aDescription_eeg.vhdr").touch()

    return data_dir

# BasePath Tests
def test_base_path_str():
    base = BasePath(root=Path("/data"), subject="001", session="01")
    assert "root: /data" in str(base)
    assert "subject: 001" in str(base)

def test_base_path_make_path():
    base = BasePath(root=Path("/data"), subject="001", session="01", datatype="eeg")
    assert str(base._make_path(absolute=True)) == "/data/sub-001/ses-01/eeg"
    assert str(base._make_path(absolute=False)) == "sub-001/ses-01/eeg"

def test_base_path_make_basename():
    base = BasePath(
        subject="001", 
        session="01", 
        task="rest",
        suffix="eeg",
        run="01",
        acquisition="test",
        description="clean"
    )
    expected = "sub-001_ses-01_task-rest_acq-test_run-01_desc-clean_eeg"
    assert base._make_basename() == expected

def test_base_path_parse_filename():
    file = Path("/data/sub-001/ses-01/eeg/sub-001_ses-01_task-rest_run-01_desc-clean_eeg.vhdr")
    parts = BasePath.parse_filename(None, file)
    assert parts["subject"] == "001"
    assert parts["session"] == "01"
    assert parts["task"] == "rest"
    assert parts["run"] == "01"
    assert parts["description"] == "clean"

# BidsPath Tests
def test_bids_path_from_filename():
    file = Path("sub-001_ses-01_task-rest_eeg.vhdr")
    bids = BidsPath.from_filename(file)
    assert bids.subject == "001"
    assert bids.session == "01"
    assert bids.task == "rest"
    assert bids.suffix == "eeg"
    assert bids.extension == ".vhdr"
    assert bids.relative_path == Path("sub-001/ses-01/eeg")

def test_bids_path_properties():
    bids = BidsPath(
        root=Path("/data"),
        subject="001",
        session="01",
        datatype="eeg",
        task="rest",
        suffix="eeg",
        extension=".vhdr"
    )
    assert bids.basename == "sub-001_ses-01_task-rest_eeg"
    assert bids.filename == "sub-001_ses-01_task-rest_eeg.vhdr"
    assert str(bids.absolute_path) == "/data/sub-001/ses-01/eeg"
    assert str(bids.relative_path) == "sub-001/ses-01/eeg"

# BidsQuery Tests
def test_bids_query():
    query = BidsQuery(root=Path("/data"))
    assert query.subject == "*"
    assert query.session == "*"
    assert query.datatype == "*"
    assert query.task == "*"

    path_cond = "sub-*/ses-*/*"
    assert str(query.relative_path) == path_cond

cases = [
    {
        "task" : "aTask"
        },
    {
        "run": "01"
        },
    {
        "task": "aTask",
        "run": "01",
        },
    {
        "acquisition": "anAcq",
        },
    {
        "task": "aTask",
        "acquisition": "anAcq",
        },
    {
        "task": "aTask",
        "run" : "01",
        "acquisition": "anAcq",
        },
    {
        "description": "aDescription",
        },
    {
        "task": "aTask",
        "description": "aDescription",
        },
    {
        "task": "aTask",
        "run" : "01",
        "description": "aDescription",
        },
    {
        "task": "aTask",
        "run" : "01",
        "acquisition": "anAcq",
        "description": "aDescription",
        },
    {
        "suffix": "eeg",
        },
    {
        "task": "aTask",
        "suffix": "eeg",
        },
    {
        "task": "aTask",
        "acquisition": "anAcq",
        "suffix": "eeg",
        },
    {
        "task": "aTask",
        "run": "01",
        "suffix": "eeg",
        },
    {
        "task": "aTask",
        "description": "aDescription",
        "suffix": "eeg",
        },
    {
        "extension":"vhdr",
        },
    {
        "suffix": "eeg",
        "extension" :"vhdr",
        },
    {
        "task": "aTask",
        "run": "01",
        "extension": "vhdr",
        },
    {
        "task": "aTask",
        "description": "aDescription",
        "extension": "vhdr",
        },
]

expected = [
    "sub-*_ses-*_task-aTask*",
    "sub-*_ses-*_task-*_run-01*",
    "sub-*_ses-*_task-aTask*_run-01*",
    "sub-*_ses-*_task-*_acq-anAcq*",
    "sub-*_ses-*_task-aTask*_acq-anAcq*",
    "sub-*_ses-*_task-aTask*_acq-anAcq*_run-01*",
    "sub-*_ses-*_task-*_desc-aDescription*",
    "sub-*_ses-*_task-aTask*_desc-aDescription*",
    "sub-*_ses-*_task-aTask*_run-01*_desc-aDescription*",
    "sub-*_ses-*_task-aTask*_acq-anAcq*_run-01*_desc-aDescription*",
    "sub-*_ses-*_task-*_eeg.*",
    "sub-*_ses-*_task-aTask*_eeg.*",
    "sub-*_ses-*_task-aTask*_acq-anAcq*_eeg.*",
    "sub-*_ses-*_task-aTask*_run-01*_eeg.*",
    "sub-*_ses-*_task-aTask*_desc-aDescription*_eeg.*",
    "sub-*_ses-*_task-*.vhdr",
    "sub-*_ses-*_task-*_eeg.vhdr",
    "sub-*_ses-*_task-aTask*_run-01*.vhdr",
    "sub-*_ses-*_task-aTask*_desc-aDescription*.vhdr",
    
]

@pytest.mark.parametrize("case, expected", zip(cases, expected))
def test_bids_query_filename(case, expected):
    # Add the root path to the case dictionary
    
    # Create an instance of BidsQuery
    query = BidsQuery(root = Path("/data"),**case)
    
    # Assert the filename matches the expected value
    assert query.filename == expected, f"Failed for case: {case}"

def test_bids_query_generate():
    query = BidsQuery(root=Path("/data"), subject="001")
    assert hasattr(query, "generate")

@pytest.mark.parametrize("case, expected", zip(cases, expected))
def test_bids_architecture_database_call(bids_dataset, case, expected):
    architecture = BidsArchitecture(root=bids_dataset, **case)
    # Basic validation
    assert not(architecture.database.empty)
    
    # Validate structure
    assert all(col in architecture.database.columns for col in [
        'subject', 'session', 'datatype', 'task', 'run', 
        'acquisition', 'description', 'suffix', 'extension'
    ])
    
    # Validate specific queries based on case
    if 'task' in case:
        assert all(architecture.database['task'] == case['task'])
    
    if 'run' in case:
        assert all(architecture.database['run'] == case['run'])
        
    if 'acquisition' in case:
        assert all(architecture.database['acquisition'] == case['acquisition'])
        
    if 'description' in case:
        assert all(architecture.database['description'] == case['description'])
        
    if 'suffix' in case:
        assert all(architecture.database['suffix'] == case['suffix'])
        
    if 'extension' in case:
        assert all(architecture.database['extension'] == f".{case['extension']}")
    
    # Validate file existence
    assert all(Path(f).exists() for f in architecture.database['filename'])
    
    # Validate BIDS naming convention
    assert all(architecture.database['filename'].apply(
        lambda x: str(Path(x).name).startswith('sub-')
    ))
    
# BidsArchitecture Tests
def test_bids_architecture_selection_methods():
    arch = BidsArchitecture(root=Path("/data"))
    
    # Test numerical checks
    test_series = pd.Series(["1", "2", "3"])
    assert arch._is_numerical(test_series) == True
    
    # Test range selection
    test_series = pd.Series(["1", "2", "3", "4", "5"])
    result = arch._get_range(test_series, "2", "4")
    assert all(result == [False, True, True, False, False])
    
    # Test string interpretation
    result = arch._interpret_string(test_series, "2-4")
    assert all(result == [False, True, True, False, False])

def test_bids_architecture_select():
    arch = BidsArchitecture(root=Path("/data"))
    arch.database = pd.DataFrame({
        "subject": ["001", "002", "003"],
        "session": ["01", "01", "02"],
        "task": ["rest", "rest", "task"]
    })
    
    result = arch.select(subject="001", session="01")
    assert len(result) == 1
    assert result.iloc[0]["subject"] == "001"

import pytest
import pandas as pd
from pathlib import Path
from eeg_research.system.bids_selector import BidsArchitecture

@pytest.fixture
def sample_bids_root(tmp_path):
    """Create a sample BIDS directory structure for testing."""
    root = tmp_path / "bids_dataset"
    # Create sample BIDS structure
    for sub in ["01", "02", "03", "04"]:
        for ses in ["01", "02", "03"]:
            for modality in ["eeg","mri"]:
                path = root / f"sub-{sub}" / f"ses-{ses}" / modality
                path.mkdir(parents=True)
                # Create sample files
                (path / f"sub-{sub}_ses-{ses}_task-rest_run-01_desc-clean_{modality}.vhdr").touch()
                (path / f"sub-{sub}_ses-{ses}_task-anothertask_run-01_{modality}.vhdr").touch()
    return root

@pytest.fixture
def bids_selector(sample_bids_root):
    """Create BidsArchitecture instance with sample data."""
    return BidsArchitecture(root=sample_bids_root)

def test_construct_path(bids_selector):
    bids_selector.subject = "01"
    bids_selector.session = "01"
    bids_selector.datatype = "eeg"
    path = bids_selector._construct_path()
    assert str(path) == "sub-01/ses-01/eeg"

def test_construct_file(bids_selector):
    bids_selector.subject = "01"
    bids_selector.task = "rest"
    bids_selector.extension = "vhdr"
    filename = bids_selector._construct_file()
    assert "sub-01" in filename
    assert "task-rest" in filename
    assert filename.endswith(".vhdr")

def test_parse_filename(bids_selector, sample_bids_root):
    file = sample_bids_root / "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_run-01_desc-clean_eeg.vhdr"
    result = bids_selector.parse_filename(file)
    assert result["subject"] == "01"
    assert result["session"] == "01"
    assert result["task"] == "rest"
    assert result["run"] == "01"
    assert result["description"] == "clean"

def test_standardize_input_str(bids_selector):
    assert bids_selector._standardize_input_str("sub-01") == "01"
    assert bids_selector._standardize_input_str("ses-02") == "02"
    assert bids_selector._standardize_input_str("task-rest") == "rest"
    assert bids_selector._standardize_input_str("run-01") == "01"
    assert bids_selector._standardize_input_str("test") == "test"

def test_get_range(bids_selector):
    series = pd.Series(["01", "02", "03", "04"])
    result = bids_selector._get_range(series, "02", "04")
    assert all(result == [False, True, True, False])

def test_get_single_loc(bids_selector):
    series = pd.Series(["01", "02", "03"])
    result = bids_selector._get_single_loc(series, "02")
    assert all(result == [False, True, False])
    with pytest.raises(Exception):
        bids_selector._get_single_loc(series, "04")

def test_is_numerical(bids_selector):
    series = pd.Series(["01", "02", "03"])
    assert bids_selector._is_numerical(series) == True
    series = pd.Series(["01", "abc", "03"])
    assert bids_selector._is_numerical(series) == False

def test_interpret_string(bids_selector):
    series = pd.Series(["01", "02", "03", "04"])
    result = bids_selector._interpret_string(series, "02-04")
    assert all(result == [False, True, True, False])
    with pytest.raises(ValueError):
        bids_selector._interpret_string(series, "a-b")

def test_select_unique_vals(bids_selector):
    result = bids_selector.select(
        subject="01",
        session="01",
        task="rest",
        datatype = "eeg",
        suffix = "eeg",
    )
    assert len(result) == 1
    assert "sub-01" in str(result.iloc[0]["filename"])
    assert "task-rest" in str(result.iloc[0]["filename"])

def test_select_list_vals(bids_selector):
    result = bids_selector.select(
        subject=["01","02"],
        session=["01","02"],
        task="rest",
        datatype = "eeg",
        suffix = "eeg",
    )
    assert len(result) == 4
    assert "sub-01" in str(result.iloc[0]["filename"])
    assert "ses-01" in str(result.iloc[0]["filename"])
    assert "task-rest" in str(result.iloc[0]["filename"])
    assert "sub-01" in str(result.iloc[1]["filename"])
    assert "ses-02" in str(result.iloc[1]["filename"])
    assert "task-rest" in str(result.iloc[1]["filename"])
    assert "sub-02" in str(result.iloc[2]["filename"])
    assert "ses-01" in str(result.iloc[2]["filename"])
    assert "task-rest" in str(result.iloc[2]["filename"])
    assert "sub-02" in str(result.iloc[3]["filename"])
    assert "ses-02" in str(result.iloc[3]["filename"])
    assert "task-rest" in str(result.iloc[3]["filename"])

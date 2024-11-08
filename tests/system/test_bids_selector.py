from unittest.mock import MagicMock, patch

import pytest

from eeg_research.system.bids_selector import BIDSselector


@pytest.fixture
def mock_bids_layout() -> MagicMock:
    """Mock a BIDS layout with a simple structure for testing."""
    # Mock a BIDS layout with a simple structure for testing
    layout = MagicMock()
    layout.get = MagicMock(return_value=["sub-01", "sub-02", "sub-03", "sub-04"])
    return layout

@pytest.fixture
def bids_selector(mock_bids_layout):
    with patch("your_module.bids.BIDSLayout", return_value=mock_bids_layout):
        return BIDSselector(root="/mock/path")

def test_init(bids_selector):
    # Test initialization and if the layout is set correctly
    assert isinstance(bids_selector._original_layout, MagicMock)
    assert bids_selector.user_input == {
        "subject": None,
        "session": None,
        "task": None,
        "run": None,
        "datatype": None,
        "suffix": None,
        "extension": None
    }

def test_standardize_input_str(bids_selector):
    # Test standardizing input string by removing prefixes
    assert bids_selector._standardize_input_str("sub-01") == "01"
    assert bids_selector._standardize_input_str("task-rest") == "rest"
    assert bids_selector._standardize_input_str("run-02") == "02"
    assert bids_selector._standardize_input_str("ses-01") == "01"

def test_standardize_attributes(bids_selector):
    # Test standardizing all attributes
    bids_selector.subject = "sub-01"
    bids_selector.session = ["ses-01", "ses-02"]
    bids_selector._standardize_attributes()
    assert bids_selector.subject == "01"
    assert bids_selector.session == ["01", "02"]

def test_convert_input_to_list(bids_selector):
    # Test converting inputs to lists or ranges
    with patch.object(bids_selector._original_layout, "get", return_value=["01", "02", "03", "04"]):
        assert bids_selector._convert_input_to_list("subject", "01-03") == ["01", "02", "03"]
        assert bids_selector._convert_input_to_list("subject", "01,*") == ["01", "02", "03", "04"]

def test_add_method(bids_selector):
    # Test adding two BIDSselector objects
    other_selector = BIDSselector(root="/mock/path", subject="02", task="rest")
    bids_selector.subject = "01"
    bids_selector.task = ["task1"]

    new_selector = bids_selector + other_selector
    assert new_selector.subject == ["01", "02"]
    assert new_selector.task == ["task1", "rest"]

def test_to_dict(bids_selector):
    # Test if the attributes are correctly returned in a dictionary
    bids_selector.subject = "01"
    bids_selector.task = "task-rest"
    expected_dict = {
        "subject": "01",
        "session": None,
        "task": "task-rest",
        "run": None,
        "datatype": None,
        "suffix": None,
        "extension": None,
    }
    assert bids_selector.to_dict() == expected_dict

def test_set_bids_attributes(bids_selector, mock_bids_layout):
    # Test setting BIDS attributes using mock layout data
    bids_selector.subject = "01-*"
    bids_selector.set_bids_attributes()
    assert bids_selector.subject == ["01", "02", "03", "04"]

def test_layout_property(bids_selector):
    # Test layout property filters files correctly
    with patch.object(bids_selector._original_layout, "get", return_value=["file1.nii.gz", "file2.nii.gz"]):
        bids_selector.subject = "01"
        files = bids_selector.layout
        assert files == ["file1.nii.gz", "file2.nii.gz"]

def test_str_method(bids_selector):
    # Test string representation of BIDSselector object
    bids_selector.subject = "01"
    bids_selector.task = "rest"
    with patch.object(bids_selector._original_layout, "get", return_value=["file1.nii.gz", "file2.nii.gz"]):
        result_str = str(bids_selector)
        assert "Subjects: 1 (01)" in result_str
        assert "Tasks: 1 (rest)" in result_str
        assert "Files: 2" in result_str

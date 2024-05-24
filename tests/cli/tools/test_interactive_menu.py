"""Tests for eeg_research.cli.toolbox.interactive_menu.py."""

import pytest
from pyparsing import Any

import eeg_research.cli.tools.interactive_menu as script


@pytest.fixture
def menu(mocker: Any, request: pytest.FixtureRequest) -> script.InteractiveMenu:
    """Fixture to create an InteractiveMenu object."""
    mocker.patch("eeg_research.cli.tools.interactive_menu.TerminalMenu")
    menu_entries = request.param[0]
    entity = request.param[1]
    preselection = request.param[2]
    return script.InteractiveMenu(menu_entries, entity, "title", preselection)


@pytest.mark.parametrize("menu", [(["item1"], "description", [0])], indirect=True)
def test_interactive_menu_init(menu: script.InteractiveMenu) -> None:
    """Test the initialization of the InteractiveMenu class."""
    assert menu.menu_entries == ["item1"]
    assert menu.entity == "description"
    assert menu.title == "title"
    assert menu.preselection == [0]


@pytest.mark.parametrize("menu", [(["item1", "item2"], "task", [0])], indirect=True)
def test_create_menu(menu: script.InteractiveMenu) -> None:
    """Test the _create_menu method of the InteractiveMenu class."""
    assert menu._create_menu()


@pytest.mark.parametrize("menu", [(["item1", "item2"], "suffix", [0])], indirect=True)
def test_handle_user_input_select_all(menu: script.InteractiveMenu) -> None:
    """Test the _handle_user_input method when all items are selected."""
    menu.selected_indices = [0]
    assert menu._handle_user_input() == ["item1", "item2"]


@pytest.mark.parametrize(
    "menu", [(["1", "2", "3", "4"], "subject", None)], indirect=True
)
def test_handle_user_input_enter_range(
    mocker: Any, menu: script.InteractiveMenu
) -> None:
    """Test the _handle_user_input method when a range is entered."""
    mocker.patch("builtins.input", side_effect=["2", "4"])
    menu.selected_indices = [1]
    assert menu._handle_user_input() == ["2", "3", "4"]


@pytest.mark.parametrize("menu", [(["1", "2", "3"], "session", [2, 3])], indirect=True)
def test_handle_user_input_specific_items(menu: script.InteractiveMenu) -> None:
    """Test the _handle_user_input method when specific items are selected."""
    menu.selected_indices = [2, 3]
    assert menu._handle_user_input() == ["1", "2"]


@pytest.mark.parametrize("menu", [(["1", "2", "3"], "session", [2, 3])], indirect=True)
def test_get_selected_items(menu: script.InteractiveMenu) -> None:
    """Test the get_selected_items method of the InteractiveMenu class."""
    menu.selected_indices = [2, 3]
    menu.selected_items = menu._handle_user_input()
    assert menu.get_selected_items() == ["1", "2"]

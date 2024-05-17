#!/usr/bin/env -S  python  #
# -*- coding: utf-8 -*-
# ===============================================================================
# Author: Dr. Samuel Louviot, PhD
#         Dr. Alp Erkent, MD, MA
# Institution: Nathan Kline Institute
#              Child Mind Institute
# Address: 140 Old Orangeburg Rd, Orangeburg, NY 10962, USA
#          215 E 50th St, New York, NY 10022
# Date: 2024-02-27
# email: samuel DOT louviot AT nki DOT rfmh DOT org
#        alp DOT erkent AT childmind DOT org
# ===============================================================================
# LICENCE GNU GPLv3:
# Copyright (C) 2024  Dr. Samuel Louviot, PhD
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ===============================================================================

"""A module for creating and handling interactive menus."""

from simple_term_menu import TerminalMenu


class InteractiveMenu:
    """A class for creating and handling interactive menus."""

    def __init__(
        self,
        menu_entries: list[str],
        entity: str,
        title: str,
        preselection: list[int] | None = None,
    ) -> None:
        """Initialize the InteractiveMenu object."""
        self.menu_entries = menu_entries
        self.entity = entity
        self.title = title
        self.preselection = preselection
        self.menu = self._create_menu()
        self.selected_indices = self.menu.show()
        self.selected_items = self._handle_user_input()

    def _create_menu(self) -> TerminalMenu:
        """Create a terminal menu for selecting items.

        Args:
            menu_items (list[str]): A list of items to display in the menu.
            entity (str): The entity that the items belong to.

        Returns:
            TerminalMenu: The TerminalMenu object.
        """
        # Add options to the menu items if there are more than one
        if len(self.menu_entries) > 1 and self.entity != "script":
            # Add an option for user input to the menu items if the entity has integers
            if self.entity in ["subject", "session", "run"]:
                self.menu_entries = ["Enter range"] + self.menu_entries
            # Add an option for selecting all items to the menu items
            self.menu_entries = ["Select all"] + self.menu_entries

        # If entity is run, convert ints to strings
        # In pybids, only run entities are integers
        # and terminal menu only accepts strings
        if self.entity == "run":
            self.menu_entries = [str(item) for item in self.menu_entries]

        # Create a terminal menu with the given items and title
        menu = TerminalMenu(
            menu_entries=self.menu_entries,
            title=self.title,
            multi_select=True,
            show_multi_select_hint=True,
            preselected_entries=self.preselection,
        )

        return menu

    def _handle_user_input(self) -> list[str]:
        """Handle user input for the terminal menu.

        Returns:
            list[str]: A list of the selected items.
        """
        # If the user selected the "Select all" option
        if 0 in self.selected_indices and "Select all" in self.menu_entries:
            # Return all items except "Select all" and "Enter range"
            selected_items = [
                item
                for item in self.menu_entries
                if item not in ["Select all", "Enter range"]
            ]
        # If the user selected the "Enter range" option
        elif 1 in self.selected_indices and "Enter range" in self.menu_entries:
            # Ask the user to enter a range
            start = int(input("Enter the start of the range: "))
            end = int(input("Enter the end of the range: "))

            # Return items in the selected range except 'Select all' and 'Enter range'
            selected_items = [
                item for item in self.menu_entries[2:] if start <= int(item) <= end
            ]
        # If the user selected specific items
        else:
            # Return the selected items
            selected_items = [self.menu_entries[i] for i in self.selected_indices]

        return selected_items

    def get_selected_items(self) -> list[str]:
        """Return the selected items."""
        return self.selected_items

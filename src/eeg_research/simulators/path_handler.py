"""The module to handle the directory tree."""

import os
from typing import Generator


class DirectoryTree:
    """The class to handle the directory tree."""

    def __init__(self, root_dir: str | os.PathLike) -> None:
        """Initialize the class with the root directory."""
        self.root_dir = os.path.abspath(root_dir)
        self.current_dir = self.root_dir

    def generate_tree(
        self: "DirectoryTree",
    ) -> Generator[tuple[str, list[str], list[str]], None, None]:
        """A generator method to walk through the directory tree.

        It starts from the root directory. Yields tuples of (path, dirs, files)
        for each directory in the tree.
        """
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            yield dirpath, dirnames, filenames

    def list_directory_contents(
        self: "DirectoryTree",
    ) -> tuple[list[str] | str, list[str]]:
        """List the contents of the current directory."""
        try:
            with os.scandir(self.current_dir) as it:
                dirs = [entry.name for entry in it if entry.is_dir()]
                files = [entry.name for entry in it if entry.is_file()]
            return dirs, files
        except PermissionError as e:
            return f"Permission denied: {e}", []

    def change_directory(self, target_dir: str | os.PathLike) -> str | os.PathLike:
        """Change the current directory to target_dir.

        Change the current directory if it's a subdirectory
        of current_dir or a valid path.

        Args:
            target_dir (str | os.PathLike): The target directory path.
        """
        new_dir = os.path.abspath(os.path.join(self.current_dir, target_dir))
        if os.path.commonpath([self.root_dir, new_dir]) != self.root_dir:
            return f"Error: {new_dir} is not a subdirectory of the root directory."
        if os.path.isdir(new_dir):
            self.current_dir = new_dir
            return f"Changed current directory to {self.current_dir}"
        else:
            return f"Error: Directory {new_dir} does not exist."

    def print_tree(
        self: "DirectoryTree",
        start_path: str | os.PathLike = "",
        prefix: str | os.PathLike = "",
    ) -> None:
        """Print the tree in a formated way.

        Recursively prints the directory tree starting f
        rom start_path with proper formatting.

        Args:
            start_path (str | os.PathLike, optional): The starting path.
                                                      Defaults to ''.
            prefix (str | os.PathLike, optional): The prefix to be added.
                                                  Defaults to ''.
        """
        if start_path == "":
            start_path = self.root_dir
            print(start_path + "/")

        entries = [entry for entry in os.scandir(str(start_path))]
        entries = sorted(entries, key=lambda e: (e.is_file(), e.name))
        last_index = len(entries) - 1

        for index, entry in enumerate(entries):
            connector = "├──" if index != last_index else "└──"
            if entry.is_dir():
                print(f"{prefix}{connector} {entry.name}/")
                extension = "│   " if index != last_index else "    "
                self.print_tree(
                    os.path.join(start_path, entry.name), str(prefix) + str(extension)
                )
            else:
                print(f"{prefix}{connector} {entry.name}")

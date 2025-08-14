from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Union


class Loader:
    """Concatenate all .txt files under a data directory (including subfolders).

    Usage:
        loader = Loader()  # defaults to <project_root>/data
        text = loader.to_string()
        loader.to_file("data/concatenated.txt")
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None, encoding: str = "utf-8") -> None:
        project_root = Path(__file__).resolve().parent
        self.data_dir: Path = (project_root / "data") if data_dir is None else Path(data_dir).resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        self.encoding: str = encoding

    def find_txt_files(self) -> List[Path]:
        """Return all .txt files under the data directory, recursively, sorted by path."""
        txt_files = [path for path in self.data_dir.rglob("*.txt") if path.is_file()]
        return sorted(txt_files, key=lambda p: str(p).lower())

    def iter_file_contents(
        self,
        files: Optional[Iterable[Path]] = None,
        include_headers: bool = False,
        separator: str = "\n",
    ) -> Iterator[str]:
        """Yield concatenated contents for the provided files lazily.

        - include_headers: if True, yield a header line with the relative file path before each file
        - separator: text appended after each file's contents (default newline)
        """
        selected_files: List[Path] = self.find_txt_files() if files is None else list(files)
        for path in selected_files:
            if include_headers:
                header = f"===== {path.relative_to(self.data_dir)} =====\n"
                yield header
            with path.open("r", encoding=self.encoding, errors="ignore") as infile:
                for line in infile:
                    yield line
            if separator:
                yield separator

    def to_string(self, include_headers: bool = False, separator: str = "\n") -> str:
        """Return a single string containing all concatenated .txt contents."""
        return "".join(self.iter_file_contents(include_headers=include_headers, separator=separator))

    def to_file(
        self,
        output_path: Union[str, Path],
        include_headers: bool = False,
        separator: str = "\n",
        exclude_output_from_search: bool = True,
    ) -> Path:
        """Write concatenated contents to a file and return its path.

        - exclude_output_from_search: omit the output file if it already exists inside data_dir
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        files = self.find_txt_files()
        if exclude_output_from_search:
            try:
                output_abs = output_path.resolve()
            except Exception:
                output_abs = output_path
            files = [p for p in files if p.resolve() != output_abs]

        with output_path.open("w", encoding=self.encoding, errors="ignore", newline="") as outfile:
            for chunk in self.iter_file_contents(files=files, include_headers=include_headers, separator=separator):
                outfile.write(chunk)
        return output_path


__all__ = ["Loader"]

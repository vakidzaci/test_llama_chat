from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import hashlib


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "site-packages",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "node_modules",
}


@dataclass(frozen=True)
class FileRecord:
    path: Path                 # absolute path
    rel_path: Path             # relative to repo root
    size_bytes: int
    sha256: str


@dataclass
class IngestResult:
    repo_root: Path
    files: List[FileRecord] = field(default_factory=list)
    skipped: List[Tuple[Path, str]] = field(default_factory=list)  # (rel_path, reason)


class CodeReview:
    """
    Minimal skeleton for a code-review pipeline.
    Step 1: ingest local directory (collect .py files + metadata).
    Next steps later: indexing (Chroma), repo map, review passes, reporting.
    """

    def __init__(
        self,
        exclude_dirs: Optional[Set[str]] = None,
        max_file_size_bytes: int = 2_000_000,  # 2 MB safety default
        follow_symlinks: bool = False,
    ) -> None:
        self.exclude_dirs: Set[str] = set(exclude_dirs or DEFAULT_EXCLUDE_DIRS)
        self.max_file_size_bytes = int(max_file_size_bytes)
        self.follow_symlinks = bool(follow_symlinks)

        self.ingest_result: Optional[IngestResult] = None
        self._file_index: Dict[Path, FileRecord] = {}  # rel_path -> record

    # -------------------------
    # Public API
    # -------------------------

    def ingest(self, repo_dir: str | Path) -> IngestResult:
        """
        Ingest a repository from a local directory path.
        - Collects *.py files recursively
        - Skips excluded directories
        - Records size + sha256 for stability / caching later
        """
        root = Path(repo_dir).expanduser().resolve()

        if not root.exists():
            raise FileNotFoundError(f"repo_dir does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"repo_dir is not a directory: {root}")

        result = IngestResult(repo_root=root)

        for abs_path in self._iter_repo_files(root):
            if abs_path.suffix != ".py":
                continue

            rel_path = abs_path.relative_to(root)

            # Skip huge files
            try:
                size = abs_path.stat().st_size
            except OSError as e:
                result.skipped.append((rel_path, f"stat failed: {e}"))
                continue

            if size > self.max_file_size_bytes:
                result.skipped.append((rel_path, f"too large ({size} bytes)"))
                continue

            # Hash file contents (useful later for incremental embeddings/review cache)
            try:
                sha256 = self._sha256_file(abs_path)
            except OSError as e:
                result.skipped.append((rel_path, f"read/hash failed: {e}"))
                continue

            rec = FileRecord(
                path=abs_path,
                rel_path=rel_path,
                size_bytes=size,
                sha256=sha256,
            )
            result.files.append(rec)
            self._file_index[rel_path] = rec

        # Stable order for deterministic pipelines
        result.files.sort(key=lambda r: str(r.rel_path).lower())
        result.skipped.sort(key=lambda x: str(x[0]).lower())

        self.ingest_result = result
        return result

    def list_files(self) -> List[Path]:
        """Return ingested python file paths relative to repo root."""
        if not self.ingest_result:
            return []
        return [rec.rel_path for rec in self.ingest_result.files]

    def read_text(self, rel_path: str | Path, encoding: str = "utf-8") -> str:
        """Read an ingested file by relative path."""
        if not self.ingest_result:
            raise RuntimeError("Nothing ingested yet. Call ingest() first.")

        rel = Path(rel_path)
        rec = self._file_index.get(rel)
        if rec is None:
            raise KeyError(f"File not found in ingest index: {rel}")

        return rec.path.read_text(encoding=encoding, errors="replace")

    # -------------------------
    # Internal helpers
    # -------------------------

    def _iter_repo_files(self, root: Path) -> Iterable[Path]:
        """
        Recursive file iterator with directory exclusions.
        Avoids excluded dirs anywhere in the tree.
        """
        stack = [root]
        while stack:
            cur = stack.pop()

            # Symlink handling
            if cur.is_symlink() and not self.follow_symlinks:
                continue

            try:
                entries = list(cur.iterdir())
            except OSError:
                continue

            for p in entries:
                name = p.name

                # Skip excluded directories
                if p.is_dir():
                    if name in self.exclude_dirs:
                        continue
                    # Also skip hidden cache dirs by pattern if you want later
                    stack.append(p)
                    continue

                if p.is_file():
                    # Skip symlink files unless configured
                    if p.is_symlink() and not self.follow_symlinks:
                        continue
                    yield p

    @staticmethod
    def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    cr = CodeReview()
    res = cr.ingest("/path/to/your/project")

    print(f"Repo: {res.repo_root}")
    print(f"Python files: {len(res.files)}")
    if res.skipped:
        print(f"Skipped: {len(res.skipped)} (first 10 shown)")
        for rel, reason in res.skipped[:10]:
            print(f"  - {rel} :: {reason}")

    # Show first few files
    for f in cr.list_files()[:10]:
        print(f"  {f}")

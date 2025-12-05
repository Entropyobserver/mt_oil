from pathlib import Path
from typing import Optional


class PathManager:
    
    @staticmethod
    def ensure_dir(path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def ensure_parent_dir(path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_relative_path(path: Path, base: Path) -> Path:
        return Path(path).relative_to(base)
    
    @staticmethod
    def resolve_path(path: Path, base: Optional[Path] = None) -> Path:
        path = Path(path)
        if base and not path.is_absolute():
            return (base / path).resolve()
        return path.resolve()
    
    @staticmethod
    def glob_files(directory: Path, pattern: str):
        return list(Path(directory).glob(pattern))
    
    @staticmethod
    def safe_remove(path: Path, is_dir: bool = False):
        path = Path(path)
        if is_dir:
            import shutil
            if path.exists():
                shutil.rmtree(path)
        else:
            if path.exists():
                path.unlink()
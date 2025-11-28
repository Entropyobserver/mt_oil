from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nob_eng_translation")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]
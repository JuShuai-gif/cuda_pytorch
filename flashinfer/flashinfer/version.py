# Centralized version information to avoid circular imports
try:
    from ._build_meta import __version__ as __version__
    from ._build_meta import __git_version__ as __git_version__
except ModuleNotFoundError:
    __version__ = "0.0.0+unknown"
    __git_version__ = "unknown"
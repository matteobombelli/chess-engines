class AlphaMiniError(RuntimeError):
    """Expected configuration, data, or run-state failure."""


class ConfigError(AlphaMiniError):
    """A run configuration is invalid."""


class IntegrityError(AlphaMiniError):
    """An immutable object, manifest, or data file failed validation."""


class RunLockedError(AlphaMiniError):
    """Another process currently owns the run lock."""


class DependencyUnavailable(AlphaMiniError):
    """An optional training dependency is not installed."""

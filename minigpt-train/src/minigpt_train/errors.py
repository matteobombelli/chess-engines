class MiniGptError(RuntimeError):
    """Expected configuration, data, or run-state failure."""


class ConfigError(MiniGptError):
    """A run configuration is invalid."""


class IntegrityError(MiniGptError):
    """An immutable object, manifest, or data file failed validation."""


class RunLockedError(MiniGptError):
    """Another process currently owns the run lock."""


class DependencyUnavailable(MiniGptError):
    """An optional training dependency is not installed."""


class DiskSpaceError(MiniGptError):
    """Free space fell below the configured floor before a checkpoint write."""

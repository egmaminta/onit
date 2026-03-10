"""Backend abstraction layer for file operations, execution, and sandboxing."""

from .protocol import BackendProtocol, SandboxBackendProtocol  # noqa: F401
from .state import StateBackend  # noqa: F401
from .filesystem import FilesystemBackend  # noqa: F401
from .composite import CompositeBackend  # noqa: F401
from .container import ContainerSandbox  # noqa: F401
from .local_shell import LocalShellBackend  # noqa: F401

__all__ = [
    "BackendProtocol",
    "SandboxBackendProtocol",
    "StateBackend",
    "FilesystemBackend",
    "CompositeBackend",
    "ContainerSandbox",
    "LocalShellBackend",
]

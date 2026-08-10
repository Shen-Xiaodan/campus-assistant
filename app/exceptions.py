"""Domain-specific exceptions surfaced by the API and CLI."""


class CampusAssistantError(Exception):
    """Base class for expected application errors."""


class DocumentProcessingError(CampusAssistantError):
    """A document could not be parsed or indexed."""


class IndexUnavailableError(CampusAssistantError):
    """The vector index is absent or cannot be opened."""


class GenerationUnavailableError(CampusAssistantError):
    """The configured answer model is unavailable."""

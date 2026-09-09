"""Application services; individual services are loaded on demand."""

__all__ = ["DocumentParser", "parse_document", "get_document_info"]


def __getattr__(name):
    if name in __all__:
        from . import document_parser
        return getattr(document_parser, name)
    raise AttributeError(name)

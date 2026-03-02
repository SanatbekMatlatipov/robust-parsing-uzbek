"""
spaCy entry point for Uzbek language registration.

When spacy_uzbek is installed (pip install -e spacy_uzbek/), this module
registers the Uzbek language class so that:

    import spacy
    nlp = spacy.blank("uz")

works out of the box.
"""

from spacy.language import Language
from spacy_uzbek.lang.uz import Uzbek


def setup():
    """Called automatically when the package is imported."""
    # Uzbek class self-registers via Language.__init_subclass__
    # but we ensure it's available
    pass

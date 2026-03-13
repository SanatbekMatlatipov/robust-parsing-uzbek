"""
Uzbek language class for spaCy.

Since Uzbek is not natively supported in spaCy, we register a custom
Language subclass. This enables spaCy's full pipeline (tokenizer, tagger,
parser, etc.) to work with Uzbek text.

Usage:
    from spacy_uzbek.lang.uz import Uzbek
    nlp = Uzbek()
    # or after importing this module:
    import spacy
    nlp = spacy.blank("uz")
"""

from spacy.language import Language
from spacy.lang.tokenizer_exceptions import BASE_EXCEPTIONS
from spacy.util import registry

from .stop_words import STOP_WORDS
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS


class UzbekDefaults(Language.Defaults):
    stop_words = STOP_WORDS
    tokenizer_exceptions = {**BASE_EXCEPTIONS, **TOKENIZER_EXCEPTIONS}


class Uzbek(Language):
    """Custom spaCy Language class for Uzbek."""
    lang = "uz"
    Defaults = UzbekDefaults


# Register in spaCy's language registry so spacy.blank("uz") works
try:
    registry.languages.register("uz", func=Uzbek)
except Exception:
    pass  # Already registered

# Make sure it's discoverable
__all__ = ["Uzbek"]

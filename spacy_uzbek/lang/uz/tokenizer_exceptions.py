"""
Uzbek tokenizer exceptions for spaCy.

Handles common abbreviations, contractions, and multi-word tokens
that should not be split by the default tokenizer rules.
"""

TOKENIZER_EXCEPTIONS = {
    # Common abbreviations
    "masalan": [{
        "ORTH": "masalan",
    }],
    "va.h.k.": [{
        "ORTH": "va.h.k.",
    }],
    "т.": [{
        "ORTH": "т.",
    }],
    "б.э.а.": [{
        "ORTH": "б.э.а.",
    }],
    "b.e.a.": [{
        "ORTH": "b.e.a.",
    }],
    "mln.": [{
        "ORTH": "mln.",
    }],
    "mlrd.": [{
        "ORTH": "mlrd.",
    }],
    "ming.": [{
        "ORTH": "ming.",
    }],
    "prof.": [{
        "ORTH": "prof.",
    }],
    "dots.": [{
        "ORTH": "dots.",
    }],
    "akad.": [{
        "ORTH": "akad.",
    }],
}

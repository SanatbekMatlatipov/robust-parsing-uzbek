from setuptools import setup, find_packages

setup(
    name="spacy-uzbek",
    version="0.1.0",
    description="Uzbek language support for spaCy — POS tagger and dependency parser",
    author="Sanatbek Matlatipov",
    author_email="s.matlatipov@nuu.uz",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.5.0",
    ],
    extras_require={
        "transformers": [
            "spacy-transformers>=1.2.0",
            "transformers>=4.20.0",
            "torch>=1.9.0",
        ],
    },
    entry_points={
        "spacy_languages": [
            "uz = spacy_uzbek.lang.uz:Uzbek",
        ],
    },
    package_data={
        "spacy_uzbek": ["configs/*.cfg"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)

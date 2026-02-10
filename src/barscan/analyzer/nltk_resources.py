"""Centralized NLTK resource management."""

from __future__ import annotations

from enum import Enum
from typing import Final

import nltk

from barscan.exceptions import NLTKResourceError
from barscan.logging import get_logger

logger = get_logger("analyzer.nltk_resources")


class NLTKResource(Enum):
    """NLTK resources used by BarScan.

    Each value is a tuple of (data_path, download_name) where:
    - data_path: Path to check in nltk.data.find()
    - download_name: Package name for nltk.download()
    """

    PUNKT_TAB = ("tokenizers/punkt_tab", "punkt_tab")
    STOPWORDS = ("corpora/stopwords", "stopwords")
    WORDNET = ("corpora/wordnet", "wordnet")
    POS_TAGGER = ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")
    VADER_LEXICON = ("sentiment/vader_lexicon.zip", "vader_lexicon")


# Resource groups for different analysis modules
PROCESSOR_RESOURCES: Final[tuple[NLTKResource, ...]] = (
    NLTKResource.PUNKT_TAB,
    NLTKResource.STOPWORDS,
    NLTKResource.WORDNET,
)

POS_RESOURCES: Final[tuple[NLTKResource, ...]] = (NLTKResource.POS_TAGGER,)

SENTIMENT_RESOURCES: Final[tuple[NLTKResource, ...]] = (NLTKResource.VADER_LEXICON,)


def ensure_resource(resource: NLTKResource) -> None:
    """Ensure a single NLTK resource is available.

    Checks if the resource exists locally, and downloads it if not.

    Args:
        resource: NLTK resource to check/download.

    Raises:
        NLTKResourceError: If resource cannot be downloaded.
    """
    path, name = resource.value
    try:
        nltk.data.find(path)
        logger.debug("NLTK resource '%s' already available", name)
    except LookupError:
        logger.debug("Downloading NLTK resource '%s'", name)
        try:
            nltk.download(name, quiet=True)
            logger.debug("Successfully downloaded NLTK resource '%s'", name)
        except Exception as e:
            raise NLTKResourceError(
                f"Failed to download NLTK resource '{name}': {e}",
                resource_name=name,
            ) from e


def ensure_resources(resources: tuple[NLTKResource, ...]) -> None:
    """Ensure multiple NLTK resources are available.

    Args:
        resources: Tuple of NLTK resources to check/download.

    Raises:
        NLTKResourceError: If any resource cannot be downloaded.
    """
    for resource in resources:
        ensure_resource(resource)

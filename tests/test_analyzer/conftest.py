"""Shared test fixtures for analyzer tests."""

import pytest

from barscan.analyzer.models import AnalysisConfig, WordFrequency


@pytest.fixture
def sample_lyrics_text() -> str:
    """Sample lyrics with section headers."""
    return """[Verse 1]
Hello world, hello universe
The world is spinning around

[Chorus]
Round and round we go
Round and round again

[Verse 2]
Another day, another world
Hello to the world once more
"""


@pytest.fixture
def cleaned_lyrics_text() -> str:
    """Sample lyrics without section headers."""
    return (
        "Hello world, hello universe The world is spinning around "
        "Round and round we go Round and round again "
        "Another day, another world Hello to the world once more"
    )


@pytest.fixture
def default_config() -> AnalysisConfig:
    """Default analysis configuration."""
    return AnalysisConfig()


@pytest.fixture
def config_with_lemmatization() -> AnalysisConfig:
    """Analysis config with lemmatization enabled."""
    return AnalysisConfig(use_lemmatization=True)


@pytest.fixture
def config_no_stop_words() -> AnalysisConfig:
    """Analysis config without stop word filtering."""
    return AnalysisConfig(remove_stop_words=False)


@pytest.fixture
def config_min_length_3() -> AnalysisConfig:
    """Analysis config with minimum word length of 3."""
    return AnalysisConfig(min_word_length=3)


@pytest.fixture
def sample_word_frequencies() -> list[WordFrequency]:
    """Sample word frequencies for testing."""
    return [
        WordFrequency(word="world", count=4, percentage=20.0),
        WordFrequency(word="hello", count=3, percentage=15.0),
        WordFrequency(word="round", count=4, percentage=20.0),
    ]

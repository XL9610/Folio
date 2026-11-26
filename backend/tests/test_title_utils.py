"""Tests for clean_filename."""

import pytest

from app.services.generation._title_utils import clean_filename


@pytest.mark.parametrize(
    "value,expected",
    [
        ("AI_Engineering_Building_Applications_z_library_sk,_1lib_sk,.pdf", "AI Engineering Building Applications"),
        ("Data Structures and Algorithms in Java, 6th Edition, 2014.pdf", "Data Structures and Algorithms in Java, 6th Edition, 2014"),
    ],
    ids=['strips_z_library_marker', 'preserves_real_commas'],
)
def test_strips_z_library_marker(value, expected):
    result = clean_filename(value)
    assert result == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("A History of Western Music Tenth.pdf", "A History of Western Music Tenth"),
        ("some_pdf_with_underscores.pdf", "some pdf with underscores"),
    ],
    ids=['strips_extension', 'underscore_replaced_with_space'],
)
def test_strips_extension(value, expected):
    assert clean_filename(value) == expected


def test_empty_input_returns_placeholder():
    assert clean_filename("") == "Untitled document"
    assert clean_filename(None) == "Untitled document"
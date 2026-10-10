"""Tests for clean_filename."""

from app.services.generation._title_utils import clean_filename


def test_strips_z_library_marker():
    result = clean_filename("AI_Engineering_Building_Applications_z_library_sk,_1lib_sk,.pdf")
    assert result == "AI Engineering Building Applications"


def test_strips_extension():
    assert clean_filename("A History of Western Music Tenth.pdf") == "A History of Western Music Tenth"


def test_preserves_real_commas():
    result = clean_filename("Data Structures and Algorithms in Java, 6th Edition, 2014.pdf")
    assert result == "Data Structures and Algorithms in Java, 6th Edition, 2014"


def test_underscore_replaced_with_space():
    assert clean_filename("some_pdf_with_underscores.pdf") == "some pdf with underscores"


def test_empty_input_returns_placeholder():
    assert clean_filename("") == "Untitled document"
    assert clean_filename(None) == "Untitled document"
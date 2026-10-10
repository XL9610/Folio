"""Shared test fixtures — generates test PDFs on the fly using PyMuPDF.

Creates three PDF types:
1. With a clean TOC / bookmarks  →  toc_chunking.pdf
2. Without any TOC              →  no_toc.pdf
3. With an unusable TOC         →  broken_toc.pdf (pages are empty)
"""

from __future__ import annotations

import os
import tempfile

import fitz  # PyMuPDF
import pytest


# Offline-safe: point tiktoken at the bundled BPE cache (populated by
# backend/scripts/fetch_tiktoken_cache.py) when it exists, so a network-free
# machine can run pytest without tiktoken fetching cl100k_base from Azure.
_bundled_tt_cache = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "packaging", "tiktoken_cache"
)
if os.path.isdir(_bundled_tt_cache):
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", _bundled_tt_cache)


# ---------------------------------------------------------------------------
# Temporary directory for test PDFs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_pdf_dir():
    """Create a session-scoped temporary directory for test PDFs."""
    tmp = tempfile.mkdtemp(prefix="rag_test_pdfs_")
    yield tmp
    # Cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# PDF with valid TOC
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_with_toc(test_pdf_dir) -> str:
    """Create a 6-page PDF with 3-chapter TOC and meaningful text.

    Structure:
        Chapter 1 (page 1, "Introduction")
            1.1 Motivation (page 1)
            1.2 Scope (page 2)
        Chapter 2 (page 3, "Methods")
            2.1 Algorithm (page 3)
            2.2 Implementation (page 4)
        Chapter 3 (page 5, "Results")
            3.1 Analysis (page 5)
            3.2 Discussion (page 6)
    """
    path = os.path.join(test_pdf_dir, "toc_chunking.pdf")
    doc = fitz.open()

    # Filler text to ensure each page exceeds min_chunk_tokens (30)
    _FILLER = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
    )

    pages_content = [
        # Page 1: Introduction + 1.1 Motivation
        "Chapter 1: Introduction\n\nThis book covers machine learning from first principles. "
        "1.1 Motivation\n\nMachine learning has become an essential tool in modern data analysis. "
        "The ability to learn patterns from data enables systems to make predictions and decisions "
        "without explicit programming. This chapter motivates the study of neural networks. "
        + _FILLER,
        # Page 2: 1.2 Scope
        "1.2 Scope\n\nThis book is designed for graduate students and practitioners. "
        "We cover supervised learning, unsupervised learning, and reinforcement learning. "
        "Prerequisites include linear algebra, calculus, and basic probability theory. "
        "Code examples are provided in Python. "
        + _FILLER,
        # Page 3: Chapter 2 Methods + 2.1 Algorithm
        "Chapter 2: Methods\n\nThis chapter introduces the core algorithms. "
        "2.1 Gradient Descent Algorithm\n\nGradient descent is the workhorse of modern machine learning. "
        "It iteratively adjusts model parameters to minimize the loss function. "
        "The learning rate controls the step size and is a crucial hyperparameter. "
        + _FILLER,
        # Page 4: 2.2 Implementation
        "2.2 Implementation Details\n\nImplementing neural networks requires careful attention "
        "to numerical stability. Weight initialization strategies like Xavier and He "
        "initialization help prevent vanishing or exploding gradients. "
        "Batch normalization further stabilizes training. "
        + _FILLER,
        # Page 5: Chapter 3 Results + 3.1 Analysis
        "Chapter 3: Results and Evaluation\n\nWe evaluate our models on standard benchmarks. "
        "3.1 Performance Analysis\n\nOur model achieves 95% accuracy on MNIST and 89% on CIFAR-10. "
        "These results are competitive with state-of-the-art architectures. "
        "We analyze the training curves and confusion matrices. "
        + _FILLER,
        # Page 6: 3.2 Discussion
        "3.2 Discussion\n\nThe results demonstrate the effectiveness of our approach. "
        "Future work should explore attention mechanisms and transformer architectures. "
        "We also note that data augmentation played a key role in improving generalization. "
        "Limitations include computational cost and sensitivity to hyperparameters. "
        + _FILLER,
    ]

    for i, text in enumerate(pages_content):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), text, fontsize=12)

    # Set TOC / bookmarks (flat format with explicit level numbers)
    doc.set_toc([
        [1, "Chapter 1: Introduction", 1],
        [2, "1.1 Motivation", 1],
        [2, "1.2 Scope", 2],
        [1, "Chapter 2: Methods", 3],
        [2, "2.1 Gradient Descent Algorithm", 3],
        [2, "2.2 Implementation Details", 4],
        [1, "Chapter 3: Results and Evaluation", 5],
        [2, "3.1 Performance Analysis", 5],
        [2, "3.2 Discussion", 6],
    ])

    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# PDF without TOC
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_without_toc(test_pdf_dir) -> str:
    """Create a 4-page PDF with continuous text but no TOC."""
    path = os.path.join(test_pdf_dir, "no_toc.pdf")
    doc = fitz.open()

    lorem = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, "
        "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu "
        "fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in "
        "culpa qui officia deserunt mollit anim id est laborum. "
    )

    for i in range(4):
        page = doc.new_page(width=595, height=842)
        # Repeat text to fill the page
        full_text = lorem * 8
        page.insert_text((72, 72), full_text, fontsize=12)

    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# PDF with broken / unusable TOC (pages have no text)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_with_broken_toc(test_pdf_dir) -> str:
    """Create a PDF with TOC entries pointing to empty pages."""
    path = os.path.join(test_pdf_dir, "broken_toc.pdf")
    doc = fitz.open()

    # 3 empty pages
    for _ in range(3):
        doc.new_page(width=595, height=842)

    doc.set_toc([
        [1, "Ghost Chapter 1", 1],
        [1, "Ghost Chapter 2", 2],
        [1, "Ghost Chapter 3", 3],
    ])

    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# PDF with front matter + real chapters (to test TOC filtering)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_with_front_matter(test_pdf_dir) -> str:
    """Create a PDF with front matter TOC entries followed by real chapters.

    TOC structure:
        Title Page (page 1)        ← front matter
        Copyright (page 2)         ← front matter
        Contents (page 3)          ← front matter
        Preface (page 4)           ← front matter
        Part I (page 5)            ← chapter start detected
        Chapter 1 (page 6)         ← real content starts here
        Chapter 2 (page 7)         ← real content
    """
    path = os.path.join(test_pdf_dir, "front_matter.pdf")
    doc = fitz.open()

    _FILLER = (
        "This is content for this page. It contains learning material. "
        "The quick brown fox jumps over the lazy dog. " * 10
    )

    for i in range(7):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), _FILLER, fontsize=12)

    doc.set_toc([
        [1, "Title Page", 1],
        [1, "Copyright", 2],
        [1, "Contents", 3],
        [1, "Preface", 4],
        [1, "Part I: Introduction", 5],
        [1, "Chapter 1: Getting Started", 6],
        [1, "Chapter 2: Core Concepts", 7],
    ])
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# PDF with Western Music style TOC (editorial board, maps, guide, preface, chapters)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_western_music_style(test_pdf_dir) -> str:
    """Mimics the A History of Western Music TOC structure.

    EDITORIAL ADVISORY BOARD → MAPS → GUIDE → PREFACE → PART ONE → numbered chapters.
    Should skip the first 4 entries and start from PART ONE or "1 Music in Antiquity".
    """
    path = os.path.join(test_pdf_dir, "western_music_style.pdf")
    doc = fitz.open()

    _FILLER = "Study content for this page. " * 30

    for i in range(9):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), _FILLER, fontsize=12)

    doc.set_toc([
        [1, "EDITORIAL ADVISORY BOARD", 1],
        [1, "MAPS", 2],
        [1, "GUIDE TO NAWM SELECTIONS", 3],
        [1, "PREFACE TO THE TENTH EDITION", 4],
        [1, "PART ONE", 5],
        [1, "1 Music in Antiquity", 6],
        [1, "2 The Christian Church", 7],
        [1, "3 Roman Liturgy and Chant", 8],
        [1, "PART TWO", 9],
    ])
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# PDF where Introduction IS real study content (long Introduction)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_intro_as_body(test_pdf_dir) -> str:
    """Introduction that spans many pages and IS the start of study content."""
    path = os.path.join(test_pdf_dir, "intro_as_body.pdf")
    doc = fitz.open()

    _FILLER = "Real study content. " * 30

    for i in range(10):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), _FILLER, fontsize=12)

    doc.set_toc([
        [1, "Copyright", 1],
        [1, "Contents", 2],
        [1, "Introduction", 3],
        [1, "Chapter 1: Basics", 6],
        [1, "Chapter 2: Advanced", 8],
        [1, "Chapter 3: Expert", 10],
    ])
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# PDF where Introduction is front matter (short, followed by strong chapter)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_intro_as_front(test_pdf_dir) -> str:
    """Short Introduction that IS front matter (precedes Part I / Chapter 1)."""
    path = os.path.join(test_pdf_dir, "intro_as_front.pdf")
    doc = fitz.open()

    _FILLER = "Study content. " * 30

    for i in range(7):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), _FILLER, fontsize=12)

    doc.set_toc([
        [1, "Title Page", 1],
        [1, "Copyright", 1],
        [1, "Contents", 2],
        [1, "Introduction", 3],
        [1, "Part I: Foundations", 4],
        [1, "Chapter 1", 5],
        [1, "Chapter 2", 7],
    ])
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# PDF with NO chapter format (generic sections)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_no_chapters(test_pdf_dir) -> str:
    """A textbook with no explicit Chapter/Part markers, just topic titles."""
    path = os.path.join(test_pdf_dir, "no_chapters.pdf")
    doc = fitz.open()

    _FILLER = "Content. " * 30

    for i in range(4):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), _FILLER, fontsize=12)

    doc.set_toc([
        [1, "Overview", 1],
        [1, "Basic Concepts", 2],
        [1, "Advanced Topics", 3],
        [1, "Summary", 4],
    ])
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# PDF with bare-numbered chapters (textbook style: "1 Title", "2 Title")
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_bare_number_chapters(test_pdf_dir) -> str:
    """Textbook with bare-numbered chapters like '1 Music in Antiquity'."""
    path = os.path.join(test_pdf_dir, "bare_number_chapters.pdf")
    doc = fitz.open()

    _FILLER = "Chapter content. " * 30

    for i in range(8):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), _FILLER, fontsize=12)

    doc.set_toc([
        [1, "Preface", 1],
        [1, "Contents", 2],
        [1, "1 Getting Started", 3],
        [1, "2 Core Principles", 4],
        [1, "3 Applications", 5],
        [1, "4 Case Studies", 6],
        [1, "5 Future Directions", 8],
    ])
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# PDF with nested TOC (mimics AI Engineering: Chapter → Section → Subsection)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_nested_toc(test_pdf_dir) -> str:
    """AI Engineering style nested TOC with 3-level hierarchy.

    Structure:
        Cover (page 1)
        Copyright (page 2)
        Table of Contents (page 3)
        Preface (page 4)
        Chapter 1: Introduction to Building AI Applications (page 5, has_children)
            The Rise of AI Engineering (page 6, has_children)
                From Language Models to Large Language Models (page 7, leaf)
            Skills for AI Engineers (page 8, leaf)
        Chapter 2: Understanding Foundation Models (page 9, leaf)
    """
    path = os.path.join(test_pdf_dir, "nested_toc.pdf")
    doc = fitz.open()

    _FILLER = "This is real AI engineering study content. " * 30

    for i in range(10):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), _FILLER, fontsize=12)

    doc.set_toc([
        [1, "Cover", 1],
        [1, "Copyright", 2],
        [1, "Table of Contents", 3],
        [1, "Preface", 4],
        [1, "Chapter 1: Introduction to Building AI Applications", 5],
        [2, "The Rise of AI Engineering", 6],
        [3, "From Language Models to Large Language Models", 7],
        [2, "Skills for AI Engineers", 8],
        [1, "Chapter 2: Understanding Foundation Models", 9],
    ])
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# PDF with same-page parent/child (parent and first child on same page)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_same_page_parent_child(test_pdf_dir) -> str:
    """Parent and first child share a page — parent intro should be skipped."""
    path = os.path.join(test_pdf_dir, "same_page_parent_child.pdf")
    doc = fitz.open()

    page_texts = [
        "Cover page text. " * 30,
        "Shared parent-child page text. " * 30,   # page 2: both parent & child
        "Section 1.2 only content. " * 30,        # page 3: second child
        "Chapter 2 standalone content. " * 30,    # page 4: leaf chapter
    ]

    for text in page_texts:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), text, fontsize=12)

    doc.set_toc([
        [1, "Cover", 1],
        [1, "Chapter 1: Getting Started", 2],   # parent
        [2, "Section 1.1 Overview", 2],          # child — SAME page as parent
        [2, "Section 1.2 Details", 3],           # next child
        [1, "Chapter 2: Advanced", 4],
    ])
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# PDF with front-matter parent having children (Preface subtree)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_preface_with_children(test_pdf_dir) -> str:
    """Preface has child entries — entire subtree should be skipped."""
    path = os.path.join(test_pdf_dir, "preface_subtree.pdf")
    doc = fitz.open()

    _FILLER = "Study content. " * 30

    for i in range(8):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), _FILLER, fontsize=12)

    doc.set_toc([
        [1, "Cover", 1],
        [1, "Copyright", 2],
        [1, "Preface", 3],                       # front matter parent
        [2, "What This Book Is About", 3],       # child of Preface
        [2, "How to Use This Book", 4],          # child of Preface
        [1, "Chapter 1: Introduction", 5],
        [2, "Section 1.1", 6],
        [1, "Chapter 2: Core Concepts", 7],
    ])
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# PDF with wrapper root + full front/back matter (Western Music style)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_wrapper_with_front_and_back(test_pdf_dir) -> str:
    """Mimics the full A History of Western Music structure:
    book-title wrapper → front matter → body → back matter.
    """
    path = os.path.join(test_pdf_dir, "wrapper_front_back.pdf")
    doc = fitz.open()

    _FILLER = "Body content for learning. " * 30

    for i in range(20):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), _FILLER, fontsize=12)

    doc.set_toc([
        # Synthetic wrapper
        [1, "A HISTORY OF WESTERN MUSIC, 10E", 1],
        # Front matter children
        [2, "TITLE PAGE", 1],
        [2, "COPYRIGHT", 2],
        [2, "EDITORIAL ADVISORY BOARD", 3],
        [2, "MAPS", 4],
        [2, "GUIDE TO RECORDINGS", 5],
        [2, "PREFACE TO THE EDITION", 6],
        # Body
        [1, "PART ONE: ANCIENT WORLDS", 8],
        [2, "1. Music in Antiquity", 9],
        [2, "2. The Christian Church", 10],
        [1, "PART TWO: RENAISSANCE", 12],
        [2, "7. Music and the Renaissance", 13],
        [3, "Europe from 1400 to 1600", 14],
        # Back matter
        [1, "GLOSSARY", 17],
        [1, "CREDITS", 18],
        [1, "INDEX", 19],
        [1, "BIBLIOGRAPHY", 20],
    ])
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# PDF with back matter only at end (bibliography, index, glossary)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_with_back_matter(test_pdf_dir) -> str:
    """PDF with front matter, body chapters, then back matter."""
    path = os.path.join(test_pdf_dir, "back_matter.pdf")
    doc = fitz.open()

    _FILLER = "Study material. " * 30

    for i in range(15):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), _FILLER, fontsize=12)

    doc.set_toc([
        [1, "Cover", 1],
        [1, "Copyright", 2],
        [1, "Table of Contents", 3],
        [1, "Chapter 1: Basics", 5],
        [2, "Section 1.1", 6],
        [1, "Chapter 2: Advanced", 8],
        [2, "Section 2.1", 9],
        [1, "Chapter 3: Expert", 11],
        [1, "Bibliography", 13],
        [1, "References", 13],
        [1, "Index", 14],
        [1, "Glossary", 14],
        [1, "Credits", 15],
        [1, "About the Author", 15],
        [1, "Colophon", 15],
    ])
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# No-TOC PDF with front/back pages (for token fallback testing)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_no_toc_with_front_back(test_pdf_dir) -> str:
    """PDF without TOC but with obvious front/back matter on pages."""
    path = os.path.join(test_pdf_dir, "no_toc_front_back.pdf")
    doc = fitz.open()

    pages_content = [
        # Page 1: Copyright
        "Copyright © 2024. All Rights Reserved. ISBN 978-0-xxx-xxxxx-x. "
        "Library of Congress Cataloging-in-Publication Data. "
        "Printed in the United States of America. " * 5,

        # Page 2: Table of Contents
        "Table of Contents  Chapter 1 ......... 3  Chapter 2 ........ 5  "
        "Chapter 3 ......... 8  Index ........... 10 " * 5,

        # Pages 3-6: Body content
        "Machine learning is a field of study that gives computers the "
        "ability to learn without being explicitly programmed. "
        "This chapter introduces the core concepts. " * 20,
        "Supervised learning involves training a model on labeled data. "
        "The model learns to map inputs to outputs. " * 20,
        "Neural networks are composed of layers of interconnected nodes. "
        "Each connection has a weight that is adjusted during training. " * 20,
        "Deep learning uses many layers to learn hierarchical representations. "
        "This enables state-of-the-art performance on many tasks. " * 20,

        # Pages 7-8: Index/References
        "Index  A  accuracy ......... 45  activation function ...... 22  "
        "adagrad ........... 30  batch normalization ..... 25  " * 10,
        "Bibliography  [1] Smith, J. Machine Learning Fundamentals. 2023.  "
        "[2] Johnson, K. Deep Learning in Practice. 2024.  "
        "[3] Lee, S. Neural Networks and Beyond. 2025.  " * 10,
    ]

    for text in pages_content:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), text, fontsize=12)

    doc.save(path)
    doc.close()
    return path

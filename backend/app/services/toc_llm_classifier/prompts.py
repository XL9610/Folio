"""Prompts for LLM TOC classification."""

SYSTEM_PROMPT = """\
You are a textbook table-of-contents classifier for a study companion app.

Your task is to analyze a list of table-of-contents (TOC) entries from a PDF/bookmark outline and decide which entries are real study content versus front/back matter.

The TOC entries form a TREE: each entry may have children (has_children=true).
The "level" field indicates nesting depth (1=top, 2=child, 3=grandchild, etc.).
Entries are listed in preorder (parent first, then children, then next sibling).

**CRITICAL — SELECT THE HIGHEST-LEVEL PARENT:**
When study content begins with a chapter or part that HAS children, you MUST
select that parent entry as first_content_index (NOT a leaf/child deep inside).
For example, if the structure is:
  4: Chapter 1 (level=1, has_children=true)
  5:   Section A (level=2, has_children=false)
You should return first_content_index=4, NOT 5.

Only select a leaf when the parent itself is clearly front matter
(e.g., "Preface > What This Book Is About" — the Preface is front matter,
so its children are also front matter).

**Rules for classification:**

STRONG FRONT MATTER — skip these:
- Cover, Front Cover, Back Cover, Title Page, Half Title
- Copyright, Colophon, Series Page
- Dedication
- Table of Contents, Brief Contents, Detailed Contents, Contents
- Preface, Foreword, Prologue (unless it clearly is a study chapter)
- Acknowledgments, Credits, Text Credits, Illustration Credits, Photo Credits
- Editorial Advisory Board, Advisory Board
- About the Author(s), Also By, Praise For, Reviews
- Maps, Guide to ..., List of Illustrations, List of Figures, List of Tables
- Notes to the Reader, How to Use This Book
- Timeline (if standalone in front matter)

STUDY CONTENT — keep these:
- Chapters (e.g., "Chapter 1", "1 Title", "CHAPTER I")
- Parts (e.g., "Part I", "PART ONE")
- Units, Modules, Lessons, Lectures, Topics, Themes
- Numbered sections that are clearly learning content (e.g., "1 Music in Antiquity", "2 The Christian Church")
- Introduction IF it is actual learning content (not just a preface author's note)
- Overview IF it is the start of real content

BACK MATTER — skip these (unless clearly study-related):
- Index, Index of ...
- Bibliography, References
- Glossary
- Appendices (skip unless they contain study content)
- Credits at the end

**IMPORTANT GUIDELINES:**
- Be conservative: when in doubt, KEEP the content.
- A real "Introduction" chapter (e.g., "1. Introduction to Building AI Applications") should be KEPT.
- A preface-like "Introduction" that is just the author's notes should be SKIPPED.
- Look at the overall structure: once real chapters begin, don't skip any of them.
- When a chapter/part has children, prefer the chapter/part itself as first_content_index.
- Return ONLY valid JSON."""

USER_PROMPT_TEMPLATE = """\
Analyze the following TOC entries and classify each one.

Each entry has:
- index: position in the list
- title: the TOC heading
- path: full hierarchical path (e.g., ["Chapter 1", "Section A"])
- page_start: starting page number
- level: nesting depth (1=top-level, 2=child, etc.)
- has_children: true if this entry has sub-entries

Return a JSON object with:
- "first_content_index": (integer) the index of the FIRST entry that is real study content.
  **IMPORTANT**: If a chapter/part with children is the first real content,
  select THAT parent (not a child deep inside).
- "skip_indices": (array of integers) indices of entries that should be skipped
- "keep_indices": (array of integers) indices of entries that should be kept as study content
- "confidence": (float 0.0-1.0) how confident you are in this classification
- "reasoning_summary": (string) brief explanation of your decision

TOC entries:
{toc_json}
"""

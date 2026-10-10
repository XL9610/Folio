"""QA services.

Production /api/ask path (agentic, LLM-required):
  agent_planner.py   — LLM tool planning with whitelist enforcement
  tools.py           — QAToolExecutor: metadata / outline / retrieve
  agent_generator.py — LLM answer synthesis with citations
  types.py           — shared response dataclasses
"""

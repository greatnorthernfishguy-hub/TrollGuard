"""
TrollGuard — AST Extractor

Extracts text segments from Python source where prompt injections
typically hide: docstrings, comments, string literals (especially
those passed to exec/eval/subprocess), variable names, and dynamically
constructed strings.

For non-Python files (.yaml, .md), extracts full text content directly.

PRD reference: Section 5.2 — Text Extraction via AST Parsing

# ---- Changelog ----
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: ASTExtractor class using Python's built-in ast and tokenize
#         modules to extract injectable text segments from source code.
#   Why:  Vectorizing entire files as raw strings wastes compute and
#         dilutes threat signals.  AST-targeted extraction focuses
#         analysis on the attack surface — the places where human-
#         language instructions can hide in code.
#   Settings: None user-configurable; extraction targets are hard-coded
#         per PRD §5.2 (docstrings, comments, string literals,
#         identifiers, dynamic string patterns).
#   How:  Uses ast.parse() for structural extraction and tokenize for
#         comments (which ast doesn't capture).  Falls back to raw
#         text for non-Python files.
# -------------------
"""

from __future__ import annotations

import ast
import io
import logging
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("trollguard.ast_extractor")


@dataclass
class ExtractionResult:
    """Result of text extraction from a source file.

    Attributes:
        file_path: Path to the source file.
        file_type: Detected type ("python", "yaml", "markdown", "text").
        segments: List of extracted text segments with metadata.
        total_segments: Count of extracted segments.
        extraction_method: "ast" for Python, "raw" for other formats.
    """
    file_path: str = ""
    file_type: str = "text"
    segments: List[Dict[str, Any]] = field(default_factory=list)
    total_segments: int = 0
    extraction_method: str = "raw"


class ASTExtractor:
    """Extract injectable text segments from source files.

    Usage:
        extractor = ASTExtractor()

        # Extract from a Python file
        result = extractor.extract_file("skill.py")
        for segment in result.segments:
            print(segment["type"], segment["text"][:80])

        # Extract from raw text
        result = extractor.extract_text("some yaml content", file_type="yaml")
    """

    # Dangerous function names whose string arguments warrant extraction
    DANGEROUS_CALLS = {"exec", "eval", "compile", "subprocess", "os.system",
                       "os.popen", "Popen", "call", "check_output", "run"}

    def extract_file(self, file_path: str) -> ExtractionResult:
        """Extract text segments from a file on disk.

        Args:
            file_path: Path to the source file.

        Returns:
            ExtractionResult with extracted segments.
        """
        p = Path(file_path)
        if not p.exists():
            return ExtractionResult(
                file_path=file_path,
                segments=[{"type": "error", "text": f"File not found: {file_path}"}],
            )

        content = p.read_text(errors="replace")
        suffix = p.suffix.lower()

        if suffix == ".py":
            return self._extract_python(content, file_path)
        elif suffix in (".yaml", ".yml"):
            return self._extract_raw(content, file_path, "yaml")
        elif suffix in (".md", ".markdown"):
            return self._extract_raw(content, file_path, "markdown")
        else:
            return self._extract_raw(content, file_path, "text")

    def extract_text(
        self,
        content: str,
        file_type: str = "text",
        file_path: str = "<string>",
    ) -> ExtractionResult:
        """Extract text segments from a string.

        Args:
            content: The text content to extract from.
            file_type: "python", "yaml", "markdown", or "text".
            file_path: Label for the source (for logging).

        Returns:
            ExtractionResult with extracted segments.
        """
        if file_type == "python":
            return self._extract_python(content, file_path)
        return self._extract_raw(content, file_path, file_type)

    # -------------------------------------------------------------------
    # Python-specific extraction
    # -------------------------------------------------------------------

    def _extract_python(self, source: str, file_path: str) -> ExtractionResult:
        """Extract injectable segments from Python source using AST."""
        segments: List[Dict[str, Any]] = []

        # 1. AST-based extraction (docstrings, string literals, identifiers)
        try:
            tree = ast.parse(source)
            self._walk_ast(tree, segments)
        except SyntaxError as e:
            logger.warning("AST parse failed for %s: %s (falling back to raw)", file_path, e)
            return self._extract_raw(source, file_path, "python")

        # 2. Tokenize-based extraction (comments — ast doesn't capture these)
        try:
            self._extract_comments(source, segments)
        except tokenize.TokenError as e:
            logger.warning("Tokenize failed for %s: %s", file_path, e)

        return ExtractionResult(
            file_path=file_path,
            file_type="python",
            segments=segments,
            total_segments=len(segments),
            extraction_method="ast",
        )

    def _walk_ast(self, tree: ast.AST, segments: List[Dict[str, Any]]) -> None:
        """Walk the AST and extract docstrings, string literals, identifiers."""
        for node in ast.walk(tree):
            # Docstrings (module, class, function level)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    name = getattr(node, "name", "<module>")
                    segments.append({
                        "type": "docstring",
                        "text": docstring,
                        "context": f"docstring of {name}",
                        "line": getattr(node, "lineno", 0),
                    })

            # String literals (especially in dangerous calls)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Check if this string is an argument to a dangerous function
                text = node.value.strip()
                if len(text) > 3:  # Skip trivially short strings
                    segments.append({
                        "type": "string_literal",
                        "text": text,
                        "context": "string constant",
                        "line": getattr(node, "lineno", 0),
                    })

            # Function/class names (attacks encoded in identifiers)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                segments.append({
                    "type": "identifier",
                    "text": node.name,
                    "context": "function name",
                    "line": node.lineno,
                })
            elif isinstance(node, ast.ClassDef):
                segments.append({
                    "type": "identifier",
                    "text": node.name,
                    "context": "class name",
                    "line": node.lineno,
                })

            # Dangerous calls: extract string arguments to exec/eval/etc.
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name in self.DANGEROUS_CALLS:
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            segments.append({
                                "type": "dangerous_call_arg",
                                "text": arg.value,
                                "context": f"argument to {func_name}()",
                                "line": getattr(node, "lineno", 0),
                            })

    @staticmethod
    def _get_call_name(node: ast.Call) -> str:
        """Extract the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _extract_comments(self, source: str, segments: List[Dict[str, Any]]) -> None:
        """Extract comments using the tokenize module."""
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok_type, tok_string, start, _, _ in tokens:
            if tok_type == tokenize.COMMENT:
                # Strip the leading #
                comment_text = tok_string.lstrip("#").strip()
                if len(comment_text) > 3:
                    segments.append({
                        "type": "comment",
                        "text": comment_text,
                        "context": "inline comment",
                        "line": start[0],
                    })

    # -------------------------------------------------------------------
    # Raw text extraction (non-Python files)
    # -------------------------------------------------------------------

    def _extract_raw(
        self, content: str, file_path: str, file_type: str
    ) -> ExtractionResult:
        """Extract full text content for non-Python files."""
        segments = [{
            "type": "raw_text",
            "text": content,
            "context": f"full {file_type} content",
            "line": 1,
        }]

        return ExtractionResult(
            file_path=file_path,
            file_type=file_type,
            segments=segments,
            total_segments=1,
            extraction_method="raw",
        )

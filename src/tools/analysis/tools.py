"""Analysis tools — extend OpenManus BaseTool."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from src._framework import BaseTool, ToolResult


class PythonExecuteTool(BaseTool):
    """Execute Python code in a sandboxed subprocess."""

    name: str = "python_execute"
    description: str = (
        "Execute Python code and return stdout/stderr. "
        "Use for computation, data analysis, text processing. "
        "Write results to stdout with print()."
    )
    parameters: Optional[dict] = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source code to execute"},
        },
        "required": ["code"],
    }

    def __init__(self, timeout: int = 60, work_dir: str = "./data/workspace", **data):
        super().__init__(**data)
        self._timeout = timeout
        self._work_dir = Path(work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)

    async def execute(self, **kwargs) -> ToolResult:
        code = kwargs.get("code", "")

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", dir=str(self._work_dir), delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                script_path = f.name

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=self._timeout, cwd=str(self._work_dir),
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"

            Path(script_path).unlink(missing_ok=True)
            return ToolResult(output=output.strip() or "(no output)")
        except subprocess.TimeoutExpired:
            return ToolResult(error=f"Execution timed out after {self._timeout}s")
        except Exception as e:
            return ToolResult(error=str(e))

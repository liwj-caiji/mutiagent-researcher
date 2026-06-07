# Attribution

This directory contains code vendored (copied and adapted) from
**[OpenManus](https://github.com/FoundationAgents/OpenManus)**
by the OpenManus contributors, used under the **MIT License**.

## Modifications

- BaseAgent.llm: replaced `app.llm.LLM` type with a type-less `Any` field, injected via `__dict__` by the LangGraph workflow layer.
- Framework integrated into `src/_framework/` as a sub-package (not a git submodule).
- Minor structural adjustments for compatibility with this project's dependency stack.

## Original License (MIT)

Copyright (c) 2025 OpenManus contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

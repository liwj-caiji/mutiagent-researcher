# 多智能体深度研究助手

**Python · LangGraph · MCP · Anthropic/OpenAI SDK · asyncio**

基于 LangGraph 编排 6 个 ReAct Agent（Planner → Searcher → Analyst → Synthesizer → Writer → Critic）协作完成研究主题的分解、搜索、分析、撰写与评审，质量不达标自动回退重试，支持人工审核介入。

- 从 OpenManus 裁剪 ~950 行 Agent 基座（BaseAgent → ToolCallAgent），去除 264 个外部依赖；封装 4 种 LLM Provider 原生 SDK，处理 Anthropic（tool_use content blocks）与 OpenAI（tool_calls 数组）消息格式差异
- asyncio.gather + Semaphore 实现同质查询并行搜索，工厂函数隔离 Agent 实例，单查询异常不影响全局
- LangGraph interrupt() 实现 Human-in-the-Loop 人工审核，Rich Live 暂停/恢复解决终端输入覆盖问题
- Critic 6 维度评分 + Supervisor 条件路由形成质量闭环；花括号深度计数解决 LLM 嵌套 JSON 解析失败

**难点**：quality_score 恒为 0（LLM 直接 Terminate）→ 增大步数 + 请求强指令；LangGraph edge 函数状态修改被静默丢弃 → 移至 Node 返回值

from .tool_base import BaseTool


class Terminate(BaseTool):
    name: str = "terminate"
    description: str = (
        "Terminate the interaction when the request is met OR if the assistant "
        "cannot proceed further with the task. When you have finished all the "
        "tasks, call this tool to end the work."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "The finish status of the interaction.",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    async def execute(self, status: str) -> str:
        return f"The interaction has been completed with status: {status}"

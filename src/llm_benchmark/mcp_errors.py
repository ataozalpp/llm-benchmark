"""Infrastructure failures during MCP tool execution."""


class MCPExecutionError(RuntimeError):
    """Base class for MCP execution infrastructure failures."""


class McpExecutionTimeout(MCPExecutionError):
    """The MCP tool request exceeded its time budget."""


class MCPTransportFailure(MCPExecutionError):
    """The MCP connection could not carry the tool request."""


class MCPProtocolFailure(MCPExecutionError):
    """The peer rejected the request at the protocol level."""

    def __init__(self, *, code: int) -> None:
        self.code = code
        super().__init__(f"MCP protocol failure: code={code}.")
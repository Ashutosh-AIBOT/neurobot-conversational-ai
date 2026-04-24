import json
import logging
import os
import subprocess
import threading
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.neurobot_settings import get_settings


logger = logging.getLogger("NeuroBotMCP")


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str]
    working_dir: Path
    startup_timeout_seconds: float
    tool_timeout_seconds: float


class McpError(RuntimeError):
    pass


class McpClient:
    def __init__(self, config: McpServerConfig):
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._started = False

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        with self._lock:
            self._ensure_started()
            tools = self._request("tools/list", {})
            available_tools = {
                tool["name"]: tool
                for tool in tools.get("tools", [])
                if isinstance(tool, dict) and tool.get("name")
            }
            if tool_name not in available_tools:
                available = ", ".join(sorted(available_tools)) or "none"
                raise McpError(
                    f"MCP tool '{tool_name}' is not exposed by server '{self.config.name}'. "
                    f"Available tools: {available}."
                )

            result = self._request(
                "tools/call",
                {
                    "name": tool_name,
                    "arguments": arguments or {},
                },
                timeout=self.config.tool_timeout_seconds,
            )
            return self._format_tool_result(result)

    def close(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._process = None
        self._started = False

    def _ensure_started(self) -> None:
        if self._started and self._process and self._process.poll() is None:
            return

        env = os.environ.copy()
        env.update(self.config.env)
        self._process = subprocess.Popen(
            [self.config.command, *self.config.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(self.config.working_dir),
        )
        initialize_result = self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "neurobot", "version": "1.0"},
            },
            timeout=self.config.startup_timeout_seconds,
        )
        logger.info("Connected to MCP server '%s': %s", self.config.name, initialize_result)
        self._notify(
            "notifications/initialized",
            {},
        )
        self._started = True

    def _request(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        process = self._require_process()
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        self._send_message(request)
        while True:
            response = self._read_message(timeout=timeout)
            if response.get("id") != self._request_id:
                continue
            break
        if "error" in response:
            raise McpError(f"MCP error for '{method}': {response['error']}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise McpError(f"MCP result for '{method}' was not a JSON object.")
        return result

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        self._send_message(message)

    def _send_message(self, message: dict[str, Any]) -> None:
        process = self._require_process()
        payload = (json.dumps(message) + "\n").encode("utf-8")
        assert process.stdin is not None
        process.stdin.write(payload)
        process.stdin.flush()

    def _read_message(self, timeout: float | None = None) -> dict[str, Any]:
        process = self._require_process()
        assert process.stdout is not None
        
        # We don't use the timer here for simplicity in line reading, 
        # but the subprocess call should ideally be async or handled with select.
        # For now, standard line reading is more compatible with FastMCP.
        line = process.stdout.readline()
        if not line:
            stderr_output = b""
            if process.stderr is not None:
                stderr_output = process.stderr.read1(4096)
            raise McpError(
                "MCP server closed the connection unexpectedly. "
                f"stderr: {stderr_output.decode('utf-8', errors='ignore')}"
            )
        return json.loads(line.decode("utf-8"))

    def _kill_for_timeout(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.kill()

    def _require_process(self) -> subprocess.Popen[bytes]:
        if not self._process or self._process.poll() is not None:
            raise McpError("MCP process is not running.")
        return self._process

    @staticmethod
    def _format_tool_result(result: dict[str, Any]) -> str:
        content = result.get("content", [])
        if isinstance(content, list) and content:
            rendered_parts = []
            for item in content:
                if not isinstance(item, dict):
                    rendered_parts.append(str(item))
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    rendered_parts.append(item.get("text", ""))
                elif item_type == "json":
                    rendered_parts.append(json.dumps(item.get("json", {}), indent=2))
                else:
                    rendered_parts.append(json.dumps(item, indent=2))
            return "\n\n".join(part for part in rendered_parts if part).strip()

        if "structuredContent" in result:
            return json.dumps(result["structuredContent"], indent=2)

        return json.dumps(result, indent=2)


_CLIENTS: dict[str, McpClient] = {}


def _parse_server_config(server_name: str) -> McpServerConfig:
    settings = get_settings()
    server = settings.mcp_servers.get(server_name)
    if not isinstance(server, dict):
        if server_name in {settings.default_mcp_server_name, "default"}:
            base_dir = Path(__file__).resolve().parents[1]
            return McpServerConfig(
                name=settings.default_mcp_server_name,
                command=sys.executable,
                args=["-m", "src.neurobot_mcp_server"],
                env={},
                working_dir=base_dir,
                startup_timeout_seconds=8.0,
                tool_timeout_seconds=20.0,
            )
        configured = ", ".join(sorted(settings.mcp_servers)) or "none"
        raise McpError(
            f"MCP server '{server_name}' is not configured. Configured servers: {configured}."
        )

    command = server.get("command")
    if not isinstance(command, str) or not command.strip():
        raise McpError(f"MCP server '{server_name}' is missing a command.")

    args = server.get("args", [])
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise McpError(f"MCP server '{server_name}' has invalid args.")

    env = server.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env.items()
    ):
        raise McpError(f"MCP server '{server_name}' has invalid env values.")

    startup_timeout_seconds = float(server.get("startup_timeout_seconds", 8))
    tool_timeout_seconds = float(server.get("tool_timeout_seconds", 20))
    working_dir_value = server.get("working_dir")
    if working_dir_value is not None and not isinstance(working_dir_value, str):
        raise McpError(f"MCP server '{server_name}' has invalid working_dir.")
    base_dir = Path(__file__).resolve().parents[1]
    working_dir = Path(working_dir_value).expanduser().resolve() if working_dir_value else base_dir
    return McpServerConfig(
        name=server_name,
        command=command,
        args=args,
        env=env,
        working_dir=working_dir,
        startup_timeout_seconds=startup_timeout_seconds,
        tool_timeout_seconds=tool_timeout_seconds,
    )


def call_mcp_tool(server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
    client = _CLIENTS.get(server_name)
    if client is None:
        client = McpClient(_parse_server_config(server_name))
        _CLIENTS[server_name] = client
    return client.call_tool(tool_name=tool_name, arguments=arguments)

import src.neurobot_settings as neurobot_settings
from src.neurobot_settings import get_settings


def test_settings_have_runtime_paths():
    settings = get_settings()
    assert settings.runtime_dir.name == "runtime"
    assert settings.vector_store_dir.parent == settings.runtime_dir
    assert settings.checkpoint_path.parent == settings.runtime_dir
    assert settings.default_mcp_server_name == "neurobot"


def test_settings_parse_mcp_server_json(monkeypatch):
    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        '{"filesystem":{"command":"npx","args":["-y","server"],"env":{"ROOT":"/tmp"}}}',
    )
    neurobot_settings._SETTINGS = None

    settings = get_settings()

    assert settings.mcp_servers["filesystem"]["command"] == "npx"
    assert settings.mcp_servers["filesystem"]["args"] == ["-y", "server"]
    assert settings.mcp_servers["filesystem"]["env"]["ROOT"] == "/tmp"

    monkeypatch.delenv("MCP_SERVERS_JSON", raising=False)
    neurobot_settings._SETTINGS = None

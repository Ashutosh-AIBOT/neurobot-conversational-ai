from src.neurobot_settings import get_settings


def test_session_namespace_is_scoped_by_tenant():
    settings = get_settings()
    a = settings.session_namespace("acme", "chat-1")
    b = settings.session_namespace("globex", "chat-1")
    assert a != b
    assert a.startswith("acme:")

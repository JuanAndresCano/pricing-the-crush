import pytest

from crush.data.massive_client import MassiveClient, MassiveError


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.responses.pop(0)


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setattr("crush.data.massive_client.load_dotenv", lambda: None)
    with pytest.raises(MassiveError):
        MassiveClient()


def test_follows_next_url_and_sends_key_in_header():
    session = FakeSession([
        FakeResponse(200, {"results": [{"ticker": "A"}], "next_url": "https://x.test/next"}),
        FakeResponse(200, {"results": [{"ticker": "B"}]}),
    ])
    client = MassiveClient(api_key="k", base_url="https://x.test", session=session)
    out = client.option_contracts("AAPL", "2026-01-02")
    assert [c["ticker"] for c in out] == ["A", "B"]
    assert session.calls[0]["headers"] == {"Authorization": "Bearer k"}
    assert session.calls[0]["params"]["underlying_ticker"] == "AAPL"
    assert session.calls[1]["url"] == "https://x.test/next"


def test_http_error_message_does_not_leak_key():
    session = FakeSession([FakeResponse(403, {"message": "plan"})])
    client = MassiveClient(api_key="SECRET", base_url="https://x.test", session=session)
    with pytest.raises(MassiveError) as exc:
        client.option_quotes("O:AAPL261120C00200000", "2026-11-01")
    assert "SECRET" not in str(exc.value)
    assert "403" in str(exc.value)

"""Transport fault injection; fixture bytes are not biological observations."""
import hashlib
import io
import urllib.error
import pytest
from scripts import prepare_public_pilot as downloader


def entry(data):
    return {'file': 'fixture.gz', 'url': 'https://example.org/fixture',
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def test_timeout_restarts_partial_and_reuses_verified_cache(tmp_path, monkeypatch):
    data = b'verified fixture'
    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise TimeoutError('injected read timeout')
            return super().read(3)
    calls = []
    def open_url(request, timeout):
        calls.append(timeout)
        return Interrupted(data) if len(calls) == 1 else io.BytesIO(data)
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', open_url)
    monkeypatch.setattr(downloader.time, 'sleep', lambda seconds: None)
    path = downloader.fetch_locked(entry(data), tmp_path, True)
    assert path.read_bytes() == data and calls == [180, 180]
    assert downloader.fetch_locked(entry(data), tmp_path, False) == path
    assert len(calls) == 2


def test_truncated_transfer_retries(tmp_path, monkeypatch):
    data = b'complete'; responses = iter([io.BytesIO(b'part'), io.BytesIO(data)])
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *a, **k: next(responses))
    monkeypatch.setattr(downloader.time, 'sleep', lambda seconds: None)
    assert downloader.fetch_locked(entry(data), tmp_path, True).read_bytes() == data


def test_checksum_mismatch_is_not_promoted_or_retried(tmp_path, monkeypatch):
    calls = []
    def open_url(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b'bad')
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', open_url)
    with pytest.raises(ValueError, match='Source changed'):
        downloader.fetch_locked(entry(b'yes'), tmp_path, True)
    assert calls == [1] and not (tmp_path / 'fixture.gz').exists()


def test_persistent_timeout_stops_after_five_attempts(tmp_path, monkeypatch):
    calls = []
    def open_url(*args, **kwargs):
        calls.append(1)
        raise TimeoutError('injected')
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', open_url)
    monkeypatch.setattr(downloader.time, 'sleep', lambda seconds: None)
    with pytest.raises(TimeoutError):
        downloader.fetch_locked(entry(b'x'), tmp_path, True)
    assert len(calls) == 5 and not (tmp_path / 'fixture.gz').exists()


def test_404_does_not_retry(tmp_path, monkeypatch):
    calls = []
    def open_url(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError('https://example.org/fixture', 404, 'missing', {}, None)
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', open_url)
    with pytest.raises(urllib.error.HTTPError):
        downloader.fetch_locked(entry(b'x'), tmp_path, True)
    assert len(calls) == 1

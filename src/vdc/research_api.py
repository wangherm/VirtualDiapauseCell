"""Bounded OpenAI-compatible calls. No key in configs, artifacts or errors."""
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from urllib.parse import urlsplit
from .io import read_json, write_json, object_hash


def parse_object(text):
    text = text.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[8:-4]
    obj = json.loads(text)  # Never extract fields from truncated or partial JSON.
    if not isinstance(obj, dict): raise ValueError('Expected complete JSON object')
    return obj


def endpoint():
    base = os.environ.get('VDC_RESEARCH_API_BASE', '').rstrip('/')
    u = urlsplit(base)
    if (u.scheme != 'https' and not (u.scheme == 'http' and u.hostname in {'localhost', '127.0.0.1'})) or not u.hostname:
        raise ValueError('Set VDC_RESEARCH_API_BASE to HTTPS base URL ending in /v1 (or loopback HTTP)')
    if u.username or u.password or u.query or u.fragment:
        raise ValueError('API base must not contain credentials, query strings or fragments')
    if base.endswith('/chat/completions'):
        raise ValueError('Set API base only, without /chat/completions')
    if not os.environ.get('VDC_RESEARCH_API_KEY') or not os.environ.get('VDC_RESEARCH_MODEL'):
        raise ValueError('Set VDC_RESEARCH_API_KEY and VDC_RESEARCH_MODEL on the server')
    return base+'/chat/completions'


class Client:
    def __init__(self, run, config, retry_errors=False):
        self.folder = Path(run)/'api'; self.folder.mkdir(parents=True, exist_ok=True)
        self.config = config; self.retry_errors = retry_errors
        self.url = endpoint(); self.model = os.environ['VDC_RESEARCH_MODEL']

    def call(self, key, system, user, validator=None):
        payload = {'model': self.model, 'messages': [{'role':'system','content':system}, {'role':'user','content':user}],
                   'max_tokens': self.config['max_output_tokens']}
        if self.config.get('json_response_format', True): payload['response_format'] = {'type':'json_object'}
        signature = object_hash({'url': self.url, 'payload':payload})
        attempts = sorted(self.folder.glob(key+'_*.json'))
        for p in attempts:
            record = read_json(p)
            if record['signature'] != signature: raise ValueError('API prompt changed inside resumed run')
            if record['status'] == 'completed':
                parsed=parse_object(record['content'])
                if validator: validator(parsed)
                return parsed
        if attempts and not self.retry_errors:
            raise RuntimeError('Previous API error/uncertain delivery; inspect api ledger and use --retry-api-errors explicitly')
        ledger = [read_json(p) for p in self.folder.glob('*.json')]
        charge = (len(system.encode('utf-8'))+len(user.encode('utf-8')))+self.config['max_output_tokens']
        if len(ledger) >= self.config['max_api_calls'] or sum(r['budget_charge'] for r in ledger)+charge > self.config['max_token_budget']:
            raise RuntimeError('API call/token upper-bound budget exhausted; no further requests submitted')
        if charge > self.config['max_request_budget']:
            raise ValueError('Request exceeds configured context budget')
        secret = os.environ['VDC_RESEARCH_API_KEY']
        if secret in system or secret in user: raise ValueError('Credential detected in prompt')
        path = self.folder/f'{key}_{len(attempts)+1:03d}.json'
        record = {'signature':signature, 'status':'submitted_outcome_uncertain', 'model':self.model,
                  'messages':payload['messages'], 'budget_charge':charge, 'started_at':time.time()}
        write_json(path, record)  # Crash here is NOT an automatic retry/free call.
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                    headers={'Authorization':'Bearer '+secret, 'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.config['api_timeout_seconds']) as response:
                body = response.read(2_000_001)
            if len(body)>2_000_000: raise ValueError('API response too large')
            result = json.loads(body); choice = result['choices'][0]
            content = choice['message']['content']
            if not isinstance(content, str): raise ValueError('API did not return text content')
            content = content.replace(secret, '[REDACTED]')
            record.update(content=content, usage=result.get('usage'), finish_reason=choice.get('finish_reason'))
            if choice.get('finish_reason') != 'stop': raise ValueError('Generation did not finish normally; incomplete JSON not accepted')
            parsed = parse_object(content)
            if validator: validator(parsed)
            record['status'] = 'completed'; write_json(path, record)
            return parsed
        except Exception as exc:
            # Never persist arbitrary response bodies, request headers or provider error strings.
            record.update(status='failed', error_type=type(exc).__name__,
                          http_status=exc.code if isinstance(exc, urllib.error.HTTPError) else None)
            write_json(path, record)
            raise RuntimeError('Research API call failed; see redacted api ledger (no automatic retry)') from None

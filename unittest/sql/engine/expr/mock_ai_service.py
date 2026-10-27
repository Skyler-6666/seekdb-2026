"""Local protocol fixture. It never calls a model or an external endpoint."""
import argparse
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


def embedding(text, dimension):
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    return [round((digest[i % len(digest)] - 128) / 128, 6) for i in range(dimension)]


class MockAIService(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        super().__init__(address, Handler)
        self.lock = threading.Lock()
        self.requests = []
        self.attempts = {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def reply(self, value, status=200, partial=False, invalid_json=False):
        encoded = b'{invalid' if invalid_json else json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded[:len(encoded) // 2] if partial else encoded)
        self.wfile.flush()
        if partial:
            self.close_connection = True

    def do_GET(self):
        if self.path == '/__stats':
            with self.server.lock:
                records = list(self.server.requests)
            self.reply({'requests': records})
        else:
            self.reply({'error': 'unknown path'}, 404)

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 32 * 1024 * 1024:
                raise ValueError('invalid request length')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError('request must be an object')
            path = urlsplit(self.path)
            route = path.path.removeprefix('/compatible-mode')
            scenario = parse_qs(path.query).get('scenario', ['normal'])[0]
            with self.server.lock:
                self.server.attempts[self.path] = self.server.attempts.get(self.path, 0) + 1
                attempt = self.server.attempts[self.path]
                self.server.requests.append({'path': path.path, 'body': body})
            if path.path == '/__reset':
                with self.server.lock:
                    self.server.requests.clear()
                    self.server.attempts.clear()
                self.reply({'ok': True})
                return
            if scenario == 'http_500':
                self.reply({'error': 'requested test failure'}, 500)
                return
            if not isinstance(body.get('model'), str) or not body['model']:
                raise ValueError('model must be a nonempty string')
            if route in ('/v1/embeddings', '/api/embed'):
                inputs = body.get('input')
                if isinstance(inputs, str):
                    inputs = [inputs]
                dimension = body.get('dimensions', 768)
                if (not isinstance(inputs, list) or not inputs or
                        not all(isinstance(item, str) for item in inputs) or
                        type(dimension) is not int or not 0 < dimension <= 4096):
                    raise ValueError('invalid embedding request')
                vectors = [embedding(item, dimension) for item in inputs]
                if scenario == 'wrong_dimension':
                    vectors[-1].append(0)
                if scenario == 'non_numeric':
                    vectors[0][0] = 'invalid'
                if route == '/api/embed':
                    value = {'embeddings': vectors}
                else:
                    # Return unsorted local indexes to exercise row restoration.
                    data = [{'index': index, 'embedding': vector} for index, vector in enumerate(vectors)]
                    if scenario == 'duplicate_index' and len(data) > 1:
                        data[-1]['index'] = 0
                    if scenario == 'missing_result':
                        data.pop()
                    value = {'data': list(reversed(data))}
            elif route == '/api/generate':
                if body.get('stream') is not False or not isinstance(body.get('prompt'), str):
                    raise ValueError('Ollama generation must be non-streaming')
                options = body.get('options', {})
                if not isinstance(options, dict) or 'max_tokens' in body or 'temperature' in body:
                    raise ValueError('Ollama options must use the native protocol')
                value = {'response': 'mock:' + body['prompt']}
                if scenario == 'empty':
                    value['response'] = ''
            elif route == '/v1/chat/completions':
                messages = body.get('messages')
                if not isinstance(messages, list) or not messages or not isinstance(messages[-1].get('content'), str):
                    raise ValueError('invalid completion request')
                value = {'choices': [{'message': {'content': '' if scenario == 'empty' else 'mock:' + messages[-1]['content']}}]}
            elif route == '/v1/rerank':
                documents = body.get('documents')
                if not isinstance(documents, list) or not documents or not all(isinstance(item, str) for item in documents):
                    raise ValueError('documents must be a nonempty string array')
                query_tokens = set(body.get('query', '').casefold().split())
                results = [{'index': index, 'relevance_score':
                            1000 * len(query_tokens.intersection(item.casefold().split())) + len(item)}
                           for index, item in enumerate(documents)]
                if scenario == 'duplicate_index' and len(results) > 1:
                    results[-1]['index'] = 0
                if scenario == 'missing_result':
                    results.pop()
                value = {'results': list(reversed(results))}
            else:
                self.reply({'error': 'unknown path'}, 404)
                return
            self.reply(value, partial=(scenario == 'partial_once' and attempt == 1), invalid_json=scenario == 'invalid_json')
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            self.reply({'error': str(error)}, 400)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    server = MockAIService((args.host, args.port))
    print('Mock AI Service ready on %s:%d; no real model calls' % server.server_address, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

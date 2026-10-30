import json
import threading
import unittest
import urllib.error
import urllib.request
from mock_ai_service import MockAIService, embedding


class MockProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = MockAIService(('127.0.0.1', 0))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = 'http://127.0.0.1:%d' % cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, path, body):
        request = urllib.request.Request(self.url + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.load(response)

    def test_openai_unordered_embedding(self):
        inputs = ['汉字', 'a\\b"c', 'x' * 100000]
        result = self.post('/v1/embeddings', {'model': 'arbitrary-model', 'input': inputs, 'dimensions': 7})
        self.assertEqual([2, 1, 0], [item['index'] for item in result['data']])
        for item in result['data']:
            self.assertEqual(embedding(inputs[item['index']], 7), item['embedding'])

    def test_ollama_protocol(self):
        result = self.post('/api/generate', {'model': 'any', 'prompt': 'test', 'stream': False, 'options': {'num_predict': 10}})
        self.assertEqual('mock:test', result['response'])
        vectors = self.post('/api/embed', {'model': 'any', 'input': ['a', 'b'], 'dimensions': 3})['embeddings']
        self.assertEqual([embedding('a', 3), embedding('b', 3)], vectors)

    def test_ollama_rejects_streaming_and_wrong_options(self):
        for body in ({'model': 'any', 'prompt': 'a', 'stream': True},
                     {'model': 'any', 'prompt': 'a', 'stream': False, 'temperature': 1}):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.post('/api/generate', body)
            self.assertEqual(400, error.exception.code)

    def test_rerank_has_stable_ties_to_restore(self):
        result = self.post('/v1/rerank', {'model': 'any', 'query': 'q', 'documents': ['a', 'bbbb', 'c']})
        self.assertEqual([2, 1, 0], [item['index'] for item in result['results']])
        self.assertEqual([1, 4, 1], [item['relevance_score'] for item in result['results']])

    def test_contest_endpoint_alias_and_default_dimension(self):
        result = self.post('/compatible-mode/v1/embeddings', {'model': 'any', 'input': ['a']})
        self.assertEqual(embedding('a', 768), result['data'][0]['embedding'])
        result = self.post('/v1/rerank', {'model': 'any', 'query': 'apple', 'documents': ['apple', 'banana']})
        scores = {item['index']: item['relevance_score'] for item in result['results']}
        self.assertGreater(scores[0], scores[1])

    def test_empty_and_malformed_scenarios(self):
        body = {'model': 'any', 'messages': [{'role': 'user', 'content': 'q'}]}
        result = self.post('/v1/chat/completions?scenario=empty', body)
        self.assertEqual('', result['choices'][0]['message']['content'])
        with self.assertRaises(json.JSONDecodeError):
            self.post('/v1/chat/completions?scenario=invalid_json', body)

    def test_invalid_vectors_and_response_counts(self):
        body = {'model': 'any', 'input': ['a', 'b']}
        bad = self.post('/v1/embeddings?scenario=duplicate_index', body)['data']
        self.assertEqual([0, 0], [item['index'] for item in bad])
        self.assertEqual(1, len(self.post('/v1/embeddings?scenario=missing_result', body)['data']))
        self.assertEqual('invalid', self.post('/v1/embeddings?scenario=non_numeric', body)['data'][-1]['embedding'][0])


if __name__ == '__main__':
    unittest.main()

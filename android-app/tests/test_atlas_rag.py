import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient


MODULE_PATH = Path(__file__).parents[1] / "app/src/main/python/atlas_rag.py"
SPEC = importlib.util.spec_from_file_location("atlas_rag", MODULE_PATH)
atlas_rag = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = atlas_rag
SPEC.loader.exec_module(atlas_rag)


class FakeRuntime:
    def __init__(self):
        self.embeddings = []
        self.saved = []
        self.saved_batches = []
        self.queries = []
        self.assets = tempfile.TemporaryDirectory()
        self.started = False
        self.stopped = False

    def runtimeStatus(self):
        return json.dumps({"state": "ready", "error": None, "embedding": {"ready": True, "error": None}, "database": "Chroma"})

    def embedTexts(self, model, texts, kind):
        texts = json.loads(texts)
        self.embeddings.append((model, texts, kind))
        value = 0.25 if kind == "query" else 0.5
        return json.dumps([[value] * 384 for text in texts])

    def query(self, project, model, vector, limit):
        self.queries.append((project, model, json.loads(vector), limit))
        return json.dumps({"hits": [{"path": "src/tools/BashTool/bashSecurity.ts", "line": 1459, "text": "const allowed = validateCommand(command)", "distance": 0.1}]})

    def addDocuments(self, project, model, items):
        items = json.loads(items)
        self.saved_batches.append(len(items))
        self.saved.extend(items)
        return json.dumps({"saved": len(items)})

    def createProject(self, name):
        return json.dumps({"id": "project1", "name": name})

    def listProjects(self):
        return json.dumps({"projects": [{"id": "project1", "name": "Sources"}]})

    def assetFile(self, path):
        target = Path(self.assets.name) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("asset:" + path)
        return str(target)

    def serverReady(self):
        self.started = True

    def serverStopped(self):
        self.stopped = True


class AtlasRagTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.client = TestClient(atlas_rag.configure(self.runtime))

    def tearDown(self):
        self.client.close()
        self.runtime.assets.cleanup()

    def test_plain_text_rag_runs_langchain_embedding_and_retrieval(self):
        response = self.client.post("/rag", json={"project": "project1", "model": "e5", "query": "Where are commands validated?", "limit": 3})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(self.runtime.embeddings, [("e5", ["Where are commands validated?"], "query")])
        self.assertEqual(self.runtime.queries, [("project1", "e5", [0.25] * 384, 3)])
        self.assertIn("[src/tools/BashTool/bashSecurity.ts:1459]", result["context"])
        self.assertEqual(result["citations"][0]["line"], 1459)
        self.assertEqual(result["citations"][0]["citation"], "src/tools/BashTool/bashSecurity.ts:1459")

    def test_vector_search_does_not_embed_again(self):
        response = self.client.post("/search", json={"project": "project1", "model": "minilm", "vector": [0.75] * 384})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.runtime.embeddings, [])
        self.assertEqual(self.runtime.queries[0][2], [0.75] * 384)
        self.assertEqual(response.json()["hits"][0]["citation"], "src/tools/BashTool/bashSecurity.ts:1459")

    def test_file_chunking_preserves_path_and_absolute_lines(self):
        response = self.client.post("/documents", json={"project": "project1", "documents": [{"path": "src/security.ts", "line": 1457, "text": "a\nb\nc\nd\ne\nf\n"}], "chunk_lines": 3, "overlap_lines": 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["saved"], 3)
        self.assertEqual([item["line"] for item in self.runtime.saved], [1457, 1459, 1461])
        self.assertEqual([item["text"] for item in self.runtime.saved], ["a\nb\nc\n", "c\nd\ne\n", "e\nf\n"])
        self.assertTrue(all(item["path"] == "src/security.ts" for item in self.runtime.saved))
        self.assertTrue(all(item["vector"] == [0.5] * 384 for item in self.runtime.saved))
        self.assertEqual(self.runtime.embeddings[0][2], "passage")

    def test_precomputed_passage_remains_one_passage(self):
        response = self.client.post("/documents", json={"project": "project1", "model": "minilm", "items": [{"path": "src/x.ts", "text": "a\nb\nc", "line": 55, "vector": [1.0] * 384}], "chunk_lines": 1, "overlap_lines": 0})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.runtime.embeddings, [])
        self.assertEqual(len(self.runtime.saved), 1)
        self.assertEqual(self.runtime.saved[0]["line"], 55)

    def test_precomputed_vectors_allow_empty_or_missing_text(self):
        documents = [{"path": "src/empty.ts", "text": "", "vector": [0.1] * 384}, {"path": "src/missing.ts", "vector": [0.2] * 384}]
        response = self.client.post("/documents", json={"project": "project1", "documents": documents})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["saved"], 2)
        self.assertEqual([item["text"] for item in self.runtime.saved], ["", ""])
        self.assertEqual(self.runtime.embeddings, [])

    def test_missing_or_blank_document_text_requires_precomputed_vector(self):
        for document in [{"path": "a.ts"}, {"path": "a.ts", "text": "   "}]:
            response = self.client.post("/documents", json={"project": "project1", "documents": [document]})
            self.assertEqual(response.status_code, 422)
        self.assertEqual(self.runtime.saved, [])
        self.assertEqual(self.runtime.embeddings, [])

    def test_query_validation_rejects_ambiguous_and_invalid_requests(self):
        invalid = [{}, {"query": "x", "vector": [0.0] * 384}, {"vector": [0.0] * 383}, {"query": "   "}, {"query": "x", "model": "unknown"}, {"query": "x", "limit": 31}]
        for payload in invalid:
            with self.subTest(payload=payload):
                response = self.client.post("/search", json={"project": "project1", **payload})
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.runtime.embeddings, [])
        self.assertEqual(self.runtime.queries, [])

    def test_document_validation_rejects_invalid_lines_and_overlap(self):
        for payload in [{"line": 0}, {"vector": [0.0] * 3}]:
            response = self.client.post("/documents", json={"project": "project1", "documents": [{"path": "a.ts", "text": "x", **payload}]})
            self.assertEqual(response.status_code, 422)
        response = self.client.post("/documents", json={"project": "project1", "documents": [{"path": "a.ts", "text": "x"}], "chunk_lines": 3, "overlap_lines": 3})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.runtime.saved, [])

    def test_embedding_endpoint_batch_and_input_type(self):
        response = self.client.post("/v1/embeddings", json={"model": "minilm", "input": ["where?", "how?"], "input_type": "query"})
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual([item["index"] for item in data], [0, 1])
        self.assertEqual(len(data[0]["embedding"]), 384)
        self.assertEqual(self.runtime.embeddings, [("minilm", ["where?", "how?"], "query")])

    def test_lcel_invoke_envelope(self):
        response = self.client.post("/rag/invoke", json={"input": {"project": "project1", "query": "command checks"}})
        self.assertEqual(response.status_code, 200)
        self.assertIn("context", response.json()["output"])

    def test_openapi_and_offline_docs(self):
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/v1/embeddings", response.json()["paths"])
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/docs-assets/swagger-ui-bundle.js", response.text)
        self.assertNotIn("https://cdn", response.text)
        response = self.client.get("/docs-assets/swagger-ui.css")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "asset:vendor/swagger-ui.css")
        self.assertEqual(self.client.get("/docs-assets/secret").status_code, 404)

    def test_bridge_errors_reach_client(self):
        self.runtime.query = lambda *args: json.dumps({"error": "Project not found"})
        response = self.client.post("/search", json={"project": "missing", "query": "x"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Project not found")

    def test_runtime_lifecycle_callbacks_and_interface(self):
        with self.client:
            self.assertTrue(self.runtime.started)
            self.assertEqual(self.client.get("/").text, "asset:index.html")
            self.assertEqual(self.client.get("/app.js").text, "asset:app.js")
        self.assertTrue(self.runtime.stopped)

    def test_embedding_runtime_failure_is_reported(self):
        self.runtime.embedTexts = lambda *args: json.dumps({"error": "Model initialization failed"})
        response = self.client.post("/search", json={"project": "project1", "query": "x"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Model initialization failed")
        self.assertEqual(self.runtime.queries, [])

    def test_health_accepts_null_error_and_reports_real_framework_versions(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["error"])
        self.assertTrue(response.json()["embedding"]["ready"])
        frameworks = response.json()["frameworks"]
        self.assertEqual(frameworks["fastapi"], atlas_rag.fastapi.__version__)
        self.assertEqual(frameworks["langchain_core"], atlas_rag.langchain_core.__version__)
        self.assertEqual(frameworks["uvicorn"], atlas_rag.uvicorn.__version__)
        self.assertEqual(frameworks["python"], atlas_rag.platform.python_version())

    def test_models_describe_bundled_multilingual_e5(self):
        models = self.client.get("/models").json()["models"]
        self.assertEqual(models[0], {"id": "e5", "name": "intfloat/multilingual-e5-small", "dimensions": 384})

    def test_document_batches_stay_within_runtime_limit(self):
        documents = [{"path": f"src/{index}.ts", "text": str(index), "vector": [0.1] * 384} for index in range(35)]
        response = self.client.post("/documents", json={"project": "project1", "documents": documents})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["saved"], 35)
        self.assertEqual(self.runtime.saved_batches, [8, 8, 8, 8, 3])
        self.assertEqual(self.runtime.embeddings, [])


if __name__ == "__main__":
    unittest.main()

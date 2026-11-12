import json
import math
import platform
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import fastapi
import langchain_core
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict, Field, model_validator
from atlas_settings import WebSocketController, WebSocketSettings


ModelName = Literal["e5", "minilm"]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Vector = Annotated[list[FiniteFloat], Field(min_length=384, max_length=384)]
Nonempty = Annotated[str, Field(min_length=1, pattern=r"\S")]
_server = None


def bridge(runtime, method, *args):
    try:
        result = json.loads(str(getattr(runtime, method)(*args)))
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(500, detail=f"{method}: {error}") from error
    if isinstance(result, dict) and result.get("error"):
        status = 500 if method in {"embedTexts", "runtimeStatus"} else 400
        raise HTTPException(status, detail=str(result["error"]))
    return result


class AtlasEmbeddings(Embeddings):
    def __init__(self, runtime, model):
        self.runtime = runtime
        self.model = model

    def _embed(self, texts, kind):
        vectors = bridge(self.runtime, "embedTexts", self.model, json.dumps(texts), kind)
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise HTTPException(500, detail="Embedding runtime returned an invalid batch")
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != 384:
                raise HTTPException(500, detail="Embedding runtime must return 384 dimensions")
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
                raise HTTPException(500, detail="Embedding runtime returned invalid values")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]


class AtlasRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    runtime: Any
    embeddings: Any
    project: str
    model: ModelName
    limit: int | None = None

    def retrieve_vector(self, vector):
        result = bridge(self.runtime, "query", self.project, self.model, json.dumps(vector), self.limit)
        documents = []
        for hit in result["hits"]:
            metadata = {key: value for key, value in hit.items() if key != "text"}
            metadata["citation"] = f"{metadata['path']}:{metadata['line']}"
            documents.append(Document(page_content=hit["text"], metadata=metadata))
        return documents

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        return self.retrieve_vector(self.embeddings.embed_query(query))


class ProjectRequest(BaseModel):
    name: Nonempty


class SourceDocument(BaseModel):
    path: Nonempty
    text: str = ""
    line: int = Field(default=1, ge=1)
    vector: Vector | None = None

    @model_validator(mode="after")
    def text_or_vector(self):
        if self.vector is None and not self.text.strip():
            raise ValueError("Text is required when vector is not provided")
        return self


class DocumentsRequest(BaseModel):
    project: Nonempty
    model: ModelName = "e5"
    documents: list[SourceDocument] = Field(min_length=1)
    chunk_lines: int = Field(default=80, ge=1, le=200)
    overlap_lines: int = Field(default=12, ge=0, le=199)

    @model_validator(mode="before")
    @classmethod
    def accept_items(cls, value):
        if isinstance(value, dict) and "items" in value and "documents" not in value:
            value = {**value, "documents": value["items"]}
        return value

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.overlap_lines >= self.chunk_lines:
            raise ValueError("overlap_lines must be smaller than chunk_lines")
        return self


class SearchRequest(BaseModel):
    project: Nonempty
    model: ModelName = "e5"
    query: Nonempty | None = None
    vector: Vector | None = None
    limit: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def exactly_one_input(self):
        if (self.query is None) == (self.vector is None):
            raise ValueError("Provide exactly one of query or vector")
        return self


class InvokeRequest(BaseModel):
    input: SearchRequest


class EmbeddingRequest(BaseModel):
    model: ModelName = "e5"
    input: Nonempty | Annotated[list[Nonempty], Field(min_length=1)]
    input_type: Literal["query", "passage"] = "passage"
    encoding_format: Literal["float"] = "float"
    dimensions: Literal[384] = 384


def chunk_document(document, chunk_lines, overlap_lines):
    if document.vector is not None:
        return [document.model_dump()]
    lines = document.text.splitlines(keepends=True)
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + chunk_lines, len(lines))
        text = "".join(lines[start:end])
        if text.strip():
            chunks.append({"path": document.path, "line": document.line + start, "text": text})
        if end == len(lines):
            break
        start = end - overlap_lines
    return chunks


def format_documents(documents):
    return [{**document.metadata, "text": document.page_content} for document in documents]


def rag_context(documents):
    citations = format_documents(documents)
    context = "\n\n".join(f"[{hit['citation']}]\n{hit['text']}" for hit in citations)
    return {"context": context, "citations": citations}


def retrieval(runtime, request, formatter):
    embeddings = AtlasEmbeddings(runtime, request.model)
    retriever = AtlasRetriever(runtime=runtime, embeddings=embeddings, project=request.project, model=request.model, limit=request.limit)
    if request.query is not None:
        pipeline = retriever | RunnableLambda(formatter)
        return pipeline.invoke(request.query)
    pipeline = RunnableLambda(retriever.retrieve_vector) | RunnableLambda(formatter)
    return pipeline.invoke(request.vector)


def configure(runtime):
    websocket = WebSocketController(runtime, bridge)

    @asynccontextmanager
    async def lifespan(app):
        await websocket.start()
        if hasattr(runtime, "serverReady"):
            runtime.serverReady()
        try:
            yield
        finally:
            await websocket.close()
            if hasattr(runtime, "serverStopped"):
                runtime.serverStopped()

    app = FastAPI(title="Source Atlas RAG", version="1.0.0", docs_url=None, redoc_url=None, lifespan=lifespan, description="On-device LangChain retrieval with E5, MiniLM and persistent Chroma. Returns source context and citations; no generation model is configured.")
    app.state.websocket = websocket

    @app.get("/settings/websocket", tags=["Settings"])
    async def read_websocket_settings():
        return await websocket.read()

    @app.put("/settings/websocket", tags=["Settings"])
    async def update_websocket_settings(settings: WebSocketSettings):
        return await websocket.update(settings)

    def asset_response(path):
        if any(part in {"", ".", ".."} for part in path.split("/")) or "\\" in path:
            raise HTTPException(404, detail="Asset not found")
        try:
            return FileResponse(str(runtime.assetFile(path)))
        except Exception as error:
            raise HTTPException(404, detail="Asset not found") from error

    @app.get("/", include_in_schema=False)
    def interface():
        return asset_response("index.html")

    @app.get("/app.js", include_in_schema=False)
    def interface_script():
        return asset_response("app.js")

    @app.get("/app.css", include_in_schema=False)
    def interface_style():
        return asset_response("app.css")

    @app.get("/settings.js", include_in_schema=False)
    def settings_script():
        return asset_response("settings.js")

    @app.get("/vendor/{path:path}", include_in_schema=False)
    def vendor_asset(path: str):
        return asset_response("vendor/" + path)

    @app.get("/models/{path:path}", include_in_schema=False)
    def model_asset(path: str):
        return asset_response("models/" + path)

    @app.get("/health", tags=["Runtime"])
    def health():
        status = bridge(runtime, "runtimeStatus")
        return {**status, "frameworks": {"fastapi": fastapi.__version__, "langchain_core": langchain_core.__version__, "uvicorn": uvicorn.__version__, "python": platform.python_version()}}

    @app.get("/models", tags=["Embeddings"])
    def models():
        return {"models": [{"id": "e5", "name": "intfloat/multilingual-e5-small", "dimensions": 384}, {"id": "minilm", "name": "sentence-transformers/all-MiniLM-L6-v2", "dimensions": 384}]}

    @app.get("/projects", tags=["Projects"])
    def projects():
        return bridge(runtime, "listProjects")

    @app.post("/projects", tags=["Projects"], status_code=201)
    def create_project(request: ProjectRequest):
        return bridge(runtime, "createProject", request.name)

    @app.post("/documents", tags=["Documents"])
    def add_documents(request: DocumentsRequest):
        chunks = [chunk for document in request.documents for chunk in chunk_document(document, request.chunk_lines, request.overlap_lines)]
        embeddings = AtlasEmbeddings(runtime, request.model)
        pending = [chunk for chunk in chunks if chunk.get("vector") is None]
        if pending:
            vectors = embeddings.embed_documents([chunk["text"] for chunk in pending])
            for chunk, vector in zip(pending, vectors, strict=True):
                chunk["vector"] = vector
        if not chunks:
            return {"saved": 0, "files": len(request.documents), "model": request.model}
        result = bridge(runtime, "addDocuments", request.project, request.model, json.dumps(chunks))
        return {"saved": result["saved"], "files": len(request.documents), "model": request.model}

    @app.post("/search", tags=["Retrieval"])
    def search(request: SearchRequest):
        return {"hits": retrieval(runtime, request, format_documents)}

    @app.post("/rag", tags=["Retrieval"])
    def rag(request: SearchRequest):
        return retrieval(runtime, request, rag_context)

    @app.post("/rag/invoke", tags=["Retrieval"])
    def invoke(request: InvokeRequest):
        return {"output": retrieval(runtime, request.input, rag_context)}

    @app.post("/v1/embeddings", tags=["Embeddings"], description="OpenAI-shaped float embedding response. Accepts text or batches of text, model aliases e5/minilm and query/passage input_type. Token arrays, token usage accounting, and base64 responses are not implemented.")
    def create_embeddings(request: EmbeddingRequest):
        texts = [request.input] if isinstance(request.input, str) else request.input
        embeddings = AtlasEmbeddings(runtime, request.model)
        vectors = embeddings._embed(texts, request.input_type)
        return {"object": "list", "model": request.model, "data": [{"object": "embedding", "index": index, "embedding": vector} for index, vector in enumerate(vectors)]}

    @app.get("/corpus/files", tags=["Documents"])
    def corpus_files():
        return bridge(runtime, "listCorpusFiles")

    @app.get("/corpus/file/{file_id}", tags=["Documents"])
    def corpus_file(file_id: int):
        if file_id < 0:
            raise HTTPException(422, detail="file_id must be non-negative")
        return bridge(runtime, "readCorpusFile", str(file_id))

    @app.get("/folder/files", tags=["Documents"])
    def folder_files():
        return bridge(runtime, "listFolderFiles")

    @app.get("/folder/file/{file_id}", tags=["Documents"])
    def folder_file(file_id: str):
        return bridge(runtime, "readFolderFile", file_id)

    @app.get("/docs", include_in_schema=False)
    def docs():
        return get_swagger_ui_html(openapi_url="/openapi.json", title="Source Atlas API", swagger_js_url="/docs-assets/swagger-ui-bundle.js", swagger_css_url="/docs-assets/swagger-ui.css", swagger_favicon_url="data:,", swagger_ui_parameters={"validatorUrl": None, "tryItOutEnabled": True})

    @app.get("/docs-assets/{name}", include_in_schema=False)
    def docs_asset(name: str):
        if name not in {"swagger-ui-bundle.js", "swagger-ui.css"}:
            raise HTTPException(404, detail="Asset not found")
        return asset_response("vendor/" + name)

    return app


def run(runtime, host="0.0.0.0", port=8765):
    global _server
    app = configure(runtime)
    config = uvicorn.Config(app, host=host, port=port, loop="asyncio", http="h11", lifespan="on", access_log=False, **app.state.websocket.effective)
    _server = uvicorn.Server(config)
    app.state.websocket.server = _server
    _server.run()


def stop():
    if _server is not None:
        _server.should_exit = True

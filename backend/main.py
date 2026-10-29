import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import ALLOWED_ORIGINS
from repository.manager import (
    MAX_UPLOAD_SIZE,
    save_uploaded_repository,
    get_supported_files,
    is_supported_file,
    resolve_repository_path,
)

from repository.chunker import chunk_repository

from rag.embedding_service import EmbeddingService
from rag.vector_store import VectorStore
from rag.keyword_store import KeywordStore
from rag.search_service import SearchService
from rag.llm_service import (
    LLMConfigurationError,
    LLMProviderError,
    LLMService,
)
from rag.rag_service import RAGService
from rag.reranker import Reranker

from analysis.overview_analyzer import RepositoryOverviewAnalyzer
from analysis.tech_stack_detector import TechnologyStackDetector
from analysis.importance_analyzer import ImportantFileAnalyzer
from analysis.structure_analyzer import RepositoryStructureAnalyzer
from analysis.api_analyzer import APIAnalyzer
from analysis.architecture_analyzer import ArchitectureAnalyzer
from analysis.dependency_service import DependencyService
from analysis.impact_analyzer import ImpactAnalyzer
from analysis.onboarding_analyzer import OnboardingAnalyzer

from services.progress_service import progress_service

logger = logging.getLogger(__name__)
MAX_SOURCE_PREVIEW_LINES = 300
MAX_SOURCE_PREVIEW_START_LINE = 1_000_000


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="CodeLens AI",
    description="Generative AI Codebase Q&A Assistant",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SERVICE INITIALIZATION
# ============================================================

print("Initializing CodeLens AI services...")

embedding_service = EmbeddingService()

vector_store = VectorStore()

# Persistent keyword/BM25 search index.
# This complements ChromaDB semantic search.
keyword_store = KeywordStore()
reranker = Reranker()

search_service = SearchService(
    embedding_service,
    vector_store,
    keyword_store,
    reranker
)
dependency_service = DependencyService()
llm_service = LLMService()

rag_service = RAGService(
    search_service,
    llm_service,
    dependency_service
)

print("CodeLens AI services initialized successfully.")


def _repository_path(repository_id: str):
    try:
        return resolve_repository_path(repository_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid repository ID.",
        ) from exc


# ============================================================
# REQUEST MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "message": "CodeLens AI backend is running.",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "CodeLens AI",
    }


# ============================================================
# UPLOAD REPOSITORY
# ============================================================

@app.post("/api/repositories/upload")
async def upload_repository(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Upload a ZIP repository.

    This endpoint:
    1. Receives the ZIP file.
    2. Extracts the repository.
    3. Creates a unique repository ID.
    4. Returns repository information.

    Indexing is performed separately by /index.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No repository file was provided.",
        )

    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Only ZIP repository files are supported.",
        )

    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit():
        if int(content_length) > MAX_UPLOAD_SIZE + 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Uploaded archive exceeds the maximum upload size.",
            )

    try:
        repository = await save_uploaded_repository(file)

        return {
            "message": "Repository uploaded successfully.",
            "repository_id": repository["repository_id"],
            "repository": repository,
        }

    except ValueError as exc:
        detail = str(exc)
        if "unsafe path" in detail.lower():
            detail = "ZIP archive contains an unsafe path."
        raise HTTPException(
            status_code=400,
            detail=detail,
        ) from exc

    except Exception as exc:
        logger.exception("Repository upload failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to upload and extract the repository.",
        ) from exc


# ============================================================
# REPOSITORY OVERVIEW
# ============================================================

@app.get(
    "/api/repositories/{repository_id}/overview"
)
def get_repository_overview(
    repository_id: str
):
    """
    Analyze the uploaded repository and return
    deterministic repository-level information.
    """

    repository_path = _repository_path(repository_id)

    if not repository_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Repository not found.",
        )

    if not repository_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail="Repository path is invalid.",
        )

    try:
        analyzer = RepositoryOverviewAnalyzer(
            repository_path
        )

        overview = analyzer.analyze()

        return {
            "repository_id": repository_id,
            "overview": overview,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        logger.exception("Repository overview failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to analyze repository.",
        ) from exc


# ============================================================
# TECHNOLOGY STACK
# ============================================================

@app.get(
    "/api/repositories/{repository_id}/technology-stack"
)
def get_technology_stack(
    repository_id: str
):
    """
    Detect the technologies used by the repository.

    Detection is primarily deterministic and based on:
    - file extensions
    - dependency files
    - imports
    - project configuration
    """

    repository_path = _repository_path(repository_id)

    if not repository_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Repository not found.",
        )

    if not repository_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail="Repository path is invalid.",
        )

    try:
        detector = TechnologyStackDetector(
            repository_path
        )

        technology_stack = detector.detect()

        return {
            "repository_id": repository_id,
            "technology_stack": technology_stack,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Technology stack could not be detected.",
        ) from exc

    except Exception as exc:
        logger.exception("Technology stack detection failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to detect technology stack.",
        ) from exc


# ============================================================
# IMPORTANT FILES
# ============================================================

@app.get(
    "/api/repositories/{repository_id}/important-files"
)
def get_important_files(
    repository_id: str,
    top_k: int = 20,
):
    """
    Identify files that are most important for
    understanding the repository.
    """

    repository_path = _repository_path(repository_id)

    if not repository_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Repository not found.",
        )

    if not repository_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail="Repository path is invalid.",
        )

    if top_k < 1:
        raise HTTPException(
            status_code=400,
            detail="top_k must be at least 1.",
        )

    try:
        analyzer = ImportantFileAnalyzer(
            repository_path
        )

        important_files = analyzer.analyze(
            top_k=top_k
        )

        return {
            "repository_id": repository_id,
            "total_files_analyzed": len(
                analyzer.get_files()
            ),
            "important_files": important_files,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Important files could not be analyzed.",
        ) from exc

    except Exception as exc:
        logger.exception("Important file analysis failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to analyze important files.",
        ) from exc


# ============================================================
# REPOSITORY STRUCTURE
# ============================================================

@app.get(
    "/api/repositories/{repository_id}/structure"
)
def get_repository_structure(
    repository_id: str,
):
    """
    Analyze the structural organization of a repository.
    """

    repository_path = _repository_path(repository_id)

    if not repository_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Repository not found.",
        )

    if not repository_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail="Repository path is invalid.",
        )

    try:
        analyzer = RepositoryStructureAnalyzer(
            repository_path
        )

        result = analyzer.analyze()

        return {
            "repository_id": repository_id,
            "structure": result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Repository structure could not be analyzed.",
        ) from exc

    except Exception as exc:
        logger.exception("Repository structure analysis failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to analyze repository structure.",
        ) from exc


# ============================================================
# API DISCOVERY
# ============================================================

@app.get(
    "/api/repositories/{repository_id}/apis"
)
def get_repository_apis(
    repository_id: str,
):
    """
    Discover API endpoints exposed by a repository.
    """

    repository_path = _repository_path(repository_id)

    if not repository_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Repository not found.",
        )

    if not repository_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail="Repository path is invalid.",
        )

    try:
        analyzer = APIAnalyzer(
            repository_path
        )

        result = analyzer.analyze()

        return {
            "repository_id": repository_id,
            "api_discovery": result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Repository APIs could not be discovered.",
        ) from exc

    except Exception as exc:
        logger.exception("API discovery failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to discover repository APIs.",
        ) from exc


# ============================================================
# ARCHITECTURE
# ============================================================

@app.get(
    "/api/repositories/{repository_id}/architecture"
)
def get_repository_architecture(
    repository_id: str
):
    """
    Analyze the static architecture of a repository.
    """

    try:
        _repository_path(repository_id)

        analyzer = ArchitectureAnalyzer(repository_id)

        result = analyzer.analyze()

        return {
            "repository_id": repository_id,
            "architecture": result,
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Repository not found.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Repository architecture could not be analyzed.",
        ) from exc

    except Exception as exc:
        logger.exception("Architecture analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Architecture analysis failed.",
        ) from exc


# ============================================================
# DEPENDENCIES
# ============================================================

@app.get(
    "/api/repositories/{repository_id}/dependencies"
)
def analyze_dependencies(
    repository_id: str
):
    repository_path = _repository_path(repository_id)

    if not repository_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Repository not found.",
        )

    if not repository_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail="Repository path is invalid.",
        )

    try:
        _repository_path(repository_id)
        _repository_path(repository_id)

        result = dependency_service.analyze_repository(
            repository_id=repository_id,
            repository_path=repository_path,
        )

        return {
            "repository_id": repository_id,
            "dependency_analysis": result,
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Repository not found.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Repository dependencies could not be analyzed.",
        ) from exc

    except Exception as exc:
        logger.exception("Dependency analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Dependency analysis failed.",
        ) from exc
# ============================================================
# FILE IMPACT
# ============================================================

@app.get(
    "/api/repositories/{repository_id}/impact/{file_path:path}"
)
def analyze_file_impact(
    repository_id: str,
    file_path: str,
):
    _repository_path(repository_id)
    analyzer = ImpactAnalyzer(repository_id)

    try:
        return {
            "repository_id": repository_id,
            "impact_analysis": analyzer.analyze(
                file_path
            ),
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Requested file was not found in the repository.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid file path for impact analysis.",
        ) from exc

    except Exception as exc:
        logger.exception("File impact analysis failed")
        raise HTTPException(
            status_code=500,
            detail="File impact analysis failed.",
        ) from exc


# ============================================================
# REPOSITORY IMPACT
# ============================================================

@app.get(
    "/api/repositories/{repository_id}/impact"
)
def analyze_repository_impact(
    repository_id: str,
):
    _repository_path(repository_id)
    analyzer = ImpactAnalyzer(repository_id)

    try:
        return {
            "repository_id": repository_id,
            "impact_analysis": analyzer.analyze(),
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Repository not found.",
        ) from exc

    except Exception as exc:
        logger.exception("Repository impact analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Repository impact analysis failed.",
        ) from exc


# ============================================================
# ONBOARDING
# ============================================================

@app.get(
    "/api/repositories/{repository_id}/onboarding"
)
def analyze_onboarding(
    repository_id: str,
):
    _repository_path(repository_id)
    analyzer = OnboardingAnalyzer(repository_id)

    try:
        return {
            "repository_id": repository_id,
            "onboarding": analyzer.analyze(),
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Repository not found.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Repository onboarding analysis could not be generated.",
        ) from exc

    except Exception as exc:
        logger.exception("Onboarding analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Onboarding analysis failed.",
        ) from exc


# ============================================================
# INDEX REPOSITORY
# ============================================================

@app.post(
    "/api/repositories/{repository_id}/index"
)
def index_repository(
    repository_id: str
):
    """
    Index an uploaded repository into both retrieval systems.

    Indexing pipeline:

        scanning
            ↓
        chunking
            ↓
        embedding
            ↓
        ChromaDB + SQLite FTS5
            ↓
        completed

    ChromaDB:
        Semantic retrieval

    SQLite FTS5:
        Keyword/BM25 retrieval

    Progress is tracked using progress_service.
    """

    repository_path = _repository_path(repository_id)

    if not progress_service.begin(repository_id):
        raise HTTPException(
            status_code=409,
            detail="Repository indexing is already in progress.",
        )

    indexes_replaced = False

    try:

        # ====================================================
        # REPOSITORY PATH
        # ====================================================

        # ----------------------------------------------------
        # VALIDATE REPOSITORY
        # ----------------------------------------------------

        if not repository_path.exists():

            progress_service.fail(
                repository_id,
                "Repository not found."
            )

            raise HTTPException(
                status_code=404,
                detail="Repository not found."
            )

        if not repository_path.is_dir():

            progress_service.fail(
                repository_id,
                "Repository path is not a directory."
            )

            raise HTTPException(
                status_code=400,
                detail=(
                    "Repository path is not a directory."
                )
            )

        # ====================================================
        # STEP 1: SCANNING
        # ====================================================

        progress_service.update(
            repository_id,
            status="indexing",
            stage="scanning",
            progress=10,
            processed_files=0,
            total_files=0,
            processed_chunks=0,
            total_chunks=0,
            message="Scanning repository files..."
        )

        files = get_supported_files(
            repository_path
        )

        total_files = len(files)

        progress_service.update(
            repository_id,
            stage="scanning",
            progress=20,
            processed_files=0,
            total_files=total_files,
            processed_chunks=0,
            total_chunks=0,
            message=(
                f"Found {total_files} "
                "supported files."
            )
        )

        # ====================================================
        # STEP 2: CHUNKING
        # ====================================================

        progress_service.update(
            repository_id,
            stage="chunking",
            progress=30,
            processed_files=0,
            total_files=total_files,
            processed_chunks=0,
            total_chunks=0,
            message="Creating code-aware chunks..."
        )

        all_chunks = chunk_repository(
            repository_path=repository_path,
            repository_id=repository_id,
            files=files
        )

        total_chunks = len(all_chunks)

        progress_service.update(
            repository_id,
            stage="chunking",
            progress=50,
            processed_files=total_files,
            total_files=total_files,
            processed_chunks=total_chunks,
            total_chunks=total_chunks,
            message=(
                f"Created {total_chunks} "
                "chunks."
            )
        )

        # ----------------------------------------------------
        # CHECK FOR EMPTY REPOSITORY
        # ----------------------------------------------------

        if total_chunks == 0:

            progress_service.fail(
                repository_id,
                "No supported content was found to index."
            )

            raise HTTPException(
                status_code=400,
                detail=(
                    "No supported content was "
                    "found to index."
                )
            )

        # ====================================================
        # STEP 3: EMBEDDINGS
        # ====================================================

        progress_service.update(
            repository_id,
            stage="embedding",
            progress=55,
            processed_files=total_files,
            total_files=total_files,
            processed_chunks=0,
            total_chunks=total_chunks,
            message="Generating embeddings..."
        )

        documents = [
            chunk["content"]
            for chunk in all_chunks
        ]

        embeddings = (
            embedding_service.embed_documents(
                documents
            )
        )

        if len(embeddings) != total_chunks:

            raise ValueError(
                "Embedding count does not match "
                "chunk count."
            )

        progress_service.update(
            repository_id,
            stage="embedding",
            progress=75,
            processed_files=total_files,
            total_files=total_files,
            processed_chunks=total_chunks,
            total_chunks=total_chunks,
            message=(
                "Embeddings generated successfully."
            )
        )

        # ====================================================
        # STEP 4: RETRIEVAL INDEXES
        # ====================================================

        progress_service.update(
            repository_id,
            stage="indexing",
            progress=80,
            processed_files=total_files,
            total_files=total_files,
            processed_chunks=total_chunks,
            total_chunks=total_chunks,
            message=(
                "Building semantic and keyword indexes..."
            )
        )

        progress_service.update(
            repository_id,
            stage="dependency_analysis",
            progress=85,
            processed_files=total_files,
            total_files=total_files,
            processed_chunks=total_chunks,
            total_chunks=total_chunks,
            message="Analyzing repository dependencies...",
        )
        dependency_service.clear_repository(repository_id)
        dependency_analysis = dependency_service.analyze_repository(
            repository_id=repository_id,
            repository_path=repository_path,
        )

        # ----------------------------------------------------
        # Remove previous ChromaDB index.
        #
        # This prevents duplicate chunks when the same
        # repository is indexed again.
        # ----------------------------------------------------

        indexes_replaced = True
        vector_store.delete_repository(
            repository_id
        )

        # ----------------------------------------------------
        # Remove previous SQLite FTS5 keyword index.
        #
        # This keeps the keyword index synchronized with
        # the latest repository contents.
        # ----------------------------------------------------

        keyword_store.delete_repository(
            repository_id
        )

        # ----------------------------------------------------
        # Store semantic vectors in ChromaDB.
        # ----------------------------------------------------

        vector_count = vector_store.add_chunks(
            repository_id,
            all_chunks,
            embeddings
        )

        if vector_count != total_chunks:

            raise ValueError(
                "ChromaDB chunk count does not match "
                "repository chunk count."
            )

        progress_service.update(
            repository_id,
            stage="dependency_analysis",
            progress=90,
            processed_files=total_files,
            total_files=total_files,
            processed_chunks=total_chunks,
            total_chunks=total_chunks,
            message="Dependency analysis completed.",
        )
        # ----------------------------------------------------
        # Store the same chunks in SQLite FTS5.
        #
        # FTS5 will provide keyword/BM25 retrieval.
        # ----------------------------------------------------

        keyword_count = keyword_store.add_chunks(
            repository_id,
            all_chunks
        )

        if keyword_count != total_chunks:

            raise ValueError(
                "SQLite FTS5 chunk count does not match "
                "repository chunk count."
            )

        progress_service.update(
            repository_id,
            stage="indexing",
            progress=95,
            processed_files=total_files,
            total_files=total_files,
            processed_chunks=total_chunks,
            total_chunks=total_chunks,
            message=(
                "Semantic and keyword indexes "
                "created successfully."
            )
        )

        # ====================================================
        # STEP 6: COMPLETED
        # ====================================================

        progress_service.complete(
            repository_id,
            processed_files=total_files,
            total_files=total_files,
            processed_chunks=total_chunks,
            total_chunks=total_chunks
        )

        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        return {
            "repository_id": repository_id,
            "status": "completed",
            "message": (
                "Repository indexed successfully."
            ),
            "total_files": total_files,
            "total_chunks": total_chunks,
            "semantic_index": {
                "store": "ChromaDB",
                "chunks": vector_count,
            },
            "keyword_index": {
                "store": "SQLite FTS5",
                "chunks": keyword_count,
            },
            "dependency_analysis": {
    "files": dependency_analysis.get("total_files", 0),
    "entities": dependency_analysis.get("total_entities", 0),
    "dependencies": len(
        dependency_analysis.get(
            "entity_dependencies",
            []
        )
    ),
},
        }

    # ========================================================
    # HTTP EXCEPTIONS
    # ========================================================

    except HTTPException:
        raise

    # ========================================================
    # UNEXPECTED ERRORS
    # ========================================================

    except Exception as exc:
        if indexes_replaced:
            try:
                vector_store.delete_repository(repository_id)
                keyword_store.delete_repository(repository_id)
            except Exception:
                logger.exception(
                    "Failed to clean up incomplete repository indexes"
                )
        progress_service.fail(
            repository_id,
            "Indexing failed. Check backend logs for details.",
        )

        logger.exception("Repository indexing failed")

        raise HTTPException(
            status_code=500,
            detail="Repository indexing failed.",
        ) from exc


# ============================================================
# REPOSITORY INDEXING PROGRESS
# ============================================================

@app.get(
    "/api/repositories/{repository_id}/progress"
)
def get_repository_progress(repository_id: str):
    """
    Return the current indexing progress of a repository.
    """

    repository_path = _repository_path(repository_id)
    if not repository_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail="Repository not found.",
        )

    progress = progress_service.get(
        repository_id
    )

    if progress is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No progress information found "
                "for this repository."
            ),
        )

    return progress


# ============================================================
# SEMANTIC SEARCH
# ============================================================

@app.get(
    "/api/repositories/{repository_id}/files/{file_path:path}"
)
def get_repository_file(
    repository_id: str,
    file_path: str,
    start_line: int = 1,
    line_count: int = 200,
):
    """Return a bounded source preview from the selected repository."""
    repository_path = _repository_path(repository_id)
    if not repository_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail="Repository not found.",
        )

    if (
        start_line < 1
        or start_line > MAX_SOURCE_PREVIEW_START_LINE
        or line_count < 1
        or line_count > MAX_SOURCE_PREVIEW_LINES
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid source preview range.",
        )

    relative_path = Path(file_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise HTTPException(
            status_code=400,
            detail="Invalid repository file path.",
        )

    source_path = (repository_path / relative_path).resolve()
    try:
        source_path.relative_to(repository_path.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid repository file path.",
        ) from exc

    if not source_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Repository file not found.",
        )

    if not is_supported_file(source_path):
        raise HTTPException(
            status_code=415,
            detail="This file is not available for source preview.",
        )

    end_line = start_line + line_count - 1
    lines = []
    try:
        with source_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as source:
            for line_number, line in enumerate(source, start=1):
                if line_number < start_line:
                    continue
                if line_number > end_line:
                    break
                lines.append(line.rstrip("\r\n"))
    except OSError as exc:
        logger.exception("Source preview failed")
        raise HTTPException(
            status_code=500,
            detail="Could not read the repository file.",
        ) from exc

    if not lines:
        raise HTTPException(
            status_code=416,
            detail="Source line is outside the file.",
        )

    return {
        "repository_id": repository_id,
        "file_path": relative_path.as_posix(),
        "start_line": start_line,
        "end_line": start_line + len(lines) - 1,
        "content": "\n".join(lines),
    }


@app.post(
    "/api/repositories/{repository_id}/search"
)
def semantic_search(
    repository_id: str,
    request: SearchRequest,
):
    """
    Search the selected repository using hybrid retrieval.
    """

    if not request.query.strip():

        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    if request.top_k < 1:

        raise HTTPException(
            status_code=400,
            detail="top_k must be at least 1.",
        )

    try:
        repository_path = _repository_path(repository_id)
        if not repository_path.is_dir():
            raise HTTPException(
                status_code=404,
                detail="Repository not found.",
            )

        results = search_service.search(
            repository_id=repository_id,
            query=request.query,
            top_k=request.top_k,
        )

        return {
            "repository_id": repository_id,
            "query": request.query,
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Semantic search failed")

        raise HTTPException(
            status_code=500,
            detail="Semantic search failed.",
        ) from exc


# ============================================================
# ASK CODELENS
# ============================================================

@app.post(
    "/api/repositories/{repository_id}/ask"
)
def ask_repository(
    repository_id: str,
    request: AskRequest,
):
    """
    Complete RAG pipeline:

    Question
        ↓
    Hybrid semantic and keyword retrieval
        ↓
    Reranking
        ↓
    Relevant code chunks
        ↓
    Gemini
        ↓
    Grounded answer
    """

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if request.top_k < 1:

        raise HTTPException(
            status_code=400,
            detail="top_k must be at least 1.",
        )

    try:
        repository_path = _repository_path(repository_id)

        if not repository_path.is_dir():
            raise HTTPException(
                status_code=404,
                detail="Repository not found.",
            )

        result = rag_service.answer_question(
            repository_id=repository_id,
            question=request.question,
            top_k=request.top_k,
            repository_path=repository_path,
        )

        return result

    except HTTPException:
        raise
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=503,
            detail="Answer generation is unavailable. Please try again later.",
        ) from exc
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("RAG request failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to answer the question.",
        ) from exc


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print("")
    print("=" * 60)
    print("CodeLens AI backend started")
    print("API: http://127.0.0.1:8000")
    print("Docs: http://127.0.0.1:8000/docs")
    print("Frontend: http://127.0.0.1:5500")
    print("=" * 60)
    print("")
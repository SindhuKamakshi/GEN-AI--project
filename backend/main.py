from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import REPOSITORIES_DIR
from repository.manager import (save_uploaded_repository, get_supported_files,)
from rag.embedding_service import EmbeddingService
from rag.vector_store import VectorStore
from rag.search_service import SearchService
from rag.llm_service import LLMService
from rag.rag_service import RAGService
from analysis.overview_analyzer import RepositoryOverviewAnalyzer
from analysis.tech_stack_detector import TechnologyStackDetector
from analysis.importance_analyzer import ImportantFileAnalyzer
from analysis.structure_analyzer import RepositoryStructureAnalyzer
from analysis.api_analyzer import APIAnalyzer
from analysis.architecture_analyzer import ArchitectureAnalyzer
from analysis.dependency_analyzer import DependencyAnalyzer
from analysis.impact_analyzer import ImpactAnalyzer
from analysis.onboarding_analyzer import OnboardingAnalyzer
from services.progress_service import progress_service
from repository.chunker import chunk_repository
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
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
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

search_service = SearchService(
    embedding_service,
    vector_store,
)

llm_service = LLMService()

rag_service = RAGService(
    search_service,
    llm_service,
)

print("CodeLens AI services initialized successfully.")


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

    try:
        repository = await save_uploaded_repository(
            file
        )

        return {
            "message": "Repository uploaded successfully.",
            "repository_id": repository["repository_id"],
            "repository": repository,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        print(
            f"Repository upload failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to upload and extract the repository: "
                f"{str(exc)}"
            ),
        )

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

    repository_path = (
        REPOSITORIES_DIR / repository_id
    )

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

        print(
            f"Repository overview failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to analyze repository: "
                f"{str(exc)}"
            ),
        )
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

    repository_path = (
        REPOSITORIES_DIR / repository_id
    )

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
            detail=str(exc),
        )

    except Exception as exc:

        print(
            f"Technology stack detection failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to detect technology stack: "
                f"{str(exc)}"
            ),
        )
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

    repository_path = (
        REPOSITORIES_DIR / repository_id
    )

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
            detail=str(exc),
        )

    except Exception as exc:

        print(
            f"Important file analysis failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to analyze important files: "
                f"{str(exc)}"
            ),
        )
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

    repository_path = (
        REPOSITORIES_DIR / repository_id
    )

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
            detail=str(exc),
        )

    except Exception as exc:

        print(
            f"Repository structure analysis failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to analyze repository structure: "
                f"{str(exc)}"
            ),
        )
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

    repository_path = (
        REPOSITORIES_DIR / repository_id
    )

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
            detail=str(exc),
        )

    except Exception as exc:

        print(
            f"API discovery failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to discover APIs: "
                f"{str(exc)}"
            ),
        )
@app.get("/api/repositories/{repository_id}/architecture")
def get_repository_architecture(repository_id: str):
    """
    Analyze the static architecture of a repository.
    """

    try:
        analyzer = ArchitectureAnalyzer(repository_id)
        result = analyzer.analyze()

        return {
            "repository_id": repository_id,
            "architecture": result,
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Architecture analysis failed: {str(exc)}",
        )
@app.get("/api/repositories/{repository_id}/dependencies")
def analyze_dependencies(repository_id: str):
    analyzer = DependencyAnalyzer(repository_id)

    try:
        result = analyzer.analyze()

        return {
            "repository_id": repository_id,
            "dependency_analysis": result,
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
@app.get(
    "/api/repositories/{repository_id}/impact/{file_path:path}"
)
def analyze_file_impact(
    repository_id: str,
    file_path: str,
):
    analyzer = ImpactAnalyzer(
        repository_id
    )

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
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
@app.get(
    "/api/repositories/{repository_id}/impact"
)
def analyze_repository_impact(
    repository_id: str,
):
    analyzer = ImpactAnalyzer(
        repository_id
    )

    try:
        return {
            "repository_id": repository_id,
            "impact_analysis": analyzer.analyze(),
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
@app.get(
    "/api/repositories/{repository_id}/onboarding"
)
def analyze_onboarding(
    repository_id: str,
):
    analyzer = OnboardingAnalyzer(
        repository_id
    )

    try:
        return {
            "repository_id": repository_id,
            "onboarding": analyzer.analyze(),
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
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
    Index an uploaded repository into the vector database.

    Indexing pipeline:

        scanning
            ↓
        chunking
            ↓
        embedding
            ↓
        indexing
            ↓
        completed

    Progress is tracked using progress_service.
    """

    # --------------------------------------------------------
    # INITIALIZE PROGRESS
    # --------------------------------------------------------

    progress_service.initialize(
        repository_id
    )

    try:

        # ====================================================
        # REPOSITORY PATH
        # ====================================================

        repository_path = (
            REPOSITORIES_DIR / repository_id
        )

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
            message=(
                "Scanning repository files..."
            )
        )

        # Get supported files using the existing
        # repository manager function.
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
            message=(
                "Creating code-aware chunks..."
            )
        )

        # ----------------------------------------------------
        # Use the existing repository-level chunker.
        #
        # This automatically handles:
        #
        # Python AST chunking
        # Normal text chunking
        # Large-file streaming chunking
        # Chunk overlap
        # Deterministic chunk IDs
        # ----------------------------------------------------

        all_chunks = chunk_repository(
            repository_path=repository_path,
            repository_id=repository_id,
            files=files
        )

        total_chunks = len(
            all_chunks
        )

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
            message=(
                "Generating embeddings..."
            )
        )

        # Extract chunk content for embedding.
        documents = [
            chunk["content"]
            for chunk in all_chunks
        ]

        # Generate embeddings using the existing
        # EmbeddingService instance.
        embeddings = (
            embedding_service.embed_documents(
                documents
            )
        )

        # Validate embedding count.
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
        # STEP 4: VECTOR DATABASE
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
                "Storing vectors in ChromaDB..."
            )
        )

        # ----------------------------------------------------
        # Remove previous index for this repository.
        #
        # This prevents duplicate chunks when the same
        # repository is indexed again.
        # ----------------------------------------------------

        vector_store.delete_repository(
            repository_id
        )

        # ----------------------------------------------------
        # Store chunks + embeddings in ChromaDB.
        # ----------------------------------------------------

        vector_store.add_chunks(
            repository_id,
            all_chunks,
            embeddings
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
                "Repository indexed in ChromaDB."
            )
        )

        # ====================================================
        # STEP 5: COMPLETED
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
            "total_chunks": total_chunks
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

        # Update progress state to failed.
        progress_service.fail(
            repository_id,
            str(exc)
        )

        # Print error in terminal for debugging.
        print(
            "Repository indexing failed:",
            str(exc)
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )
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

    progress = progress_service.get(
        repository_id
    )

    if progress is None:
        raise HTTPException(
            status_code=404,
            detail="No progress information found for this repository.",
        )

    return progress
# ============================================================
# SEMANTIC SEARCH
# ============================================================

@app.post(
    "/api/repositories/{repository_id}/search"
)
def semantic_search(
    repository_id: str,
    request: SearchRequest,
):
    """
    Perform semantic search over the selected repository.
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

    except Exception as exc:

        print(
            f"Semantic search failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Semantic search failed: "
                f"{str(exc)}"
            ),
        )


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
    Semantic retrieval
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

        result = rag_service.answer_question(
            repository_id=repository_id,
            question=request.question,
            top_k=request.top_k,
        )

        return result

    except Exception as exc:

        print(
            f"RAG request failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


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
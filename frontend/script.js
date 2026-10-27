const CODELENS_API_BASE = (
    document.querySelector('meta[name="codelens-api-base"]')?.content?.trim() ||
    "http://127.0.0.1:8000"
).replace(/\/+$/, "");
const CURRENT_REPOSITORY_KEY = "codelens_current_repository";
const HISTORY_KEY = "codelens_history";
let selectedRepositoryFile = null;
let currentRepository = null;
let isBusy = false;
let activeConversationId = null;
let repositoryLoadVersion = 0;

async function readApiResponse(response) {
    const text = await response.text();
    if (!text) return {};
    try {
        return JSON.parse(text);
    } catch (_) {
        return { detail: text };
    }
}

function apiError(data, fallback, status = 0) {
    if (status >= 500) {
        if (status === 503) {
            const safeMessages = new Set([
                "Answer generation is unavailable. Please try again later.",
                "Answer generation credentials are invalid or unavailable. Please contact the administrator.",
                "Answer generation is temporarily rate-limited. Please try again later.",
                "Answer generation is temporarily unavailable. Please try again later."
            ]);
            if (safeMessages.has(data?.detail)) return data.detail;
            return "Answer generation is temporarily unavailable. Please try again later.";
        }
        return fallback;
    }

    const detail = typeof data?.detail === "string"
        ? data.detail.trim()
        : "";
    if (detail && detail.length <= 240 && !/traceback|exception|stack trace|fastapi|chromadb|gemini|embedding|database error/i.test(detail)) {
        return detail;
    }
    return fallback;
}

async function apiRequest(url, options = {}, fallback = "The request could not be completed.") {
    let response;
    try {
        response = await fetch(url, options);
    } catch (error) {
        console.error("CodeLens request failed:", error);
        throw new Error("Could not reach CodeLens. Check that the application is running and try again.");
    }

    const data = await readApiResponse(response);
    if (!response.ok) {
        throw new Error(apiError(data, fallback, response.status));
    }
    return data;
}

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatFileSize(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 KB";

    const units = ["B", "KB", "MB", "GB"];
    let size = bytes;
    let index = 0;

    while (size >= 1024 && index < units.length - 1) {
        size /= 1024;
        index += 1;
    }

    return `${size >= 10 || index === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[index]}`;
}

function showToast(message, isError = false) {
    const toast = document.getElementById("toast");
    const toastMessage = document.getElementById("toastMessage");

    if (!toast || !toastMessage) return;

    toastMessage.textContent = message;
    toast.classList.toggle("error", isError);
    toast.classList.add("visible");

    window.clearTimeout(showToast.timer);

    showToast.timer = window.setTimeout(() => {
        toast.classList.remove("visible");
    }, 3500);
}

function refreshIcons() {
    if (
        window.lucide &&
        typeof window.lucide.createIcons === "function"
    ) {
        window.lucide.createIcons();
    }
}

function saveCurrentRepository(repository) {
    currentRepository = repository;

    localStorage.setItem(
        CURRENT_REPOSITORY_KEY,
        JSON.stringify(repository)
    );
}

function loadCurrentRepository() {
    try {
        const value = localStorage.getItem(
            CURRENT_REPOSITORY_KEY
        );

        return value ? JSON.parse(value) : null;
    } catch (_) {
        return null;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    refreshIcons();

    initializeSidebar();
    initializeFolders();
    initializeUploadModal();
    initializeChat();
    initializeSuggestions();
    initializeTextarea();
    initializeHistory();
    initializeDashboard();
    initializeSourcePreview();

    const saved = loadCurrentRepository();

    if (saved) {
        updateRepositoryUI(saved);
    }
});

function initializeSidebar() {
    const sidebar = document.getElementById("sidebar");
    const menuButton = document.getElementById("mobileMenuButton");
    const overlay = document.getElementById("sidebarOverlay");

    if (!sidebar || !menuButton || !overlay) return;

    const close = () => {
        sidebar.classList.remove("open");
        overlay.classList.remove("visible");

        menuButton.setAttribute(
            "aria-expanded",
            "false"
        );
    };

    menuButton.addEventListener("click", () => {
        const opening = !sidebar.classList.contains("open");

        sidebar.classList.toggle("open", opening);
        overlay.classList.toggle("visible", opening);

        menuButton.setAttribute(
            "aria-expanded",
            String(opening)
        );
    });

    overlay.addEventListener("click", close);

    document.querySelectorAll("[data-scroll-target]").forEach(button => {
        button.addEventListener("click", () => {
            document.getElementById(button.dataset.scrollTarget)?.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        });
    });

    document.querySelector('[aria-label="Search"]')?.addEventListener("click", () => {
        document.getElementById("repositorySearchInput")?.focus();
        document.getElementById("repositorySearchForm")?.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    });
}

function initializeFolders() {
    document.querySelectorAll(".tree-folder").forEach(folder => {
        folder.addEventListener("click", () => {
            const folderId = folder.dataset.folder;
            const children = document.getElementById(folderId);

            if (!children) return;

            const isOpen = children.classList.toggle(
                "collapsed"
            );

            folder.classList.toggle(
                "collapsed",
                isOpen
            );
        });
    });
    }

function initializeUploadModal() {
    const modal = document.getElementById("uploadModal");
    const openButton = document.getElementById("newProjectButton");
    const closeButton = document.getElementById("closeModal");
    const fileInput = document.getElementById("repositoryFile");
    const uploadZone = document.getElementById("uploadZone");
    const selectedFile = document.getElementById("selectedFile");
    const selectedName = document.getElementById("selectedFileName");
    const selectedSize = document.getElementById("selectedFileSize");
    const removeButton = document.getElementById("removeSelectedFile");
    const analyzeButton = document.getElementById("analyzeRepository");

    if (!modal || !fileInput || !analyzeButton) {
        return;
    }

    const openModal = () => {
        modal.classList.add("visible");
        modal.setAttribute("aria-hidden", "false");
    };

    const closeModal = () => {
        if (isBusy) return;

        modal.classList.remove("visible");
        modal.setAttribute("aria-hidden", "true");
    };

    const resetSelection = () => {
        selectedRepositoryFile = null;
        fileInput.value = "";

        if (selectedFile) {
            selectedFile.hidden = true;
        }

        analyzeButton.disabled = true;

        analyzeButton.innerHTML =
            '<i data-lucide="scan-search"></i> Analyze repository';

        refreshIcons();
    };

    const chooseFile = file => {
        if (!file) return;

        if (!file.name.toLowerCase().endsWith(".zip")) {
            resetSelection();

            showToast(
                "Please select a ZIP repository.",
                true
            );

            return;
        }

        selectedRepositoryFile = file;

        if (selectedName) {
            selectedName.textContent = file.name;
        }

        if (selectedSize) {
            selectedSize.textContent =
                formatFileSize(file.size);
        }

        if (selectedFile) {
            selectedFile.hidden = false;
        }

        analyzeButton.disabled = false;
        analyzeButton.removeAttribute("disabled");

        refreshIcons();
    };

    if (openButton) {
        openButton.addEventListener(
            "click",
            openModal
        );
    }

    if (closeButton) {
        closeButton.addEventListener(
            "click",
            closeModal
        );
    }

    if (removeButton) {
        removeButton.addEventListener(
            "click",
            resetSelection
        );
    }

    if (uploadZone) {
        uploadZone.addEventListener(
            "click",
            () => fileInput.click()
        );
    }

    fileInput.addEventListener(
        "change",
        event => {
            chooseFile(
                event.target.files?.[0]
            );
        }
    );

    if (uploadZone) {
        uploadZone.addEventListener(
            "dragover",
            event => {
                event.preventDefault();
                uploadZone.classList.add("dragging");
            }
        );

        uploadZone.addEventListener(
            "dragleave",
            () => {
                uploadZone.classList.remove(
                    "dragging"
                );
            }
        );

        uploadZone.addEventListener(
            "drop",
            event => {
                event.preventDefault();

                uploadZone.classList.remove(
                    "dragging"
                );

                chooseFile(
                    event.dataTransfer.files?.[0]
                );
            }
        );
    }

    analyzeButton.addEventListener(
        "click",
        async () => {
            if (!selectedRepositoryFile || isBusy) {
                return;
            }

            await handleRepositoryUpload(
                selectedRepositoryFile,
                analyzeButton,
                closeModal
            );
        }
    );

    modal.addEventListener(
        "click",
        event => {
            if (event.target === modal) {
                closeModal();
            }
        }
    );
}

async function handleRepositoryUpload(
    file,
    button,
    closeModal
) {
    isBusy = true;

    button.disabled = true;

    button.innerHTML =
        '<i data-lucide="loader-circle"></i> Uploading...';

    refreshIcons();

    try {
        const formData = new FormData();

        formData.append(
            "file",
            file
        );

        const uploadData = await apiRequest(
            `${CODELENS_API_BASE}/api/repositories/upload`,
            {
                method: "POST",
                body: formData
            },
            "Repository upload failed."
        );

        const repository =
            uploadData.repository || {};

        const repositoryId =
            uploadData.repository_id ||
            repository.repository_id;

        if (!repositoryId) {
            throw new Error(
                "The backend did not return a repository ID."
            );
        }

        const state = {
            repository_id: repositoryId,
            name: file.name.replace(
                /\.zip$/i,
                ""
            ),
            file_count:
                repository.file_count || 0,
            files:
                repository.files || [],
            file_types:
                repository.file_types || {},
            updated_at: Date.now()
        };

        saveCurrentRepository(state);
        addRepositoryToHistory(state);
        updateRepositoryUI(state);
        setIndexProgress({
            status: "indexing",
            stage: "initialized",
            progress: 0,
            processed_files: 0,
            total_files: 0
        });

        button.innerHTML =
            '<i data-lucide="loader-circle"></i> Indexing...';

        refreshIcons();
        await indexRepositoryWithProgress(repositoryId);
        state.indexed = true;
        saveCurrentRepository(state);
        addRepositoryToHistory(state);

        button.innerHTML =
            '<i data-lucide="check"></i> Repository ready';

        refreshIcons();

        showToast(
            `${state.name} is ready to explore.`
        );

        window.setTimeout(() => {
            closeModal();
        }, 500);

    } catch (error) {
        console.error(
            "Repository upload error:",
            error
        );

        button.disabled = false;

        button.innerHTML =
            '<i data-lucide="scan-search"></i> Analyze repository';

        refreshIcons();

        setIndexProgress({ status: "failed", progress: 0 });
        showToast(error.message || "Could not prepare the repository.", true);

    } finally {
        isBusy = false;
    }
}

async function indexRepositoryWithProgress(repositoryId) {
    let indexResult;
    let indexFinished = false;
    const indexRequest = apiRequest(
        `${CODELENS_API_BASE}/api/repositories/${encodeURIComponent(repositoryId)}/index`,
        { method: "POST" },
        "Repository indexing could not be completed."
    ).then(
        data => {
            indexFinished = true;
            indexResult = { data };
        },
        error => {
            indexFinished = true;
            indexResult = { error };
        }
    );

    const startedAt = Date.now();
    while (Date.now() - startedAt < 10 * 60 * 1000) {
        let progress;
        try {
            progress = await apiRequest(
                `${CODELENS_API_BASE}/api/repositories/${encodeURIComponent(repositoryId)}/progress`,
                {},
                "Could not read repository progress."
            );
        } catch (error) {
            if (!error.message.includes("No progress information")) {
                throw error;
            }
        }

        if (progress) {
            setIndexProgress(progress);
            if (progress.status === "failed") {
                await indexRequest;
                throw new Error("Repository analysis did not finish. Please try again.");
            }
            if (progress.status === "completed") {
                await indexRequest;
                if (indexResult?.error) throw indexResult.error;
                return indexResult?.data;
            }
        }

        if (indexFinished && indexResult?.error) {
            throw indexResult.error;
        }
        if (indexFinished && indexResult?.data) {
            return indexResult.data;
        }

        await new Promise(resolve => window.setTimeout(resolve, 800));
    }

    throw new Error("Repository analysis is taking longer than expected. Check its status and try again.");
}

function setIndexProgress(progress = {}) {
    const percent = Math.max(0, Math.min(100, Number(progress.progress) || 0));
    const progressBar = document.getElementById("indexProgressBar");
    const track = progressBar?.parentElement;
    if (progressBar) progressBar.style.width = `${percent}%`;
    if (track) track.setAttribute("aria-valuenow", String(percent));

    const status = progress.status || "initialized";
    const stateLabel = document.getElementById("analysisState");
    const progressLabel = document.getElementById("overviewIndexProgress");
    const detail = document.getElementById("overviewIndexMessage");
    const statusText = document.getElementById("repositoryStatusText");

    const labels = {
        initialized: "Starting",
        scanning: "Reviewing files",
        chunking: "Preparing repository",
        embedding: "Building search index",
        dependency_analysis: "Mapping dependencies",
        indexing: "Finishing setup",
        completed: "Ready to explore",
        failed: "Analysis incomplete",
        unknown: "Status unavailable"
    };
    const label = labels[progress.stage] || labels[status] || "Preparing repository";

    if (progressLabel) {
        progressLabel.textContent = status === "completed"
            ? "Ready"
            : status === "failed"
                ? "Needs attention"
                : status === "unknown"
                    ? "Unknown"
                : `${percent}%`;
    }
    if (detail) {
        const counts = progress.total_files
            ? ` · ${progress.processed_files || 0}/${progress.total_files} files`
            : "";
        detail.textContent = status === "failed"
            ? "Please try indexing again."
            : status === "unknown"
                ? "No recent indexing status is available."
            : `${label}${counts}`;
    }
    if (stateLabel) {
        stateLabel.textContent = label;
        stateLabel.classList.toggle("is-ready", status === "completed");
        stateLabel.classList.toggle("is-error", status === "failed" || status === "unknown");
    }
    if (statusText) statusText.textContent = label;
}

function initializeChat() {
    const form =
        document.getElementById("chatForm");

    const input =
        document.getElementById("questionInput");

    const sendButton =
        document.getElementById("sendButton");

    if (!form || !input) return;

    document.getElementById("clearChat")?.addEventListener("click", () => {
        activeConversationId = null;
        const messages = getChatMessages();
        if (messages) {
            messages.innerHTML = `
                <article class="message assistant-message">
                    <div class="message-avatar"><i data-lucide="scan-search"></i></div>
                    <div class="message-body">
                        <div class="message-meta"><strong>CodeLens</strong><span>New conversation</span></div>
                        <div class="message-content"><p>Ask a question to explore the selected repository.</p></div>
                    </div>
                </article>
            `;
        }
        updateSourcesSection([], []);
        document.getElementById("historyContext")?.setAttribute("hidden", "");
        refreshIcons();
    });

    document.getElementById("closeHistoryContext")?.addEventListener("click", () => {
        document.getElementById("historyContext")?.setAttribute("hidden", "");
    });

    form.addEventListener(
        "submit",
        async event => {
            event.preventDefault();

            const question =
                input.value.trim();

            if (!question || isBusy) {
                return;
            }

            await askCodeLens(
                question,
                input,
                sendButton
            );
        }
    );
}

async function askCodeLens(
    question,
    input,
    sendButton
) {
    const repository =
        currentRepository ||
        loadCurrentRepository();

    if (!repository?.repository_id) {
        showToast(
            "Upload a repository before asking questions.",
            true
        );

        return;
    }

    isBusy = true;

    if (sendButton) {
        sendButton.disabled = true;
    }

    const conversation = ensureConversation(repository, question);
    appendConversationMessage(conversation, {
        role: "user",
        content: question,
        created_at: Date.now()
    });
    appendUserMessage(question);

    input.value = "";

    resizeTextarea(input);

    const thinking =
        appendThinkingMessage();

    try {
        const data = await apiRequest(
            `${CODELENS_API_BASE}/api/repositories/${encodeURIComponent(repository.repository_id)}/ask`,
            {
                method: "POST",
                headers: {
                    "Content-Type":
                        "application/json"
                },
                body: JSON.stringify({
                    question,
                    top_k: 5
                })
            },
            "CodeLens could not answer the question."
        );

        thinking.remove();

        const answer = data.answer ||
            "I couldn't find enough evidence in the uploaded repository.";
        appendAssistantMessage(answer);

        updateSourcesSection(
            data.sources || [],
            data.retrieved_chunks || []
        );
        appendConversationMessage(conversation, {
            role: "assistant",
            content: answer,
            sources: data.sources || [],
            retrieved_chunks: data.retrieved_chunks || [],
            created_at: Date.now()
        });

    } catch (error) {
        thinking.remove();

        appendErrorMessage(error.message || "CodeLens could not answer the question. Please try again.");

        console.error(
            "CodeLens request error:",
            error
        );

    } finally {
        isBusy = false;

        if (sendButton) {
            sendButton.disabled = false;
        }

        scrollChatToBottom();
    }
}

function getChatMessages() {
    return document.getElementById(
        "chatMessages"
    );
}

function appendUserMessage(question) {
    const container =
        getChatMessages();

    if (!container) return;

    const article =
        document.createElement(
            "article"
        );

    article.className =
        "chat-message user-message";

    article.innerHTML = `
        <div class="message-avatar">S</div>
        <div class="message-content">
            <div class="message-role">You</div>
            <div class="message-bubble">${escapeHtml(question)}</div>
        </div>
    `;

    container.appendChild(article);
}

function appendThinkingMessage() {
    const container =
        getChatMessages();

    const article =
        document.createElement(
            "article"
        );

    article.className =
        "chat-message assistant-message thinking-message";

    article.innerHTML = `
        <div class="assistant-avatar"><span></span></div>
        <div class="message-content">
            <div class="message-role">CodeLens Assistant</div>
            <div class="message-bubble">
                Thinking about your repository...
            </div>
        </div>
    `;

    if (container) {
        container.appendChild(article);
    }

    scrollChatToBottom();

    return article;
}

function appendAssistantMessage(answer) {
    const container =
        getChatMessages();

    if (!container) return;
        const article =
        document.createElement(
            "article"
        );

    article.className =
        "chat-message assistant-message";

    article.innerHTML = `
        <div class="assistant-avatar"><span></span></div>
        <div class="message-content">
            <div class="message-role">CodeLens Assistant</div>
            <div class="message-bubble assistant-answer">
                ${formatAnswer(answer)}
            </div>
        </div>
    `;

    container.appendChild(article);

    scrollChatToBottom();
}

function appendErrorMessage(message) {
    const container =
        getChatMessages();

    if (!container) return;

    const article =
        document.createElement(
            "article"
        );

    article.className =
        "chat-message assistant-message";

    article.innerHTML = `
        <div class="assistant-avatar"><span></span></div>
        <div class="message-content">
            <div class="message-role">CodeLens Assistant</div>
            <div class="message-bubble assistant-error">
                ${escapeHtml(message)}

                <div class="assistant-error-hint">
                    Please check that the backend is running and try again.
                </div>
            </div>
        </div>
    `;

    container.appendChild(article);

    scrollChatToBottom();
}

function formatAnswer(answer) {
    const safe =
        escapeHtml(answer);

    const parts =
        safe.split(/```/);

    return parts
        .map((part, index) => {
            if (index % 2 === 1) {
                const lines =
                    part.split("\n");

                if (
                    lines[0] &&
                    /^[a-zA-Z0-9_+#.-]+$/.test(
                        lines[0].trim()
                    )
                ) {
                    lines.shift();
                }

                return `
                    <pre class="assistant-code">
                        <code>${lines.join("\n")}</code>
                    </pre>
                `;
            }

            return part
                .split(/\n\n+/)
                .map(
                    paragraph =>
                        paragraph.trim()
                            ? `<p>${paragraph.replace(
                                /\n/g,
                                "<br>"
                            )}</p>`
                            : ""
                )
                .join("");
        })
        .join("");
}

function scrollChatToBottom() {
    const container =
        getChatMessages();

    if (!container) return;

    window.requestAnimationFrame(() => {
        container.scrollTop =
            container.scrollHeight;
    });
}

function initializeSuggestions() {
    document.querySelectorAll(
        "[data-question], .suggestion-card, .prompt-card"
    ).forEach(element => {
        element.addEventListener(
            "click",
            () => {
                const input =
                    document.getElementById(
                        "questionInput"
                    );

                if (!input) return;

                const question =
                    element.dataset.question ||
                    element
                        .querySelector("strong")
                        ?.textContent
                        ?.trim() ||
                    element.textContent.trim();

                if (!question) return;

                input.value =
                    question.replace(
                        /[“”]/g,
                        ""
                    );

                input.focus();

                resizeTextarea(input);
            }
        );
    });
}

function initializeTextarea() {
    const input =
        document.getElementById(
            "questionInput"
        );

    if (!input) return;

    input.addEventListener(
        "input",
        () => resizeTextarea(input)
    );

    input.addEventListener(
        "keydown",
        event => {
            if (
                event.key === "Enter" &&
                !event.shiftKey
            ) {
                event.preventDefault();

                document
                    .getElementById("chatForm")
                    ?.requestSubmit();
            }
        }
    );
}

function resizeTextarea(input) {
    input.style.height = "auto";

    input.style.height =
        `${Math.min(
            input.scrollHeight,
            180
        )}px`;
}

function updateRepositoryUI(repository) {
    if (!repository) return;

    currentRepository =
        repository;

    saveCurrentRepository(
        repository
    );

    const name =
        repository.name ||
        "Repository";

    const count =
        Number(
            repository.file_count ||
            repository.files?.length ||
            0
        );

    setText(
        "currentRepositoryName",
        name
    );

    setText(
        "currentRepositoryFileCount",
        `${count} files`
    );

    setText(
        "breadcrumbRepositoryName",
        name
    );

    setText(
        "heroRepositoryName",
        name
    );

    setText(
        "heroRepositoryFileCount",
        count
    );

    setText(
        "fileCount",
        count
    );

    setText(
        "repositoryStatusText",
        repository.indexed ? "Ready to explore" : "Repository selected"
    );

    renderRepositoryFiles(
        repository.files || []
    );

    updateSourcesSection([], []);
    setPanelMessage("repositorySearchResults", "Search code and documentation to find matching repository files.");
    document.getElementById("impactDetails")?.remove();
    renderHistory();
    loadRepositoryDashboard(repository);
}

function initializeDashboard() {
    checkBackendStatus();

    document.getElementById("repositorySearchForm")?.addEventListener("submit", async event => {
        event.preventDefault();
        const query = document.getElementById("repositorySearchInput")?.value.trim();
        const repository = currentRepository;
        const results = document.getElementById("repositorySearchResults");
        if (!query || !repository?.repository_id || !results) {
            if (!repository?.repository_id) showToast("Select a repository before searching.", true);
            return;
        }

        results.innerHTML = '<p class="panel-empty">Searching repository files…</p>';
        const button = document.getElementById("repositorySearchButton");
        if (button) button.disabled = true;
        try {
            const data = await apiRequest(
                `${CODELENS_API_BASE}/api/repositories/${encodeURIComponent(repository.repository_id)}/search`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query, top_k: 10 })
                },
                "Repository search could not be completed."
            );
            renderSearchResults(data.results || []);
        } catch (error) {
            results.innerHTML = `<p class="panel-empty">${escapeHtml(error.message)}</p>`;
        } finally {
            if (button) button.disabled = false;
        }
    });

    document.getElementById("refreshDependencies")?.addEventListener("click", () => {
        if (currentRepository) loadDependencyAnalysis(currentRepository);
    });

    document.getElementById("dependencyFileSelect")?.addEventListener("change", event => {
        renderDependencyGraph(currentDependencyAnalysis?.dependency_graph, event.target.value);
        if (event.target.value) loadFileImpact(event.target.value);
    });
}

async function checkBackendStatus() {
    const status = document.getElementById("backendStatus");
    const label = document.getElementById("connectionStatusText");
    if (!status || !label) return;

    try {
        await apiRequest(`${CODELENS_API_BASE}/health`, {}, "CodeLens is unavailable.");
        label.textContent = "Connected";
        status.classList.remove("is-offline");
    } catch (error) {
        console.error("Backend health check failed:", error);
        label.textContent = "Unavailable";
        status.classList.add("is-offline");
    }
}

let currentDependencyAnalysis = null;
let currentImpactAnalysis = null;

async function loadRepositoryDashboard(repository) {
    const version = ++repositoryLoadVersion;
    const repositoryId = repository?.repository_id;
    if (!repositoryId) return;

    const state = document.getElementById("analysisState");
    if (state) {
        state.textContent = "Loading repository insights";
        state.classList.remove("is-error", "is-ready");
    }

    const requests = await Promise.allSettled([
        apiRequest(`${CODELENS_API_BASE}/api/repositories/${encodeURIComponent(repositoryId)}/overview`, {}, "Repository overview is unavailable."),
        apiRequest(`${CODELENS_API_BASE}/api/repositories/${encodeURIComponent(repositoryId)}/technology-stack`, {}, "Technology details are unavailable."),
        apiRequest(`${CODELENS_API_BASE}/api/repositories/${encodeURIComponent(repositoryId)}/important-files?top_k=8`, {}, "Important files are unavailable."),
        apiRequest(`${CODELENS_API_BASE}/api/repositories/${encodeURIComponent(repositoryId)}/dependencies`, {}, "Dependency analysis is unavailable."),
        apiRequest(`${CODELENS_API_BASE}/api/repositories/${encodeURIComponent(repositoryId)}/impact`, {}, "Change-impact analysis is unavailable."),
        apiRequest(`${CODELENS_API_BASE}/api/repositories/${encodeURIComponent(repositoryId)}/progress`, {}, "Indexing status is unavailable.")
    ]);

    if (version !== repositoryLoadVersion || currentRepository?.repository_id !== repositoryId) return;

    const [overview, technology, important, dependencies, impact, progress] = requests;
    if (overview.status === "fulfilled") {
        renderRepositoryOverview(overview.value.overview || {});
    } else {
        setText("overviewDescription", "Repository summary could not be loaded. Try selecting the repository again.");
    }
    if (technology.status === "fulfilled") {
        renderTechnologyStack(technology.value.technology_stack || {});
    } else {
        setPanelMessage("technologyList", technology.reason.message);
    }
    if (important.status === "fulfilled") {
        renderImportantFiles(important.value.important_files || []);
    } else {
        setPanelMessage("importantFileList", important.reason.message);
    }
    if (dependencies.status === "fulfilled") {
        currentDependencyAnalysis = dependencies.value.dependency_analysis || {};
        renderDependencyAnalysis(currentDependencyAnalysis);
    } else {
        currentDependencyAnalysis = null;
        setPanelMessage("dependencyGraph", dependencies.reason.message);
        setText("dependencySummary", dependencies.reason.message);
    }
    if (impact.status === "fulfilled") {
        currentImpactAnalysis = impact.value.impact_analysis || {};
        renderImpactSummary(currentImpactAnalysis);
    } else {
        currentImpactAnalysis = null;
        setPanelMessage("impactList", impact.reason.message);
    }
    if (progress.status === "fulfilled") {
        setIndexProgress(progress.value);
    } else {
        setIndexProgress({ status: "unknown", stage: "unknown", progress: 0 });
    }

    if (state && [overview, technology, important].some(result => result.status === "fulfilled")) {
        state.textContent = "Repository insights loaded";
        state.classList.add("is-ready");
    } else if (state) {
        state.textContent = "Some insights unavailable";
        state.classList.add("is-error");
    }
}

function renderRepositoryOverview(overview) {
    const languages = overview.languages || {};
    const fileTypes = overview.file_types || {};
    const entries = overview.entry_point_candidates || [];
    setText("overviewFileCount", overview.total_files ?? "—");
    setText("overviewFileTypes", `${Object.keys(fileTypes).length} file types detected`);
    setText("overviewLanguageCount", Object.keys(languages).length);
    setText("overviewLanguages", Object.keys(languages).slice(0, 3).join(", ") || "No languages detected");
    setText("overviewEntryCount", entries.length);
    setText("overviewEntryPoints", entries.slice(0, 2).join(", ") || "No likely entry points found");
    setText("overviewDescription", `${overview.total_files ?? 0} files · ${formatBytes(overview.total_size_bytes || 0)} · ${overview.directories?.length || 0} directories`);
}

function formatBytes(bytes) {
    if (!bytes) return "0 bytes";
    const units = ["bytes", "KB", "MB", "GB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit += 1;
    }
    return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

function renderTechnologyStack(stack) {
    const list = document.getElementById("technologyList");
    if (!list) return;
    const labels = {
        languages: "Languages",
        frameworks: "Frameworks",
        ai_ml: "AI & machine learning",
        databases: "Databases",
        data_science: "Data tools",
        libraries: "Libraries",
        testing: "Testing",
        build_tools: "Build tools",
        package_managers: "Package managers"
    };
    const tags = Object.entries(stack)
        .flatMap(([category, items]) => (Array.isArray(items) ? items : []).map(name => ({
            name,
            category: labels[category] || category
        })));
    setText("technologyCount", `${tags.length} detected`);
    list.replaceChildren();
    if (!tags.length) {
        setPanelMessage("technologyList", "No technology details were detected.");
        return;
    }
    tags.slice(0, 24).forEach(tag => {
        const item = document.createElement("span");
        item.className = "technology-tag";
        item.textContent = tag.name;
        item.title = tag.category;
        list.appendChild(item);
    });
}

function renderImportantFiles(files) {
    const list = document.getElementById("importantFileList");
    if (!list) return;
    setText("importantFileCount", `${files.length} ranked`);
    list.replaceChildren();
    if (!files.length) {
        setPanelMessage("importantFileList", "No important files were identified.");
        return;
    }
    files.slice(0, 6).forEach(file => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "important-file-item";
        button.innerHTML = `<i data-lucide="file-code-2"></i><span>${escapeHtml(file.file_path || "Repository file")}</span><small>${escapeHtml(file.importance_level || "Relevant")}</small>`;
        button.addEventListener("click", () => openSourcePreview(file.file_path, 1));
        list.appendChild(button);
    });
    refreshIcons();
}

function setPanelMessage(id, message) {
    const panel = document.getElementById(id);
    if (!panel) return;
    panel.replaceChildren();
    const text = document.createElement("p");
    text.className = "panel-empty";
    text.textContent = message;
    panel.appendChild(text);
}

function renderSearchResults(results) {
    const container = document.getElementById("repositorySearchResults");
    if (!container) return;
    container.replaceChildren();
    if (!results.length) {
        setPanelMessage("repositorySearchResults", "No matching repository content was found.");
        updateSourcesSection([], []);
        return;
    }
    results.forEach(result => {
        const metadata = result.metadata || {};
        const file = String(metadata.file_path || "Repository file").replace(/\\/g, "/");
        const button = document.createElement("button");
        button.type = "button";
        button.className = "search-result-item";
        const title = document.createElement("strong");
        title.textContent = file;
        const excerpt = document.createElement("small");
        excerpt.textContent = String(result.content || "").slice(0, 260);
        button.append(title, excerpt);
        button.addEventListener("click", () => openSourcePreview(file, metadata.start_line || 1));
        container.appendChild(button);
    });
    updateSourcesSection([], results);
}

async function loadDependencyAnalysis(repository) {
    const button = document.getElementById("refreshDependencies");
    if (button) button.disabled = true;
    setPanelMessage("dependencyGraph", "Refreshing dependency relationships…");
    try {
        const data = await apiRequest(
            `${CODELENS_API_BASE}/api/repositories/${encodeURIComponent(repository.repository_id)}/dependencies`,
            {},
            "Dependency analysis is unavailable."
        );
        currentDependencyAnalysis = data.dependency_analysis || {};
        renderDependencyAnalysis(currentDependencyAnalysis);
    } catch (error) {
        setPanelMessage("dependencyGraph", error.message);
        setText("dependencySummary", error.message);
    } finally {
        if (button) button.disabled = false;
    }
}

function renderDependencyAnalysis(analysis) {
    const graph = analysis.dependency_graph || { nodes: [], edges: [] };
    const selector = document.getElementById("dependencyFileSelect");
    if (selector) {
        selector.replaceChildren(new Option("All files", ""));
        (graph.nodes || []).forEach(node => selector.add(new Option(node.id, node.id)));
        selector.disabled = !(graph.nodes || []).length;
    }
    const refresh = document.getElementById("refreshDependencies");
    if (refresh) refresh.disabled = false;
    const graphNote = graph.nodes?.length > 50 || graph.edges?.length > 100
        ? " · Map displays up to 50 files and 100 relationships"
        : "";
    const pythonFiles = Number(analysis.total_python_files || 0);
    const notebooks = Number(analysis.total_notebooks || 0);
    const fileSummary = `${pythonFiles} Python files${notebooks ? ` · ${notebooks} ${notebooks === 1 ? "notebook" : "notebooks"} analyzed` : ""}`;
    setText("dependencySummary", `${fileSummary} · ${graph.edges?.length || 0} internal import relationships${graphNote}`);
    renderDependencyGraph(graph);
}

function renderDependencyGraph(graph = {}, focusedFile = "") {
    const container = document.getElementById("dependencyGraph");
    if (!container) return;
    const allNodes = graph.nodes || [];
    const allEdges = graph.edges || [];
    let nodes = allNodes;
    let edges = allEdges;
    if (focusedFile) {
        const adjacent = new Set([focusedFile]);
        allEdges.forEach(edge => {
            if (edge.source === focusedFile) adjacent.add(edge.target);
            if (edge.target === focusedFile) adjacent.add(edge.source);
        });
        nodes = allNodes.filter(node => adjacent.has(node.id));
        edges = allEdges.filter(edge => adjacent.has(edge.source) && adjacent.has(edge.target));
    }
    nodes = nodes.slice(0, 50);
    const nodeIds = new Set(nodes.map(node => node.id));
    edges = edges.filter(edge => nodeIds.has(edge.source) && nodeIds.has(edge.target)).slice(0, 100);
    container.replaceChildren();
    if (!nodes.length) {
        setPanelMessage("dependencyGraph", "No Python files or notebook code cells were available for dependency analysis.");
        return;
    }

    const ns = "http://www.w3.org/2000/svg";
    const rowCount = Math.ceil(nodes.length / 2);
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 820 ${Math.max(150, rowCount * 58 + 28)}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Repository file dependency graph. Select a file to inspect its impact.");
    const defs = document.createElementNS(ns, "defs");
    const marker = document.createElementNS(ns, "marker");
    marker.setAttribute("id", "dependencyArrow");
    marker.setAttribute("viewBox", "0 0 10 10");
    marker.setAttribute("refX", "8");
    marker.setAttribute("refY", "5");
    marker.setAttribute("markerWidth", "5");
    marker.setAttribute("markerHeight", "5");
    marker.setAttribute("orient", "auto-start-reverse");
    const arrow = document.createElementNS(ns, "path");
    arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    arrow.setAttribute("fill", "rgba(129, 140, 248, 0.72)");
    marker.appendChild(arrow);
    defs.appendChild(marker);
    svg.appendChild(defs);
    const positions = new Map();
    nodes.forEach((node, index) => {
        positions.set(node.id, {
            x: index % 2 === 0 ? 24 : 510,
            y: 18 + Math.floor(index / 2) * 58
        });
    });
    edges.forEach(edge => {
        const source = positions.get(edge.source);
        const target = positions.get(edge.target);
        const line = document.createElementNS(ns, "path");
        const sourceOnLeft = source.x < target.x;
        const startX = sourceOnLeft ? source.x + 270 : source.x;
        const endX = sourceOnLeft ? target.x : target.x + 270;
        line.setAttribute("d", `M ${startX} ${source.y + 17} C ${startX + 70} ${source.y + 17}, ${endX - 70} ${target.y + 17}, ${endX} ${target.y + 17}`);
        line.setAttribute("class", "dependency-edge");
        line.setAttribute("marker-end", "url(#dependencyArrow)");
        svg.appendChild(line);
    });
    nodes.forEach(node => {
        const position = positions.get(node.id);
        const group = document.createElementNS(ns, "g");
        group.setAttribute("class", `dependency-node${node.id === focusedFile ? " is-selected" : ""}`);
        group.setAttribute("transform", `translate(${position.x} ${position.y})`);
        group.setAttribute("tabindex", "0");
        group.setAttribute("role", "button");
        group.setAttribute("aria-label", `View change impact for ${node.id}`);
        const rect = document.createElementNS(ns, "rect");
        rect.setAttribute("width", "270");
        rect.setAttribute("height", "34");
        rect.setAttribute("rx", "7");
        const text = document.createElementNS(ns, "text");
        text.setAttribute("x", "10");
        text.setAttribute("y", "22");
        const label = node.id.length > 38 ? `…${node.id.slice(-35)}` : node.id;
        text.textContent = label;
        group.append(rect, text);
        const activate = () => {
            const selector = document.getElementById("dependencyFileSelect");
            if (selector) selector.value = node.id;
            renderDependencyGraph(graph, node.id);
            loadFileImpact(node.id);
        };
        group.addEventListener("click", activate);
        group.addEventListener("keydown", event => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                activate();
            }
        });
        svg.appendChild(group);
    });
    container.appendChild(svg);
}

function renderImpactSummary(analysis) {
    const container = document.getElementById("impactList");
    if (!container) return;
    const files = analysis.files_with_impact || [];
    container.replaceChildren();
    if (!files.length) {
        setPanelMessage("impactList", "No Python file impact summary is available.");
        return;
    }
    files.slice(0, 12).forEach(item => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "impact-item";
        const badge = document.createElement("span");
        badge.className = `impact-badge ${(item.impact_level || "").toLowerCase()}`;
        badge.textContent = item.impact_level || "UNKNOWN";
        const path = document.createElement("span");
        path.textContent = item.file;
        const count = document.createElement("small");
        count.textContent = `${item.total_affected_files || 0} affected`;
        button.append(badge, path, count);
        button.addEventListener("click", () => loadFileImpact(item.file));
        container.appendChild(button);
    });
}

async function loadFileImpact(filePath) {
    const container = document.getElementById("impactList");
    if (!container || !currentRepository) return;
    try {
        const encodedPath = String(filePath).split("/").map(encodeURIComponent).join("/");
        const data = await apiRequest(
            `${CODELENS_API_BASE}/api/repositories/${encodeURIComponent(currentRepository.repository_id)}/impact/${encodedPath}`,
            {},
            "Change impact could not be loaded."
        );
        const impact = data.impact_analysis || {};
        let details = document.getElementById("impactDetails");
        if (!details) {
            details = document.createElement("section");
            details.id = "impactDetails";
            details.className = "impact-details";
            container.after(details);
        }
        const affected = [
            ...(impact.direct_dependents || []).map(path => `Direct: ${path}`),
            ...(impact.indirect_dependents || []).map(path => `Indirect: ${path}`),
            ...(impact.affected_tests || []).map(path => `Test: ${path}`),
            ...(impact.affected_entry_points || []).map(path => `Entry point: ${path}`)
        ];
        details.innerHTML = `<h3>${escapeHtml(impact.file || filePath)} · ${escapeHtml(impact.impact?.impact_level || "Impact")}</h3><p>${escapeHtml((impact.impact?.reasons || []).join(" "))}</p>${affected.length ? `<ul>${affected.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : '<p>No affected files were detected.</p>'}`;
    } catch (error) {
        setPanelMessage("impactList", error.message);
    }
}

function setText(id, value) {
    const element =
        document.getElementById(id);

    if (element) {
        element.textContent = value;
    }
}

function renderRepositoryFiles(files) {
    const tree =
        document.getElementById(
            "repositoryFileTree"
        );

    if (!tree) return;

    tree.innerHTML = "";

    if (!files.length) {
        tree.innerHTML =
            '<p class="history-empty">No files found.</p>';

        return;
    }
        const folders =
        new Map();

    files.forEach(filePath => {
        const normalized =
            String(filePath)
                .replace(/\\/g, "/");

        const parts =
            normalized
                .split("/")
                .filter(Boolean);

        if (parts.length === 1) {
            tree.appendChild(
                createTreeFile(
                    normalized
                )
            );

            return;
        }

        const folder =
            parts[0];

        if (!folders.has(folder)) {
            folders.set(
                folder,
                []
            );
        }

        folders
            .get(folder)
            .push(normalized);
    });

    folders.forEach(
        (folderFiles, folderName) => {
            const folderButton =
                document.createElement(
                    "button"
                );

            folderButton.className =
                "tree-folder";

            folderButton.type =
                "button";

            folderButton.innerHTML = `
                <i data-lucide="chevron-down"></i>
                <i data-lucide="folder"></i>
                <span>${escapeHtml(folderName)}</span>
            `;

            const children =
                document.createElement(
                    "div"
                );

            children.className =
                "tree-children";

            folderFiles.forEach(
                filePath => {
                    children.appendChild(
                        createTreeFile(
                            filePath
                        )
                    );
                }
            );

            folderButton.addEventListener(
                "click",
                () => {
                    children.classList.toggle(
                        "collapsed"
                    );

                    folderButton.classList.toggle(
                        "collapsed"
                    );
                }
            );

            tree.appendChild(
                folderButton
            );

            tree.appendChild(
                children
            );
        }
    );

    refreshIcons();
}

function createTreeFile(filePath) {
    const button =
        document.createElement(
            "button"
        );

    button.className =
        "tree-file";

    button.type =
        "button";

    button.innerHTML = `
        <i data-lucide="file-code-2"></i>
        <span title="${escapeHtml(filePath)}">
            ${escapeHtml(
                filePath.split("/").pop()
            )}
        </span>
    `;

    button.addEventListener(
        "click",
        () => openSourcePreview(filePath, 1)
    );

    return button;
}

function updateSourcesSection(
    sources,
    retrievedChunks
) {
    const grid =
        document.getElementById(
            "sourceGrid"
        );

    const count =
        document.getElementById(
            "sourceCount"
        );

    if (!grid) return;

    const normalized =
        normalizeSources(
            sources,
            retrievedChunks
        );

    if (count) {
        const fileCount = new Set(normalized.map(source => source.file)).size;
        count.textContent = fileCount
            ? `${fileCount} source file${fileCount === 1 ? "" : "s"}`
            : "No evidence yet";
    }

    if (!normalized.length) {
        setPanelMessage("sourceGrid", "No source evidence is available for this result.");
        return;
    }

    grid.replaceChildren();
    normalized.forEach(source => {
        const article = document.createElement("article");
        article.className = "source-card";
        const header = document.createElement("div");
        header.className = "source-top";
        const icon = document.createElement("div");
        icon.className = "source-file-icon python";
        icon.innerHTML = '<i data-lucide="file-code-2"></i>';
        const name = document.createElement("div");
        name.className = "source-file";
        const fileButton = document.createElement("button");
        fileButton.type = "button";
        fileButton.className = "source-file-button";
        fileButton.textContent = source.file;
        fileButton.title = `Open ${source.file}`;
        fileButton.addEventListener("click", () => openSourcePreview(source.file, source.startLine));
        const type = document.createElement("span");
        type.textContent = source.type || "Repository source";
        name.append(fileButton, type);
        const open = document.createElement("button");
        open.type = "button";
        open.className = "source-open";
        open.setAttribute("aria-label", `Open ${source.file}`);
        open.innerHTML = '<i data-lucide="external-link"></i>';
        open.addEventListener("click", () => openSourcePreview(source.file, source.startLine));
        header.append(icon, name, open);
        const lines = document.createElement("div");
        lines.className = "source-lines";
        lines.textContent = source.lines || "Line information unavailable";
        const preview = document.createElement("pre");
        preview.className = "code-preview";
        preview.textContent = source.preview || "No source excerpt was returned.";
        article.append(header, lines, preview);
        grid.appendChild(article);
    });
    refreshIcons();
}

function normalizeSources(
    sources,
    chunks
) {
    const items =
        Array.isArray(sources)
            ? sources
            : [];

    const chunkItems =
        Array.isArray(chunks)
            ? chunks
            : [];

    const result = new Map();

    items.forEach(item => {
        const metadata =
            item?.metadata ||
            item ||
            {};

        const file =
            String(metadata.file_path ||
            metadata.file ||
            item.file_path ||
            "Unknown file").replace(/\\/g, "/");

        const key = `${file}:${metadata.start_line ?? ""}:${metadata.end_line ?? ""}`;
        result.set(key, {
            file,
            type:
                metadata.chunk_type ||
                metadata.language ||
                "Source",
            lines:
                metadata.chunk_type === "notebook_cell" && metadata.cell_number
                    ? `Notebook cell ${metadata.cell_number}`
                    : metadata.start_line && metadata.end_line
                        ? `Lines ${metadata.start_line}–${metadata.end_line}`
                        : "Line information unavailable",
            preview: item.content || metadata.content || "",
            startLine: metadata.chunk_type === "notebook_cell" ? 1 : metadata.start_line || 1
        });
    });

    chunkItems.forEach(item => {
        const metadata =
            item?.metadata ||
            {};

        const file = String(metadata.file_path || "Unknown file").replace(/\\/g, "/");

        const key = `${file}:${metadata.start_line ?? ""}:${metadata.end_line ?? ""}`;
        const existing = result.get(key);
        if (existing) {
            if (!existing.preview) existing.preview = item.content || "";
            if (metadata.chunk_type === "notebook_cell" && metadata.cell_number) {
                existing.lines = `Notebook cell ${metadata.cell_number}`;
                existing.startLine = 1;
            }
            if (!existing.type || existing.type === "Source") {
                existing.type = metadata.chunk_type || metadata.language || "Source";
            }
            return;
        }

        result.set(key, {
            file,
            type:
                metadata.chunk_type ||
                metadata.language ||
                "Source",
            lines:
                metadata.chunk_type === "notebook_cell" && metadata.cell_number
                    ? `Notebook cell ${metadata.cell_number}`
                    : metadata.start_line && metadata.end_line
                        ? `Lines ${metadata.start_line}–${metadata.end_line}`
                        : "Line information unavailable",
            preview: item.content || "",
            startLine: metadata.chunk_type === "notebook_cell" ? 1 : metadata.start_line || 1
        });
    });

    return Array.from(result.values()).slice(0, 6);
}

function initializeSourcePreview() {
    const modal = document.getElementById("sourcePreviewModal");
    const closeButton = document.getElementById("closeSourcePreview");
    if (!modal) return;

    const close = () => {
        modal.classList.remove("visible");
        modal.setAttribute("aria-hidden", "true");
    };
    closeButton?.addEventListener("click", close);
    modal.addEventListener("click", event => {
        if (event.target === modal) close();
    });
    document.addEventListener("keydown", event => {
        if (event.key === "Escape" && modal.classList.contains("visible")) close();
    });
}

async function openSourcePreview(filePath, startLine = 1) {
    if (!currentRepository?.repository_id || !filePath) {
        showToast("Select a repository before opening a file.", true);
        return;
    }

    const modal = document.getElementById("sourcePreviewModal");
    const title = document.getElementById("sourcePreviewTitle");
    const range = document.getElementById("sourcePreviewRange");
    const code = document.getElementById("sourcePreviewCode");
    if (!modal || !title || !range || !code) return;

    title.textContent = filePath;
    range.textContent = "Loading source preview…";
    code.textContent = "";
    modal.classList.add("visible");
    modal.setAttribute("aria-hidden", "false");

    try {
        const encodedPath = String(filePath).split("/").map(encodeURIComponent).join("/");
        const firstLine = Number.isFinite(Number(startLine)) ? Math.max(1, Number(startLine)) : 1;
        const data = await apiRequest(
            `${CODELENS_API_BASE}/api/repositories/${encodeURIComponent(currentRepository.repository_id)}/files/${encodedPath}?start_line=${firstLine}&line_count=160`,
            {},
            "This file preview is unavailable."
        );
        title.textContent = data.file_path || filePath;
        range.textContent = `Lines ${data.start_line}–${data.end_line}`;
        code.textContent = data.content || "No source content was returned.";
    } catch (error) {
        range.textContent = error.message;
    }
}

function getHistory() {
    try {
        const value =
            localStorage.getItem(
                HISTORY_KEY
            );

        const history = value ? JSON.parse(value) : {};
        return {
            repositories: Array.isArray(history.repositories) ? history.repositories : [],
            conversations: Array.isArray(history.conversations) ? history.conversations : []
        };

    } catch (_) {
        return {
            repositories: [],
            conversations: []
        };
    }
}

function saveHistory(history) {
    localStorage.setItem(
        HISTORY_KEY,
        JSON.stringify(history)
    );
}

function addRepositoryToHistory(
    repository
) {
    const history =
        getHistory();

    const item = {
        repository_id:
            repository.repository_id,

        name:
            repository.name,

        file_count:
            repository.file_count ||
            repository.files?.length ||
            0,

        files:
            repository.files ||
            [],

        updated_at:
            Date.now()
    };

    history.repositories = [
        item,

        ...history.repositories.filter(
            r =>
                r.repository_id !==
                item.repository_id
        )
    ].slice(0, 10);

    saveHistory(history);

    renderHistory();
}

function ensureConversation(repository, question) {
    const history =
        getHistory();
    let conversation = history.conversations.find(item => item.id === activeConversationId);
    if (!conversation || conversation.repository_id !== repository.repository_id) {
        conversation = {
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            repository_id: repository.repository_id,
            repository_name: repository.name || "Repository",
            question,
            messages: [],
            created_at: Date.now(),
            updated_at: Date.now()
        };
        history.conversations = [conversation, ...history.conversations].slice(0, 20);
        activeConversationId = conversation.id;
    }

    saveHistory(history);
    renderHistory();
    return conversation;
}

function appendConversationMessage(conversation, message) {
    const history = getHistory();
    const stored = history.conversations.find(item => item.id === conversation.id);
    if (!stored) return;
    if (!Array.isArray(stored.messages)) stored.messages = [];

    const safeMessage = { ...message };
    if (Array.isArray(safeMessage.retrieved_chunks)) {
        safeMessage.retrieved_chunks = safeMessage.retrieved_chunks.slice(0, 6).map(chunk => ({
            chunk_id: chunk.chunk_id,
            content: String(chunk.content || "").slice(0, 3000),
            metadata: chunk.metadata || {}
        }));
    }
    if (Array.isArray(safeMessage.sources)) {
        safeMessage.sources = safeMessage.sources.slice(0, 6);
    }
    stored.messages.push(safeMessage);
    stored.updated_at = Date.now();
    saveHistory(history);
    renderHistory();
}

function initializeHistory() {
    renderHistory();

    const clearButton =
        document.getElementById(
            "clearHistoryButton"
        );

    if (clearButton) {
        clearButton.addEventListener(
            "click",
            () => {
                localStorage.removeItem(
                    HISTORY_KEY
                );

                renderHistory();

                showToast(
                    "History cleared."
                );
            }
        );
    }
}

function renderHistory() {
    const list =
        document.getElementById(
            "historyList"
        );

    const empty =
        document.getElementById(
            "historyEmpty"
        );

    if (!list) return;

    const history =
        getHistory();

    if (
        !history.repositories.length &&
        !history.conversations.length
    ) {
        list.innerHTML = "";

        if (empty) {
            empty.hidden = false;
        }

        return;
    }

    if (empty) {
        empty.hidden = true;
    }

    list.innerHTML = "";

    history.repositories
        .slice(0, 5)
        .forEach(repository => {
            const button =
                document.createElement(
                    "button"
                );

            button.className =
                "history-item";

            button.type =
                "button";

            button.innerHTML = `
                <span class="history-icon repository-history-icon">
                    <i data-lucide="box"></i>
                </span>

                <span class="history-info">
                    <strong>
                        ${escapeHtml(
                            repository.name
                        )}
                    </strong>

                    <small>
                        Repository · ${
                            repository.file_count ||
                            0
                        } files
                    </small>
                </span>

                <i
                    class="history-arrow"
                    data-lucide="chevron-right"
                ></i>
            `;

            button.addEventListener(
                "click",
                () =>
                    openRepositoryFromHistory(
                        repository
                    )
            );

            list.appendChild(
                button
            );
        });

    history.conversations
        .slice(0, 5)
        .forEach(conversation => {
            const button =
                document.createElement(
                    "button"
                );

            button.className =
                "history-item";

            button.type =
                "button";

            button.innerHTML = `
                <span class="history-icon conversation-history-icon">
                    <i data-lucide="message-square"></i>
                </span>

                <span class="history-info">
                    <strong>
                        ${escapeHtml(
                            conversation.question || "Conversation"
                        )}
                    </strong>

                    <small>
                        ${escapeHtml(
                            conversation.repository_name
                        )}
                        · Conversation
                    </small>
                </span>

                <i
                    class="history-arrow"
                    data-lucide="chevron-right"
                ></i>
            `;

            button.addEventListener(
                "click",
                () =>
                    openConversationFromHistory(
                        conversation
                    )
            );

            list.appendChild(
                button
            );
        });

    refreshIcons();
}

function openRepositoryFromHistory(
    repository
) {
    saveCurrentRepository(
        repository
    );

    updateRepositoryUI(
        repository
    );

    showToast(
        `${repository.name} selected.`
    );
}

function openConversationFromHistory(
    conversation
) {
    const repository =
        getHistory()
            .repositories
            .find(
                item =>
                    item.repository_id ===
                    conversation.repository_id
            );

    if (repository) {
        updateRepositoryUI(
            repository
        );
    } else {
        showToast("This conversation's repository is no longer available.", true);
        return;
    }

    activeConversationId = conversation.id;
    const messages = getChatMessages();
    if (messages && Array.isArray(conversation.messages) && conversation.messages.length) {
        messages.replaceChildren();
        conversation.messages.forEach(message => {
            if (message.role === "user") {
                appendUserMessage(message.content || "");
            } else if (message.role === "assistant") {
                appendAssistantMessage(message.content || "");
                updateSourcesSection(message.sources || [], message.retrieved_chunks || []);
            }
        });
    } else {
        const input = document.getElementById("questionInput");
        if (input) {
            input.value = conversation.question || "";
            input.focus();
            resizeTextarea(input);
        }
    }
    setText("historyContextTitle", conversation.question || "Previous conversation");
    setText("historyContextRepository", conversation.repository_name || repository.name || "Repository");
    document.getElementById("historyContext")?.removeAttribute("hidden");
    scrollChatToBottom();
}
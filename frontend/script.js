const CODELENS_API_BASE = "http://127.0.0.1:8000";
const CURRENT_REPOSITORY_KEY = "codelens_current_repository";
const HISTORY_KEY = "codelens_history";
let selectedRepositoryFile = null;
let currentRepository = null;
let isBusy = false;

async function readApiResponse(response) {
    const text = await response.text();
    if (!text) return {};
    try {
        return JSON.parse(text);
    } catch (_) {
        return { detail: text };
    }
}

function apiError(data, fallback) {
    if (!data) return fallback;
    if (typeof data === "string" && data.trim()) return data;
    if (typeof data.detail === "string" && data.detail.trim()) return data.detail;
    if (typeof data.message === "string" && data.message.trim()) return data.message;
    return fallback;
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

    const saved = loadCurrentRepository();

    if (saved) {
        saveCurrentRepository(saved);
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

        const uploadResponse = await fetch(
            `${CODELENS_API_BASE}/api/repositories/upload`,
            {
                method: "POST",
                body: formData
            }
        );

        const uploadData =
            await readApiResponse(
                uploadResponse
            );

        if (!uploadResponse.ok) {
            throw new Error(
                apiError(
                    uploadData,
                    "Repository upload failed."
                )
            );
        }

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

        button.innerHTML =
            '<i data-lucide="loader-circle"></i> Indexing...';

        refreshIcons();

        const indexResponse = await fetch(
            `${CODELENS_API_BASE}/api/repositories/${encodeURIComponent(repositoryId)}/index`,
            {
                method: "POST"
            }
        );

        const indexData =
            await readApiResponse(
                indexResponse
            );
                    if (!indexResponse.ok) {
            throw new Error(
                apiError(
                    indexData,
                    "Repository indexing failed."
                )
            );
        }

        saveCurrentRepository(state);

        addRepositoryToHistory(state);

        updateRepositoryUI(state);

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

        showToast(
            error.message ||
            "Could not analyze the repository.",
            true
        );

    } finally {
        isBusy = false;
    }
}

function initializeChat() {
    const form =
        document.getElementById("chatForm");

    const input =
        document.getElementById("questionInput");

    const sendButton =
        document.getElementById("sendButton");

    if (!form || !input) return;

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

    appendUserMessage(question);

    input.value = "";

    resizeTextarea(input);

    const thinking =
        appendThinkingMessage();

    try {
        const response = await fetch(
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
            }
        );

        const data =
            await readApiResponse(
                response
            );

        thinking.remove();

        if (!response.ok) {
            throw new Error(
                apiError(
                    data,
                    "CodeLens could not answer the question."
                )
            );
        }

        appendAssistantMessage(
            data.answer ||
            "I couldn't find enough evidence in the uploaded repository."
        );

        updateSourcesSection(
            data.sources || [],
            data.retrieved_chunks || []
        );

        addConversationToHistory(
            repository.repository_id,
            repository.name,
            question
        );

    } catch (error) {
        thinking.remove();

        appendErrorMessage(
            error.message ||
            "Unable to reach CodeLens."
        );

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
        "Repository ready"
    );

    renderRepositoryFiles(
        repository.files || []
    );

    renderHistory();
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
        () => showToast(filePath)
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
        count.textContent =
            `${normalized.length} relevant file${
                normalized.length === 1
                    ? ""
                    : "s"
            }`;
    }

    if (!normalized.length) {
        grid.innerHTML =
            '<p class="history-empty">No source files were returned.</p>';

        return;
    }

    grid.innerHTML =
        normalized
            .map(
                source => `
                    <article class="source-card">
                        <div class="source-top">
                            <div class="source-file-icon python">
                                <i data-lucide="file-code-2"></i>
                            </div>

                            <div class="source-file">
                                <strong>
                                    ${escapeHtml(
                                        source.file
                                    )}
                                </strong>

                                <span>
                                    ${escapeHtml(
                                        source.type ||
                                        source.name ||
                                        "Repository source"
                                    )}
                                </span>
                            </div>
                        </div>

                        <div class="source-lines">
                            ${escapeHtml(
                                source.lines ||
                                "Relevant code"
                            )}
                        </div>

                        <div class="code-preview">
                            ${escapeHtml(
                                source.preview ||
                                "Relevant repository context was retrieved for this answer."
                            )}
                        </div>
                    </article>
                `
            )
            .join("");

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

    const result = [];
    const seen = new Set();

    items.forEach(item => {
        const metadata =
            item?.metadata ||
            item ||
            {};

        const file =
            metadata.file_path ||
            metadata.file ||
            item.file_path ||
            "Unknown file";

        if (seen.has(file)) {
            return;
        }

        seen.add(file);

        result.push({
            file,

            type:
                metadata.chunk_type ||
                metadata.language ||
                "Source",

            lines:
                metadata.start_line &&
                metadata.end_line
                    ? `Lines ${metadata.start_line}–${metadata.end_line}`
                    : "Relevant code",

            preview:
                item.content ||
                metadata.content ||
                "Relevant repository context."
        });
    });

    chunkItems.forEach(item => {
        const metadata =
            item?.metadata ||
            {};

        const file =
            metadata.file_path ||
            "Unknown file";

        if (seen.has(file)) {
            return;
        }

        seen.add(file);

        result.push({
            file,

            type:
                metadata.chunk_type ||
                metadata.language ||
                "Source",

            lines:
                metadata.start_line &&
                metadata.end_line
                    ? `Lines ${metadata.start_line}–${metadata.end_line}`
                    : "Retrieved chunk",

            preview:
                item.content ||
                "Retrieved repository context."
        });
    });

    return result.slice(
        0,
        6
    );
}
function getHistory() {
    try {
        const value =
            localStorage.getItem(
                HISTORY_KEY
            );

        return value
            ? JSON.parse(value)
            : {
                repositories: [],
                conversations: []
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

function addConversationToHistory(
    repositoryId,
    repositoryName,
    question
) {
    const history =
        getHistory();

    const item = {
        id:
            `${Date.now()}-${Math.random()
                .toString(36)
                .slice(2, 8)}`,

        repository_id:
            repositoryId,

        repository_name:
            repositoryName,

        question,

        created_at:
            Date.now()
    };

    history.conversations = [
        item,
        ...history.conversations
    ].slice(0, 20);

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
                            conversation.question
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
        saveCurrentRepository(
            repository
        );

        updateRepositoryUI(
            repository
        );
    }

    const input =
        document.getElementById(
            "questionInput"
        );

    if (input) {
        input.value =
            conversation.question;

        input.focus();

        resizeTextarea(
            input
        );
    }
}
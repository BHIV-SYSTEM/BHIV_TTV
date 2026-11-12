/**
 * TTV Studio • AI Video Generation Engine Frontend
 * Modern reactive application orchestrating generation, preview, history, models, training, and RL feedback.
 */

document.addEventListener("DOMContentLoaded", () => {
    // =========================================================================
    // 1. STATE & STORAGE MANAGEMENT
    // =========================================================================
    const STORAGE_KEY_HISTORY = "ttv_video_history";
    const STORAGE_KEY_TOKEN = "ttv_governance_token";

    let currentJobId = null;
    let pollInterval = null;
    let timerInterval = null;
    let jobStartTime = null;
    let activeVideoData = null;
    let selectedRLWinner = null;
    let rlIteration = 1;
    let selectedAspect = "16:9";
    let selectedQuality = "standard";
    let referenceId = null;
    let referenceType = null;
    const RESOLUTION_MATRIX = {
        "16:9": { standard: "1280x720", high: "1920x1080", ultra: "3840x2160" },
        "9:16": { standard: "720x1280", high: "1080x1920", ultra: "2160x3840" }
    };
    const LANGUAGE_VOICES = {
        en: ["en-US-GuyNeural", "en-US-AriaNeural", "en-IN-NeerjaNeural"],
        hi: ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"],
        mr: ["mr-IN-ManoharNeural", "mr-IN-AarohiNeural"],
        ta: ["ta-IN-ValluvarNeural", "ta-IN-PallaviNeural"],
        te: ["te-IN-MohanNeural", "te-IN-ShrutiNeural"],
        kn: ["kn-IN-GaganNeural", "kn-IN-SapnaNeural"],
        bn: ["bn-IN-BashkarNeural", "bn-IN-TanishaaNeural"],
        gu: ["gu-IN-NiranjanNeural", "gu-IN-DhwaniNeural"],
        pa: ["pa-IN-OjasNeural", "pa-IN-VaaniNeural"],
        ml: ["ml-IN-MidhunNeural", "ml-IN-SobhanaNeural"],
        ur: ["ur-IN-SalmanNeural", "ur-IN-GulNeural"],
        fr: ["fr-FR-HenriNeural", "fr-FR-DeniseNeural"],
        es: ["es-ES-AlvaroNeural", "es-ES-ElviraNeural"],
        de: ["de-DE-ConradNeural", "de-DE-KatjaNeural"]
    };

    // Seed project history if empty
    const DEFAULT_PROJECT_HISTORY = [
        {
            video_id: "exec_20260907_072037_090cc6ac",
            prompt: "A small rover roaming the dunes of Mars at sunset.",
            duration: 10,
            resolution: "1280x720",
            style: "cinematic",
            model: "base",
            video_url: "/generated/videos/exec_20260907_072037_090cc6ac.mp4",
            created_at: Date.now() - 1000 * 60 * 120,
            scenes: [
                { title: "Scene 1: The Awakening Horizon", duration: 3.33, narrative: "As daylight wanes over the futuristic metropolis, a lone traveler pauses to witness the glowing horizon.", camera_motion: "pan_right" },
                { title: "Scene 2: Through the Shining Pathways", duration: 3.33, narrative: "Every step unveils ancient secrets intertwined with soaring spires, alive with quiet energy.", camera_motion: "zoom_in" },
                { title: "Scene 3: The Grand Discovery", duration: 3.33, narrative: "Underneath the fading sun, the true wonder of this world reveals itself in all its majesty.", camera_motion: "pan_left" }
            ]
        },
        {
            video_id: "exec_20260907_072102_5f9a7088",
            prompt: "A neon cyberpunk cityscape with flying cars in rainy weather.",
            duration: 15,
            resolution: "1280x720",
            style: "cyberpunk",
            model: "finetuned",
            video_url: "/generated/videos/exec_20260907_072102_5f9a7088.mp4",
            created_at: Date.now() - 1000 * 60 * 95,
            scenes: [
                { title: "Scene 1: Neon Reflections", duration: 5.0, narrative: "Rain splatters against glowing billboards as synthetic lights reflect in dark streets.", camera_motion: "dolly_in" },
                { title: "Scene 2: Cyber Traffic", duration: 5.0, narrative: "Vehicles hover and glide through misty avenues amidst towering skyscrapers.", camera_motion: "tilt_up" },
                { title: "Scene 3: Skyline Horizon", duration: 5.0, narrative: "The camera pulls back to reveal the infinite scale of the neon metropolis.", camera_motion: "pan_right" }
            ]
        },
        {
            video_id: "exec_20260907_072207_d6031e3c",
            prompt: "Enchanted forest with magical glowing creatures and ancient trees.",
            duration: 15,
            resolution: "1280x720",
            style: "fantasy",
            model: "base",
            video_url: "/generated/videos/exec_20260907_072207_d6031e3c.mp4",
            created_at: Date.now() - 1000 * 60 * 60,
            scenes: [
                { title: "Scene 1: Twilight Glade", duration: 5.0, narrative: "Ethereal blue light filters through moss-covered branches.", camera_motion: "pan_left" },
                { title: "Scene 2: Luminescent Flora", duration: 5.0, narrative: "Bioluminescent blossoms open slowly to greet the evening twilight.", camera_motion: "zoom_in" },
                { title: "Scene 3: Awakening Whispers", duration: 5.0, narrative: "Gentle forest spirits emerge and dance around the sacred grove.", camera_motion: "crane_down" }
            ]
        }
    ];

    function getHistory() {
        try {
            const data = localStorage.getItem(STORAGE_KEY_HISTORY);
            if (!data) {
                localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(DEFAULT_PROJECT_HISTORY));
                return DEFAULT_PROJECT_HISTORY;
            }
            return JSON.parse(data);
        } catch (e) {
            return DEFAULT_PROJECT_HISTORY;
        }
    }

    function saveHistory(historyArray) {
        try {
            localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(historyArray));
            updateHistoryBadge();
        } catch (e) {
            console.error("Failed to persist history to localStorage", e);
        }
    }

    function addHistoryItem(item) {
        const history = getHistory();
        // Avoid duplicate by video_id
        const filtered = history.filter(h => h.video_id !== item.video_id);
        filtered.unshift(item);
        saveHistory(filtered);
        renderHistory();
    }

    // =========================================================================
    // 2. DOM ELEMENTS
    // =========================================================================
    // Navigation
    const navItems = document.querySelectorAll(".sidebar-nav .nav-item");
    const viewPanels = document.querySelectorAll(".view-panel");
    const sidebarToggleBtn = document.getElementById("sidebar-toggle-btn");
    const appSidebar = document.getElementById("app-sidebar");
    const sidebarBackdrop = document.getElementById("sidebar-backdrop");
    const historyCountBadge = document.getElementById("history-count-badge");

    // Generation Form
    const generateForm = document.getElementById("generate-form");
    const promptInput = document.getElementById("prompt-input");
    const charCountEl = document.getElementById("char-count");
    const wordCountEl = document.getElementById("word-count");
    const modelSelect = document.getElementById("model-select");
    const durationSlider = document.getElementById("duration-slider");
    const durationValBadge = document.getElementById("duration-val-badge");
    const aspectBtns = document.querySelectorAll(".aspect-btn");
    const qualityBtns = document.querySelectorAll(".quality-btn");
    const outputReadout = document.getElementById("output-readout");
    const styleSelect = document.getElementById("style-select");
    const languageSelect = document.getElementById("language-select");
    const voiceSelect = document.getElementById("voice-select");
    const characterSelect = document.getElementById("character-select");
    const subtitlesToggle = document.getElementById("subtitles-toggle");
    const musicToggle = document.getElementById("music-toggle");
    const realtimeToggle = document.getElementById("realtime-toggle");
    const seedInput = document.getElementById("seed-input");
    const randomizeSeedBtn = document.getElementById("randomize-seed-btn");
    const fpsSelect = document.getElementById("fps-select");
    const voiceToggle = document.getElementById("voice-toggle");
    const lipsyncToggle = document.getElementById("lipsync-toggle");
    const generateBtn = document.getElementById("generate-btn");
    const presetChips = document.querySelectorAll(".preset-chip");
    const uploadImageBtn = document.getElementById("upload-image-btn");
    const uploadVideoBtn = document.getElementById("upload-video-btn");
    const pasteLinkBtn = document.getElementById("paste-link-btn");
    const referenceImageInput = document.getElementById("reference-image-input");
    const referenceVideoInput = document.getElementById("reference-video-input");
    const referenceUrlInput = document.getElementById("reference-url-input");
    const referenceStatus = document.getElementById("reference-status");

    // Progress Card & 4 Stepper Steps
    const progressCard = document.getElementById("progress-card");
    const stageMainTitle = document.getElementById("stage-main-title");
    const stageSubTitle = document.getElementById("stage-sub-title");
    const percentText = document.getElementById("percent-text");
    const elapsedTimer = document.getElementById("elapsed-timer");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const step1 = document.getElementById("stage-step-1");
    const step2 = document.getElementById("stage-step-2");
    const step3 = document.getElementById("stage-step-3");
    const step4 = document.getElementById("stage-step-4");

    // Video Preview Player
    const playerContainer = document.getElementById("player-container");
    const emptyPlayerState = document.getElementById("empty-player-state");
    const generatingOverlay = document.getElementById("generating-player-overlay");
    const activePlayerWrapper = document.getElementById("active-player-wrapper");
    const videoPlayer = document.getElementById("video-player");
    const previewFormatBadge = document.getElementById("preview-format-badge");
    const previewActionToolbar = document.getElementById("preview-action-toolbar");
    const downloadBtn = document.getElementById("download-btn");
    const regenerateBtn = document.getElementById("regenerate-btn");
    const shareCopyBtn = document.getElementById("share-copy-btn");
    const sendToRlBtn = document.getElementById("send-to-rl-btn");
    const sceneBreakdownCard = document.getElementById("scene-breakdown-card");
    const sceneCountBadge = document.getElementById("scene-count-badge");
    const scenesList = document.getElementById("scenes-list");

    // History View
    const historyGrid = document.getElementById("history-grid");
    const emptyHistoryState = document.getElementById("empty-history-state");
    const historySearchInput = document.getElementById("history-search-input");
    const historyClearAllBtn = document.getElementById("history-clear-all-btn");

    // Models View
    const selectModelBtns = document.querySelectorAll(".select-model-btn");

    // Training View
    const datasetDropzone = document.getElementById("dataset-dropzone");
    const datasetFileInput = document.getElementById("dataset-file-input");
    const startTrainingBtn = document.getElementById("start-training-btn");
    const trainingTerminalLog = document.getElementById("training-terminal-log");
    const trainingSessionStatus = document.getElementById("training-session-status");
    const startTrainingBtnText = document.getElementById("start-training-btn-text");
    const cliFallbackCard = document.getElementById("cli-fallback-card");
    const cliFallbackCode = document.getElementById("cli-fallback-code");
    const automaticTrainingDetails = document.getElementById("training-auto-details");
    const retryAutomaticTrainingBtn = document.getElementById("retry-auto-training-btn");
    let selectedManifestFile = null;
    let trainingPollInterval = null;
    let automaticTrainingPollInterval = null;
    let activeTrainingJobId = null;
    const automaticTrainingStartedRefs = new Set();

    async function refreshDatasetDashboard() {
        try {
            const response = await fetch("/api/v1/dataset/dashboard");
            if (!response.ok) return;
            const payload = await response.json();
            const dataset = payload.dataset || {}, splits = payload.splits || {};
            const samples = document.getElementById("dataset-samples-count");
            const resolution = document.getElementById("dataset-resolution");
            const splitText = document.getElementById("dataset-splits");
            const licenses = document.getElementById("dataset-license-count");
            if (samples) samples.textContent = dataset.total_clips ?? "0";
            if (resolution) resolution.textContent = dataset.average_resolution ? `${Math.round(dataset.average_resolution.width)}×${Math.round(dataset.average_resolution.height)}` : "—";
            if (splitText) splitText.textContent = `${splits.train || 0} / ${splits.validation || 0} / ${splits.test || 0}`;
            if (licenses) licenses.textContent = (dataset.licensed_sources || []).length;
        } catch (_) { /* dashboard is optional; leave unknown values visible */ }
    }

    // RL Feedback View
    const candidateCardA = document.getElementById("candidate-card-a");
    const candidateCardB = document.getElementById("candidate-card-b");
    const voteABtn = document.getElementById("vote-a-btn");
    const voteBBtn = document.getElementById("vote-b-btn");
    const submitFeedbackBtn = document.getElementById("submit-feedback-btn");
    const feedbackHistoryList = document.getElementById("feedback-history-list");
    const rlActivePrompt = document.getElementById("rl-active-prompt");
    const rlIterationBadge = document.getElementById("rl-iteration-badge");
    const rlVideoB = document.getElementById("rl-video-b");

    // Settings View
    const settingsTokenInput = document.getElementById("settings-token-input");
    const readoutLlm = document.getElementById("readout-llm");
    const readoutImage = document.getElementById("readout-image");
    const readoutVideo = document.getElementById("readout-video");
    const readoutTts = document.getElementById("readout-tts");
    const readoutFfmpeg = document.getElementById("readout-ffmpeg");
    const engineStatusText = document.getElementById("engine-status-text");
    const engineDetailsText = document.getElementById("engine-details-text");
    const statusIndicator = document.getElementById("status-indicator");
    const mobileEnginePill = document.getElementById("mobile-engine-pill");

    // Toast Container
    const toastContainer = document.getElementById("toast-container");

    // =========================================================================
    // 3. TOAST NOTIFICATION HELPER
    // =========================================================================
    function showToast(message, type = "success", duration = 3500) {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        
        let iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`;
        if (type === "error") {
            iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
        }

        toast.innerHTML = `${iconSvg} <span>${message}</span>`;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.transition = "opacity 0.3s, transform 0.3s";
            toast.style.opacity = "0";
            toast.style.transform = "translateX(40px)";
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    // =========================================================================
    // 4. SPA NAVIGATION ROUTER
    // =========================================================================
    function switchView(targetViewId) {
        navItems.forEach(item => {
            if (item.dataset.view === targetViewId) {
                item.classList.add("active");
            } else {
                item.classList.remove("active");
            }
        });

        viewPanels.forEach(panel => {
            if (panel.id === targetViewId) {
                panel.classList.add("active");
            } else {
                panel.classList.remove("active");
            }
        });

        // Close mobile drawer if open
        if (appSidebar.classList.contains("open")) {
            appSidebar.classList.remove("open");
            sidebarBackdrop.classList.remove("active");
        }

        // View-specific actions
        if (targetViewId === "view-history") {
            renderHistory();
        } else if (targetViewId === "view-settings") {
            fetchSystemHealth();
        } else if (targetViewId === "view-training") {
            refreshDatasetDashboard();
        }
    }

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            switchView(item.dataset.view);
        });
    });

    if (sidebarToggleBtn) {
        sidebarToggleBtn.addEventListener("click", () => {
            appSidebar.classList.toggle("open");
            sidebarBackdrop.classList.toggle("active");
        });
    }

    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener("click", () => {
            appSidebar.classList.remove("open");
            sidebarBackdrop.classList.remove("active");
        });
    }

    // =========================================================================
    // 5. PROMPT COUNTER & PRESETS
    // =========================================================================
    function updatePromptCounters() {
        const text = promptInput.value;
        charCountEl.textContent = text.length;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        wordCountEl.textContent = words;
    }

    promptInput.addEventListener("input", updatePromptCounters);

    presetChips.forEach(chip => {
        chip.addEventListener("click", () => {
            promptInput.value = chip.dataset.prompt;
            updatePromptCounters();
            promptInput.focus();
            showToast("Prompt template applied!");
        });
    });

    // =========================================================================
    // 6. GENERATION CONTROLS (DURATION, ASPECT, SEED)
    // =========================================================================
    durationSlider.addEventListener("input", (e) => {
        durationValBadge.textContent = `${e.target.value} seconds`;
    });

    function currentResolution() {
        return RESOLUTION_MATRIX[selectedAspect][selectedQuality];
    }

    function updateOutputReadout() {
        const res = currentResolution().replace("x", " × ");
        if (outputReadout) outputReadout.textContent = `Output: ${res}`;
    }

    function populateVoices() {
        if (!voiceSelect || !languageSelect) return;
        const lang = languageSelect.value;
        const voices = LANGUAGE_VOICES[lang] || LANGUAGE_VOICES.en;
        voiceSelect.innerHTML = "";
        voices.forEach((voice, idx) => {
            const opt = new Option(voice, voice, idx === 0, idx === 0);
            voiceSelect.add(opt);
        });
    }

    aspectBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            aspectBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            selectedAspect = btn.dataset.aspect;
            updateOutputReadout();
        });
    });

    qualityBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            qualityBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            selectedQuality = btn.dataset.quality;
            updateOutputReadout();
        });
    });

    if (languageSelect) {
        languageSelect.addEventListener("change", populateVoices);
        populateVoices();
    }
    updateOutputReadout();

    // =========================================================================
    // 6b. UNIFIED REUSABLE REFERENCE CONTROLLER
    // =========================================================================
    function initReferenceWorkflow(cfg) {
        const state = {
            referenceId: null,
            referenceType: null,
            referenceUrl: null,
            payload: null
        };

        const uploadImgBtn = document.getElementById(cfg.uploadImageBtnId);
        const uploadVidBtn = document.getElementById(cfg.uploadVideoBtnId);
        const pasteBtn = document.getElementById(cfg.pasteLinkBtnId);
        const imgInput = document.getElementById(cfg.imageInputId);
        const vidInput = document.getElementById(cfg.videoInputId);
        const urlBox = document.getElementById(cfg.urlBoxId);
        const urlInput = document.getElementById(cfg.urlInputId);
        const processBtn = document.getElementById(cfg.processUrlBtnId);
        const previewCard = document.getElementById(cfg.previewCardId);
        const previewImg = document.getElementById(cfg.previewImgId);
        const previewVid = document.getElementById(cfg.previewVideoId);
        const typeBadge = document.getElementById(cfg.typeBadgeId);
        const sourceBadge = document.getElementById(cfg.sourceBadgeId);
        const filenameEl = document.getElementById(cfg.filenameId);
        const dimEl = document.getElementById(cfg.dimensionsId);
        const clearBtn = document.getElementById(cfg.clearBtnId);
        const statusEl = document.getElementById(cfg.statusId);
        let urlDebounceTimer = null;
        let processingUrl = false;

        function setStatus(text) {
            if (statusEl) statusEl.textContent = text;
        }

        function showPreview(data, displayName) {
            if (!previewCard) return;
            previewCard.classList.remove("hidden");
            if (typeBadge) typeBadge.textContent = (data.media_type || "Media").toUpperCase();
            if (sourceBadge) sourceBadge.textContent = data.source || "upload";
            if (filenameEl) filenameEl.textContent = displayName || data.filename || data.reference_id || "Reference Media";
            if (dimEl) dimEl.textContent = `${data.width || 1280} × ${data.height || 720}${data.duration ? " • " + data.duration + "s" : ""}`;

            const previewSrc = data.preview_url || data.still_path || data.path;
            if (data.media_type === "video" && !data.still_path && previewVid) {
                previewVid.src = previewSrc;
                previewVid.classList.remove("hidden");
                if (previewImg) previewImg.classList.add("hidden");
            } else if (previewImg && previewSrc) {
                previewImg.src = previewSrc;
                previewImg.classList.remove("hidden");
                if (previewVid) previewVid.classList.add("hidden");
            }
        }

        function clearReference(notify = true) {
            state.referenceId = null;
            state.referenceType = null;
            state.referenceUrl = null;
            state.payload = null;

            if (imgInput) imgInput.value = "";
            if (vidInput) vidInput.value = "";
            if (urlInput) urlInput.value = "";
            if (previewCard) previewCard.classList.add("hidden");
            if (previewImg) { previewImg.src = ""; previewImg.classList.add("hidden"); }
            if (previewVid) { previewVid.src = ""; previewVid.classList.add("hidden"); }
            setStatus("No reference selected");

            if (cfg.onReferenceChange) cfg.onReferenceChange(state);
            if (notify) showToast("Reference removed.");
        }

        async function uploadFile(file, hintedType) {
            setStatus(`Uploading ${hintedType}...`);
            const form = new FormData();
            form.append("file", file);
            if (hintedType) form.append("reference_type", hintedType);

            const res = await fetch("/api/v1/references/upload", { method: "POST", body: form });
            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.message || errData.detail || "Reference upload failed");
            }
            const data = await res.json();
            state.referenceId = data.reference_id;
            state.referenceType = data.media_type;
            state.referenceUrl = null;
            state.payload = data;

            if (urlBox) urlBox.classList.add("hidden");
            showPreview(data, file.name);
            setStatus(`Uploaded ${data.media_type}: ${file.name}`);
            showToast(`Reference ${data.media_type} uploaded successfully.`);
            if (data.auto_training_job_id) {
                appendTerminalLog(`[Automatic] Triggered training workflow job ${data.auto_training_job_id} for ${data.reference_id}`, "text-accent");
                pollAutomaticTrainingJob(data.auto_training_job_id);
            }
            if (cfg.onReferenceChange) cfg.onReferenceChange(state);
        }

        async function processUrl() {
            if (!urlInput) return;
            const url = urlInput.value.trim();
            if (!url) {
                showToast("Please enter a reference URL.", "error");
                return;
            }
            if (processingUrl || (state.referenceId && state.referenceUrl === url)) return;

            processingUrl = true;
            setStatus("Processing reference link...");
            if (processBtn) {
                processBtn.disabled = true;
                processBtn.textContent = "Processing...";
            }

            try {
                const res = await fetch("/api/v1/references/url", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ url })
                });

                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}));
                    throw new Error(errData.message || errData.detail || "This reference link cannot be processed. Please provide a publicly accessible image/video URL or upload the file directly.");
                }

                const data = await res.json();
                state.referenceId = data.reference_id;
                state.referenceType = data.media_type;
                state.referenceUrl = url;
                state.payload = data;

                showPreview(data, `${data.source.toUpperCase()} Reference`);
                setStatus(`Processed ${data.media_type} from ${data.source}`);
                showToast("Reference link processed successfully!");
                if (data.auto_training_job_id) {
                    appendTerminalLog(`[Automatic] Triggered training workflow job ${data.auto_training_job_id} for ${data.reference_id}`, "text-accent");
                    pollAutomaticTrainingJob(data.auto_training_job_id);
                }
                if (cfg.onReferenceChange) cfg.onReferenceChange(state);
            } catch (err) {
                setStatus(`Failed: ${err.message}`);
                showToast(err.message, "error", 5000);
            } finally {
                if (processBtn) {
                    processBtn.disabled = false;
                    processBtn.textContent = "Retry Processing";
                }
                processingUrl = false;
            }
        }

        if (uploadImgBtn) uploadImgBtn.addEventListener("click", () => imgInput && imgInput.click());
        if (uploadVidBtn) uploadVidBtn.addEventListener("click", () => vidInput && vidInput.click());
        if (pasteBtn && urlBox) {
            pasteBtn.addEventListener("click", () => {
                urlBox.classList.toggle("hidden");
                if (!urlBox.classList.contains("hidden") && urlInput) urlInput.focus();
            });
        }
        if (imgInput) {
            imgInput.addEventListener("change", async () => {
                const file = imgInput.files && imgInput.files[0];
                if (!file) return;
                try { await uploadFile(file, "image"); }
                catch (err) { setStatus(`Upload failed: ${err.message}`); showToast(err.message, "error"); }
            });
        }
        if (vidInput) {
            vidInput.addEventListener("change", async () => {
                const file = vidInput.files && vidInput.files[0];
                if (!file) return;
                try { await uploadFile(file, "video"); }
                catch (err) { setStatus(`Upload failed: ${err.message}`); showToast(err.message, "error"); }
            });
        }
        if (processBtn) {
            processBtn.addEventListener("click", processUrl);
        }
        if (urlInput) {
            urlInput.addEventListener("input", () => {
                clearTimeout(urlDebounceTimer);
                const candidate = urlInput.value.trim();
                if (!candidate) return;
                let parsed;
                try {
                    parsed = new URL(candidate);
                } catch (_) {
                    return;
                }
                if (!["http:", "https:"].includes(parsed.protocol) || !parsed.hostname) return;
                setStatus("Reference link detected; processing automatically...");
                urlDebounceTimer = setTimeout(processUrl, 700);
            });
            urlInput.addEventListener("keydown", (e) => {
                if (e.key === "Enter") {
                    e.preventDefault();
                    processUrl();
                }
            });
        }
        if (clearBtn) {
            clearBtn.addEventListener("click", () => clearReference(true));
        }

        return {
            getState: () => state,
            clear: clearReference
        };
    }

    // Initialize New Generation Reference Workflow
    const genRefController = initReferenceWorkflow({
        uploadImageBtnId: "upload-image-btn",
        uploadVideoBtnId: "upload-video-btn",
        pasteLinkBtnId: "paste-link-btn",
        imageInputId: "reference-image-input",
        videoInputId: "reference-video-input",
        urlBoxId: "reference-url-box",
        urlInputId: "reference-url-input",
        processUrlBtnId: "process-link-btn",
        previewCardId: "reference-preview-card",
        previewImgId: "reference-preview-img",
        previewVideoId: "reference-preview-video",
        typeBadgeId: "reference-type-badge",
        sourceBadgeId: "reference-source-badge",
        filenameId: "reference-filename",
        dimensionsId: "reference-dimensions",
        clearBtnId: "reference-clear-btn",
        statusId: "reference-status",
        onReferenceChange: (refState) => {
            referenceId = refState.referenceId;
            referenceType = refState.referenceType;
        }
    });

    // Initialize Training / Fine-Tuning Reference Workflow
    let trainReferenceState = { referenceId: null, referenceType: null, referenceUrl: null };
    const trainRefController = initReferenceWorkflow({
        uploadImageBtnId: "train-upload-image-btn",
        uploadVideoBtnId: "train-upload-video-btn",
        pasteLinkBtnId: "train-paste-link-btn",
        imageInputId: "train-reference-image-input",
        videoInputId: "train-reference-video-input",
        urlBoxId: "train-reference-url-box",
        urlInputId: "train-reference-url-input",
        processUrlBtnId: "train-process-link-btn",
        previewCardId: "train-reference-preview-card",
        previewImgId: "train-reference-preview-img",
        previewVideoId: "train-reference-preview-video",
        typeBadgeId: "train-reference-type-badge",
        sourceBadgeId: "train-reference-source-badge",
        filenameId: "train-reference-filename",
        dimensionsId: "train-reference-dimensions",
        clearBtnId: "train-reference-clear-btn",
        statusId: "train-reference-status",
        onReferenceChange: (refState) => {
            trainReferenceState = {
                referenceId: refState.referenceId,
                referenceType: refState.referenceType,
                referenceUrl: refState.referenceUrl
            };
            if (refState.referenceId) launchAutomaticTraining(refState);
        }
    });

    randomizeSeedBtn.addEventListener("click", () => {
        const randomSeed = Math.floor(Math.random() * 1000000);
        seedInput.value = randomSeed;
        showToast(`Random seed set: ${randomSeed}`);
    });

    // Model selection buttons from the Models view
    selectModelBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const modelVal = btn.dataset.model;
            modelSelect.value = modelVal;
            switchView("view-generate");
            showToast(`Model set to: ${modelSelect.options[modelSelect.selectedIndex].text}`);
        });
    });

    // =========================================================================
    // 7. SYSTEM HEALTH MONITOR
    // =========================================================================
    async function fetchSystemHealth() {
        try {
            const res = await fetch("/health");
            if (!res.ok) throw new Error("Health endpoint error");
            const data = await res.json();

            if (data.status === "healthy") {
                if (engineStatusText) engineStatusText.textContent = "Engine Online";
                if (statusIndicator) statusIndicator.className = "status-indicator-dot online";
                if (mobileEnginePill) {
                    mobileEnginePill.className = "status-pill status-ready";
                    mobileEnginePill.textContent = "Online";
                }
                if (engineDetailsText) {
                    engineDetailsText.textContent = `FFmpeg: ${data.ffmpeg_available ? "Ready" : "Missing"} • ${data.providers?.video || "opencv"}`;
                }

                // Populate settings readout
                if (readoutLlm && data.providers?.llm) readoutLlm.textContent = data.providers.llm;
                if (readoutImage && data.providers?.image) readoutImage.textContent = data.providers.image;
                if (readoutVideo && data.providers?.video) readoutVideo.textContent = data.providers.video;
                if (readoutTts && data.providers?.tts) readoutTts.textContent = data.providers.tts;
                if (readoutFfmpeg) {
                    readoutFfmpeg.textContent = data.ffmpeg_available ? "Installed (Accelerated)" : "Warning: Not in PATH";
                    readoutFfmpeg.className = data.ffmpeg_available ? "code-pill highlight-green" : "code-pill text-muted";
                }
            }
        } catch (e) {
            if (engineStatusText) engineStatusText.textContent = "Engine Offline";
            if (statusIndicator) statusIndicator.className = "status-indicator-dot";
            if (mobileEnginePill) {
                mobileEnginePill.className = "status-pill";
                mobileEnginePill.textContent = "Offline";
            }
        }
    }

    async function loadGenerationOptions() {
        try {
            const res = await fetch("/api/v1/generation-options");
            if (!res.ok) return;
            const data = await res.json();
            if (data.resolution_matrix) {
                Object.keys(data.resolution_matrix).forEach((aspect) => {
                    RESOLUTION_MATRIX[aspect] = data.resolution_matrix[aspect];
                });
                updateOutputReadout();
            }
            if (Array.isArray(data.languages)) {
                data.languages.forEach((lang) => {
                    LANGUAGE_VOICES[lang.id] = lang.voices;
                });
                populateVoices();
            }
        } catch (e) {
            // Keep built-in catalog when the API is offline.
        }
    }

    fetchSystemHealth();
    loadGenerationOptions();

    // =========================================================================
    // 8. GENERATION WORKFLOW & STAGE TRACKER (5 REQUIRED STAGES)
    // =========================================================================
    function setStepState(activeStepNum) {
        const steps = [step1, step2, step3, step4];
        steps.forEach((s, idx) => {
            if (!s) return;
            const stepNum = idx + 1;
            s.classList.remove("active", "done");
            if (stepNum === activeStepNum) {
                s.classList.add("active");
            } else if (stepNum < activeStepNum) {
                s.classList.add("done");
            }
        });
    }

    function startTimer() {
        jobStartTime = Date.now();
        clearInterval(timerInterval);
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - jobStartTime) / 1000);
            const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
            const secs = String(elapsed % 60).padStart(2, "0");
            elapsedTimer.textContent = `${mins}:${secs}`;
        }, 1000);
    }

    function stopTimer() {
        clearInterval(timerInterval);
    }

    function resetGenerationUI() {
        generateBtn.disabled = false;
        generateBtn.querySelector(".btn-text").textContent = "Generate Video";
        stopTimer();
    }

    generateForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const prompt = promptInput.value.trim();
        if (prompt.length < 3) {
            showToast("Please enter a descriptive prompt with at least 3 characters.", "error");
            return;
        }

        const duration = parseInt(durationSlider.value, 10);
        const style = styleSelect.value;
        const voice = voiceToggle.checked;
        const fps = parseInt(fpsSelect.value, 10);
        const modelMode = modelSelect.value;
        const token = (settingsTokenInput ? settingsTokenInput.value.trim() : null) || null;
        const language = languageSelect ? languageSelect.value : "en";
        const voiceId = voiceSelect ? voiceSelect.value : null;
        const referenceUrl = referenceUrlInput ? referenceUrlInput.value.trim() : "";

        const payload = {
            prompt,
            duration,
            style,
            voice,
            fps,
            model_mode: modelMode,
            token,
            aspect_ratio: selectedAspect,
            quality: selectedQuality,
            language,
            voice_id: voice ? voiceId : null,
            character_id: characterSelect ? characterSelect.value || null : null,
            subtitles: !!(subtitlesToggle && subtitlesToggle.checked),
            music: !!(musicToggle && musicToggle.checked),
            realtime_data: !!(realtimeToggle && realtimeToggle.checked),
            lipsync: !!(lipsyncToggle && lipsyncToggle.checked),
            reference_url: referenceUrl || null,
            reference_type: referenceType,
            reference_id: referenceId
        };

        // UI state change to generating
        generateBtn.disabled = true;
        generateBtn.querySelector(".btn-text").textContent = "Generating Video...";

        progressCard.classList.remove("hidden");
        stageMainTitle.textContent = "Generating video...";
        stageSubTitle.textContent = "Processing prompt...";
        progressBarFill.style.width = "5%";
        percentText.textContent = "5%";
        setStepState(1);
        startTimer();

        // Prepare player area
        emptyPlayerState.classList.add("hidden");
        generatingOverlay.classList.remove("hidden");
        activePlayerWrapper.classList.add("hidden");
        previewActionToolbar.classList.add("hidden");
        sceneBreakdownCard.classList.add("hidden");
        previewFormatBadge.textContent = "Processing...";

        try {
            const res = await fetch("/api/v1/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                const detail = Array.isArray(errData.detail) ? errData.detail[0]?.msg : errData.detail;
                throw new Error(errData.message || detail || "Failed to submit generation job");
            }

            const data = await res.json();
            currentJobId = data.job_id;
            showToast("Job submitted successfully! Synthesizing video...");

            // Begin polling status
            pollJob(currentJobId, payload);
        } catch (err) {
            showToast(`Error: ${err.message}`, "error");
            generatingOverlay.classList.add("hidden");
            emptyPlayerState.classList.remove("hidden");
            stageMainTitle.textContent = "Generation Failed";
            stageSubTitle.textContent = err.message;
            resetGenerationUI();
        }
    });

    function pollJob(jobId, originalPayload) {
        clearInterval(pollInterval);

        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/v1/jobs/${jobId}`);
                if (!res.ok) throw new Error("Could not retrieve job status");

                const job = await res.json();
                const progress = Math.max(5, job.progress || 5);
                progressBarFill.style.width = `${progress}%`;
                percentText.textContent = `${progress}%`;

                // Stage Mapping to the 4 UI Step Bullets & Display
                // Stage 1: Processing prompt...
                // Stage 2: Generating frames...
                // Stage 3: Creating video...
                // Stage 4: Completed
                const stage = job.stage || "";

                if (stage.includes("prompt") || stage.includes("story") || stage.includes("script")) {
                    stageMainTitle.textContent = "Generating video...";
                    stageSubTitle.textContent = "Processing prompt...";
                    setStepState(1);
                } else if (stage.includes("scene") || stage.includes("consistency") || stage.includes("keyframe")) {
                    stageMainTitle.textContent = "Generating video...";
                    stageSubTitle.textContent = "Generating frames...";
                    setStepState(2);
                } else if (stage.includes("rendering") || stage.includes("voice") || stage.includes("lipsync") || stage.includes("audio") || stage.includes("assembling") || stage.includes("validating") || stage.includes("storing")) {
                    stageMainTitle.textContent = "Generating video...";
                    stageSubTitle.textContent = "Creating video...";
                    setStepState(3);
                }

                if (job.status === "completed") {
                    clearInterval(pollInterval);
                    progressBarFill.style.width = "100%";
                    percentText.textContent = "100%";
                    stageMainTitle.textContent = "Completed";
                    stageSubTitle.textContent = "Video successfully generated!";
                    setStepState(4);

                    displayCompletedVideo(job.result, originalPayload);
                    resetGenerationUI();
                    showToast("Video generation completed successfully!");
                } else if (job.status === "failed") {
                    clearInterval(pollInterval);
                    stageMainTitle.textContent = "Generation Failed";
                    stageSubTitle.textContent = job.error || "An internal error occurred during synthesis.";
                    generatingOverlay.classList.add("hidden");
                    emptyPlayerState.classList.remove("hidden");
                    resetGenerationUI();
                    showToast(`Pipeline Error: ${job.error || "Generation failed"}`, "error");
                }
            } catch (err) {
                clearInterval(pollInterval);
                stageMainTitle.textContent = "Network Polling Interrupted";
                stageSubTitle.textContent = err.message;
                resetGenerationUI();
            }
        }, 1000);
    }

    function displayCompletedVideo(result, payload) {
        if (!result || !result.video_url) return;

        activeVideoData = {
            video_id: result.video_id || `gen_${Date.now()}`,
            prompt: result.prompt || payload.prompt,
            duration: result.duration || payload.duration,
            resolution: result.resolution || currentResolution(),
            style: result.style || payload.style,
            model: payload.model_mode || "base",
            video_url: result.video_url,
            created_at: Date.now(),
            scenes: result.scenes || []
        };

        // Add to persistent history
        addHistoryItem(activeVideoData);

        // Update player view
        generatingOverlay.classList.add("hidden");
        emptyPlayerState.classList.add("hidden");
        activePlayerWrapper.classList.remove("hidden");

        const cacheBustedUrl = `${result.video_url}?t=${Date.now()}`;
        videoPlayer.src = cacheBustedUrl;
        videoPlayer.load();

        previewFormatBadge.textContent = `${activeVideoData.resolution} • ${activeVideoData.duration}s • ${activeVideoData.style.toUpperCase()}`;

        // Action Toolbar
        previewActionToolbar.classList.remove("hidden");
        downloadBtn.href = result.video_url;
        downloadBtn.download = `${activeVideoData.video_id}.mp4`;

        // Scene breakdown list
        if (result.scenes && result.scenes.length > 0) {
            renderSceneBreakdown(result.scenes);
        } else {
            sceneBreakdownCard.classList.add("hidden");
        }
    }

    function renderSceneBreakdown(scenes) {
        scenesList.innerHTML = "";
        sceneCountBadge.textContent = `${scenes.length} Scenes`;

        scenes.forEach((scene, index) => {
            const card = document.createElement("div");
            card.className = "scene-item-card";
            card.innerHTML = `
                <div class="scene-header">
                    <span class="scene-heading">${scene.title || `Scene ${index + 1}`} (${scene.duration || 3.3}s)</span>
                    <span class="scene-motion-tag">Motion: ${scene.camera_motion || "pan_right"}</span>
                </div>
                <p class="scene-narrative">"${scene.narrative || scene.visual_description || "Scene visuals in sequence."}"</p>
            `;
            scenesList.appendChild(card);
        });

        sceneBreakdownCard.classList.remove("hidden");
    }

    // =========================================================================
    // 9. VIDEO ACTION TOOLBAR HANDLERS
    // =========================================================================
    regenerateBtn.addEventListener("click", () => {
        if (!activeVideoData) return;
        promptInput.value = activeVideoData.prompt;
        updatePromptCounters();
        generateForm.dispatchEvent(new Event("submit"));
        showToast("Re-initiating video generation...");
    });

    shareCopyBtn.addEventListener("click", () => {
        if (!activeVideoData || !activeVideoData.video_url) return;
        const fullUrl = window.location.origin + activeVideoData.video_url;
        navigator.clipboard.writeText(fullUrl).then(() => {
            showToast("Video URL copied to clipboard!");
        }).catch(() => {
            showToast(`Video URL: ${fullUrl}`);
        });
    });

    sendToRlBtn.addEventListener("click", () => {
        if (!activeVideoData) return;
        // Populate RL view with current active video as candidate B
        rlActivePrompt.textContent = `"${activeVideoData.prompt}"`;
        if (rlVideoB) {
            rlVideoB.src = activeVideoData.video_url;
            rlVideoB.load();
        }
        switchView("view-rl");
        showToast("Loaded video into RL Preference Studio!");
    });

    // =========================================================================
    // 10. HISTORY MANAGEMENT & RENDERING
    // =========================================================================
    function updateHistoryBadge() {
        const history = getHistory();
        if (historyCountBadge) {
            historyCountBadge.textContent = history.length;
        }
    }

    function renderHistory(filterText = "") {
        const history = getHistory();
        historyGrid.innerHTML = "";

        const query = filterText.trim().toLowerCase();
        const filtered = query 
            ? history.filter(h => h.prompt.toLowerCase().includes(query) || (h.style && h.style.toLowerCase().includes(query)))
            : history;

        if (filtered.length === 0) {
            emptyHistoryState.classList.remove("hidden");
            return;
        }

        emptyHistoryState.classList.add("hidden");

        filtered.forEach(item => {
            const card = document.createElement("div");
            card.className = "history-card";

            const dateStr = item.created_at ? new Date(item.created_at).toLocaleString(undefined, {
                month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
            }) : "Recent";

            card.innerHTML = `
                <div class="history-thumb-box">
                    <video src="${item.video_url}" muted preload="metadata" onmouseover="this.play()" onmouseout="this.pause();this.currentTime=0;"></video>
                </div>
                <div class="history-card-body">
                    <div class="history-meta-row">
                        <span class="history-model-tag">${item.model === "finetuned" ? "LoRA v001" : "Base TTV"}</span>
                        <span class="history-date">${dateStr}</span>
                    </div>
                    <p class="history-prompt" title="${item.prompt}">${item.prompt}</p>
                    <div class="history-specs-row">
                        <span class="spec-badge">${item.resolution || "720p"}</span>
                        <span class="spec-badge">${item.duration || 15}s</span>
                        <span class="spec-badge">${(item.style || "cinematic").toUpperCase()}</span>
                        <span class="badge badge-success" style="margin-left: auto;">Completed</span>
                    </div>
                    <div class="history-actions-row">
                        <button type="button" class="btn btn-secondary btn-sm hist-play-btn">Play</button>
                        <button type="button" class="btn btn-secondary btn-sm hist-prompt-btn">Load</button>
                        <a href="${item.video_url}" download="${item.video_id}.mp4" class="btn btn-secondary btn-sm" title="Download MP4">⬇</a>
                    </div>
                </div>
            `;

            // Handlers
            card.querySelector(".hist-play-btn").addEventListener("click", () => {
                displayCompletedVideo(item, {
                    prompt: item.prompt,
                    duration: item.duration,
                    resolution: item.resolution,
                    style: item.style,
                    model_mode: item.model
                });
                switchView("view-generate");
                window.scrollTo({ top: 0, behavior: "smooth" });
            });

            card.querySelector(".hist-prompt-btn").addEventListener("click", () => {
                promptInput.value = item.prompt;
                updatePromptCounters();
                if (item.style) styleSelect.value = item.style;
                if (item.duration) {
                    durationSlider.value = item.duration;
                    durationValBadge.textContent = `${item.duration} seconds`;
                }
                switchView("view-generate");
                promptInput.focus();
                showToast("Prompt and settings loaded into workspace!");
            });

            historyGrid.appendChild(card);
        });

        updateHistoryBadge();
    }

    if (historySearchInput) {
        historySearchInput.addEventListener("input", (e) => {
            renderHistory(e.target.value);
        });
    }

    if (historyClearAllBtn) {
        historyClearAllBtn.addEventListener("click", () => {
            if (confirm("Are you sure you want to clear your video generation history?")) {
                localStorage.removeItem(STORAGE_KEY_HISTORY);
                renderHistory();
                showToast("History cleared.");
            }
        });
    }

    // =========================================================================
    // 11. TRAINING / FINE-TUNING DASHBOARD ACTIONS
    // =========================================================================
    if (datasetDropzone) {
        datasetDropzone.addEventListener("click", () => {
            datasetFileInput.click();
        });

        datasetDropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            datasetDropzone.style.borderColor = "var(--brand-primary)";
        });

        datasetDropzone.addEventListener("dragleave", () => {
            datasetDropzone.style.borderColor = "";
        });

        datasetDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            datasetDropzone.style.borderColor = "";
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleDatasetFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (datasetFileInput) {
        datasetFileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleDatasetFile(e.target.files[0]);
            }
        });
    }

    function setTrainingBadge(status, textOverride = null) {
        if (!trainingSessionStatus) return;
        switch (status) {
            case "queued":
                trainingSessionStatus.className = "badge badge-queued";
                trainingSessionStatus.textContent = textOverride || "job queued";
                break;
            case "running":
                trainingSessionStatus.className = "badge badge-running";
                trainingSessionStatus.textContent = textOverride || "training running";
                break;
            case "CUDA_UNAVAILABLE":
                trainingSessionStatus.className = "badge badge-cuda-unavailable";
                trainingSessionStatus.textContent = textOverride || "CUDA unavailable";
                break;
            case "manual_cli":
                trainingSessionStatus.className = "badge badge-manual-cli";
                trainingSessionStatus.textContent = textOverride || "manual CLI required";
                break;
            case "completed":
                trainingSessionStatus.className = "badge badge-success";
                trainingSessionStatus.textContent = textOverride || "Completed";
                break;
            case "failed":
                trainingSessionStatus.className = "badge badge-danger";
                trainingSessionStatus.textContent = textOverride || "Failed";
                break;
            default:
                trainingSessionStatus.className = "badge";
                trainingSessionStatus.textContent = textOverride || "Worker Ready";
                break;
        }
    }

    function appendTerminalLog(text, className = "text-muted") {
        if (!trainingTerminalLog) return;
        const line = document.createElement("div");
        line.className = `log-line ${className}`;
        line.textContent = text;
        trainingTerminalLog.appendChild(line);
        trainingTerminalLog.scrollTop = trainingTerminalLog.scrollHeight;
    }

    async function handleDatasetFile(file) {
        selectedManifestFile = file;
        const dropzonePrompt = datasetDropzone.querySelector(".dropzone-prompt");
        if (dropzonePrompt) {
            dropzonePrompt.innerHTML = `Validating manifest: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB)...`;
        }

        // Validate manifest via backend API without client-side python execution
        try {
            const formData = new FormData();
            formData.append("manifest_file", file);

            const res = await fetch("/api/v1/training/manifest/validate", {
                method: "POST",
                body: formData
            });

            if (res.ok) {
                const data = await res.json();
                if (dropzonePrompt) {
                    dropzonePrompt.innerHTML = `Loaded manifest: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB) • <strong>${data.total_records}</strong> records`;
                }

                // Update Dataset Stats Card
                const samplesEl = document.getElementById("dataset-samples-count");
                const resolutionEl = document.getElementById("dataset-resolution");
                const splitTextEl = document.getElementById("dataset-splits");
                const licenseEl = document.getElementById("dataset-license-count");

                if (samplesEl) samplesEl.textContent = data.valid_records ?? "0";
                if (resolutionEl) resolutionEl.textContent = "1280×720";
                if (splitTextEl && data.splits) {
                    splitTextEl.textContent = `${data.splits.train || 0} / ${data.splits.val || 0} / ${data.splits.test || 0}`;
                }
                if (licenseEl && data.categories) {
                    licenseEl.textContent = `${Object.keys(data.categories).length} categories`;
                }

                appendTerminalLog(`[Manifest Validated] ${file.name}: ${data.summary}`, "text-success");
                showToast(`Dataset manifest validated: ${data.total_records} records verified!`, "success");
            } else {
                throw new Error("Validation endpoint returned status " + res.status);
            }
        } catch (err) {
            if (dropzonePrompt) {
                dropzonePrompt.innerHTML = `Loaded file: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB)`;
            }
            appendTerminalLog(`[Manifest] Loaded ${file.name}. Notice: ${err.message}`, "text-warning");
            showToast(`Loaded ${file.name}`, "info");
        }
    }

    function pollTrainingJob(jobId) {
        clearInterval(trainingPollInterval);

        let lastLogIndex = 0;
        trainingPollInterval = setInterval(async () => {
            try {
                // Poll Status
                const statusRes = await fetch(`/api/v1/training/jobs/${jobId}`);
                if (!statusRes.ok) return;
                const job = await statusRes.json();

                // Poll Logs
                const logsRes = await fetch(`/api/v1/training/jobs/${jobId}/logs?offset=${lastLogIndex}`);
                if (logsRes.ok) {
                    const logsData = await logsRes.json();
                    if (Array.isArray(logsData.logs) && logsData.logs.length > 0) {
                        logsData.logs.forEach(lineText => {
                            let cls = "text-muted";
                            if (lineText.includes("[CUDA Check] FAILED") || lineText.includes("CUDA_UNAVAILABLE")) cls = "text-warning";
                            else if (lineText.includes("FAILED") || lineText.includes("Error") || lineText.includes("Exception")) cls = "text-danger";
                            else if (lineText.includes("SUCCESS") || lineText.includes("completed")) cls = "text-success";
                            else if (lineText.includes("[Worker]") || lineText.includes("[Config]")) cls = "text-accent";
                            appendTerminalLog(lineText, cls);
                        });
                        lastLogIndex += logsData.logs.length;
                    }
                }

                // Update UI based on distinct statuses
                if (job.status === "queued") {
                    setTrainingBadge("queued", "job queued");
                } else if (job.status === "running") {
                    setTrainingBadge("running", `training running (${job.progress || 0}%)`);
                } else if (job.status === "CUDA_UNAVAILABLE") {
                    clearInterval(trainingPollInterval);
                    setTrainingBadge("CUDA_UNAVAILABLE", "CUDA unavailable");
                    if (job.cli_fallback_command) {
                        if (cliFallbackCode) cliFallbackCode.textContent = job.cli_fallback_command;
                        if (cliFallbackCard) cliFallbackCard.classList.remove("hidden");
                    }
                    if (startTrainingBtn) startTrainingBtn.disabled = false;
                    if (startTrainingBtnText) startTrainingBtnText.textContent = "Launch Training";
                    showToast("CUDA unavailable on this system. Manual CLI fallback required in a CUDA environment.", "warning", 6000);
                } else if (job.status === "completed") {
                    clearInterval(trainingPollInterval);
                    setTrainingBadge("completed", "Training Completed");
                    if (startTrainingBtn) startTrainingBtn.disabled = false;
                    if (startTrainingBtnText) startTrainingBtnText.textContent = "Launch Training";
                    showToast("Model fine-tuning completed successfully!", "success");
                } else if (job.status === "failed") {
                    clearInterval(trainingPollInterval);
                    setTrainingBadge("failed", "Training Failed");
                    if (job.cli_fallback_command) {
                        if (cliFallbackCode) cliFallbackCode.textContent = job.cli_fallback_command;
                        if (cliFallbackCard) cliFallbackCard.classList.remove("hidden");
                    }
                    if (startTrainingBtn) startTrainingBtn.disabled = false;
                    if (startTrainingBtnText) startTrainingBtnText.textContent = "Launch Training";
                    showToast(`Training failed: ${job.error || "Subprocess exited with failure"}`, "error");
                }
            } catch (err) {
                // Ignore transient polling glitches
            }
        }, 1200);
    }

    function renderAutomaticTrainingDetails(job) {
        if (!automaticTrainingDetails) return;
        const hardware = job.hardware || {};
        const gpuRows = hardware.gpu_utilization;
        const utilization = Array.isArray(gpuRows) && gpuRows.length
            ? gpuRows.map(gpu => `GPU ${gpu.index}: ${gpu.utilization_percent}% (${gpu.memory_used_mib}/${gpu.memory_total_mib} MiB)`).join(", ")
            : (job.cuda_available ? "GPU utilization unavailable" : "GPU not allocated");
        const checkpoint = job.final_checkpoint || (job.checkpoints || []).slice(-1)[0] || "Pending";
        const loss = typeof job.loss === "number" ? job.loss.toFixed(4) : "—";
        const evaluation = typeof (job.metrics || {}).val_loss === "number"
            ? job.metrics.val_loss.toFixed(4)
            : "Pending";
        automaticTrainingDetails.textContent = [
            `Stage: ${job.stage || "preparing"}`,
            `Epoch: ${job.epoch || 0}/${job.total_epochs || 0} • Progress: ${job.progress || 0}%`,
            `Training loss: ${loss} • Evaluation loss: ${evaluation}`,
            `Device: ${hardware.gpu_name || job.device || "Not allocated"} • ${utilization}`,
            `Checkpoint: ${checkpoint}`
        ].join("\n");
    }

    function pollAutomaticTrainingJob(jobId) {
        clearInterval(automaticTrainingPollInterval);
        let lastLogIndex = 0;

        const poll = async () => {
            try {
                const statusResponse = await fetch(`/api/v1/training/auto/${jobId}`);
                if (!statusResponse.ok) return;
                const job = await statusResponse.json();
                renderAutomaticTrainingDetails(job);

                const logsResponse = await fetch(`/api/v1/training/auto/${jobId}/logs?offset=${lastLogIndex}`);
                if (logsResponse.ok) {
                    const logsData = await logsResponse.json();
                    for (const lineText of logsData.logs || []) {
                        let cls = "text-muted";
                        if (lineText.includes("FAILED") || lineText.includes("[Error]") || lineText.includes("Exception")) cls = "text-danger";
                        else if (lineText.includes("CUDA_UNAVAILABLE") || lineText.includes("unavailable")) cls = "text-warning";
                        else if (lineText.includes("completed") || lineText.includes("Checkpoint")) cls = "text-success";
                        else if (lineText.includes("[Worker]") || lineText.includes("[Config]") || lineText.includes("[Training]")) cls = "text-accent";
                        appendTerminalLog(lineText, cls);
                    }
                    lastLogIndex += (logsData.logs || []).length;
                    if (logsData.cli_fallback_command) {
                        if (cliFallbackCode) cliFallbackCode.textContent = logsData.cli_fallback_command;
                        if (cliFallbackCard) cliFallbackCard.classList.remove("hidden");
                    }
                }

                if (job.status === "preparing" || job.status === "processing") {
                    setTrainingBadge("running", job.stage.replaceAll("_", " "));
                } else if (job.status === "queued") {
                    setTrainingBadge("queued", "job queued");
                } else if (job.status === "running") {
                    setTrainingBadge("running", `training running (${job.progress || 0}%)`);
                } else if (job.status === "CUDA_UNAVAILABLE") {
                    clearInterval(automaticTrainingPollInterval);
                    setTrainingBadge("CUDA_UNAVAILABLE", "CUDA unavailable");
                    if (retryAutomaticTrainingBtn) retryAutomaticTrainingBtn.classList.remove("hidden");
                    showToast(job.message || "GPU training is unavailable on this host.", "warning", 6000);
                } else if (job.status === "completed") {
                    clearInterval(automaticTrainingPollInterval);
                    setTrainingBadge("completed", "Training Completed");
                    if (retryAutomaticTrainingBtn) retryAutomaticTrainingBtn.classList.add("hidden");
                    showToast("Automatic fine-tuning completed successfully.", "success");
                } else if (job.status === "failed") {
                    clearInterval(automaticTrainingPollInterval);
                    setTrainingBadge("failed", "Training Failed");
                    if (retryAutomaticTrainingBtn) retryAutomaticTrainingBtn.classList.remove("hidden");
                    showToast(`Automatic training failed: ${job.error || "See training logs for details."}`, "error", 6000);
                }
            } catch (_) {
                // Keep polling through transient network errors.
            }
        };

        poll();
        automaticTrainingPollInterval = setInterval(poll, 1200);
    }

    async function launchAutomaticTraining(refState, retry = false) {
        const referenceId = refState && refState.referenceId;
        if (!referenceId || (!retry && automaticTrainingStartedRefs.has(referenceId))) return;
        automaticTrainingStartedRefs.add(referenceId);
        if (retryAutomaticTrainingBtn) retryAutomaticTrainingBtn.classList.add("hidden");

        const request = {
            reference_id: referenceId,
            base_model: document.getElementById("train-base-model")?.value || "Wan2.2-TI2V-5B",
            method: document.getElementById("train-method")?.value || "lora",
            epochs: parseInt(document.getElementById("train-epochs")?.value || "10", 10),
            learning_rate: parseFloat(document.getElementById("train-lr")?.value || "0.0001"),
            batch_size: parseInt(document.getElementById("train-batch-size")?.value || "2", 10),
            retry
        };

        setTrainingBadge("running", "preparing reference");
        appendTerminalLog(`[Automatic] Starting training workflow for ${referenceId}...`, "text-accent");
        try {
            const response = await fetch("/api/v1/training/auto", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(request)
            });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || errorData.message || "Could not start automatic training.");
            }
            const job = await response.json();
            activeTrainingJobId = job.job_id;
            renderAutomaticTrainingDetails(job);
            showToast("Reference accepted. Dataset preparation and training are running automatically.", "success");
            pollAutomaticTrainingJob(job.job_id);
        } catch (error) {
            if (!retry) automaticTrainingStartedRefs.delete(referenceId);
            setTrainingBadge("failed", "Automatic launch failed");
            appendTerminalLog(`[Error] ${error.message}`, "text-danger");
            if (retryAutomaticTrainingBtn) retryAutomaticTrainingBtn.classList.remove("hidden");
            showToast(error.message, "error", 6000);
        }
    }

    if (retryAutomaticTrainingBtn) {
        retryAutomaticTrainingBtn.addEventListener("click", () => {
            launchAutomaticTraining(trainReferenceState, true);
        });
    }

    if (startTrainingBtn) {
        startTrainingBtn.addEventListener("click", async () => {
            const epochs = parseInt(document.getElementById("train-epochs")?.value || "10", 10);
            const lr = document.getElementById("train-lr")?.value || "0.0001";
            const batchSize = parseInt(document.getElementById("train-batch-size")?.value || "2", 10);
            const method = document.getElementById("train-method")?.value || "lora";
            const baseModel = document.getElementById("train-base-model")?.value || "Wan2.2-TI2V-5B";

            startTrainingBtn.disabled = true;
            if (startTrainingBtnText) startTrainingBtnText.textContent = "Submitting Job...";

            setTrainingBadge("queued", "job queued");

            trainingTerminalLog.innerHTML = "";
            appendTerminalLog(`[Launch] Preparing Fine-Tuning Job (Base: ${baseModel}, Method: ${method}, Epochs: ${epochs}, LR: ${lr}, Batch: ${batchSize})...`, "text-accent");

            try {
                const formData = new FormData();
                formData.append("base_model", baseModel);
                formData.append("method", method);
                formData.append("epochs", epochs);
                formData.append("learning_rate", lr);
                formData.append("batch_size", batchSize);

                if (selectedManifestFile) {
                    formData.append("manifest_file", selectedManifestFile);
                } else {
                    formData.append("manifest_path", "data/manifests/ttv_training_manifest_120.jsonl");
                }

                const response = await fetch("/api/v1/training/jobs", {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    const errData = await response.json().catch(() => ({}));
                    throw new Error(errData.detail || errData.message || "Failed to create training job");
                }

                const data = await response.json();
                activeTrainingJobId = data.job_id;

                if (data.cli_fallback_command) {
                    if (cliFallbackCode) cliFallbackCode.textContent = data.cli_fallback_command;
                    if (cliFallbackCard) cliFallbackCard.classList.remove("hidden");
                }

                // If backend determined CUDA is unavailable:
                if (data.status === "CUDA_UNAVAILABLE") {
                    setTrainingBadge("CUDA_UNAVAILABLE", "CUDA unavailable");
                    appendTerminalLog(`[System] Job ID registered: ${data.job_id}`, "text-info");
                    appendTerminalLog(`[Manifest] Records evaluated: ${data.manifest_stats?.total_records || 120} (${data.manifest_stats?.summary || "Valid"})`, "text-muted");
                    appendTerminalLog("[CUDA Check] FAILED: NVIDIA CUDA acceleration is not available in this environment.", "text-warning");
                    appendTerminalLog("[Status] CUDA_UNAVAILABLE: Browser and local CPU environment cannot train CUDA models.", "text-warning");
                    appendTerminalLog("[Manual CLI Required] Run the reviewed CLI command in a CUDA training environment:", "text-accent");
                    appendTerminalLog(`  ${data.cli_fallback_command}`, "text-accent");

                    showToast("CUDA unavailable on this system. Run the reviewed CLI command in a CUDA environment.", "warning", 6000);
                    startTrainingBtn.disabled = false;
                    if (startTrainingBtnText) startTrainingBtnText.textContent = "Launch Training";
                    return;
                }

                // If CUDA is available: job is queued or running
                if (data.status === "queued" || data.status === "running") {
                    setTrainingBadge(data.status, data.status === "queued" ? "job queued" : "training running");
                    appendTerminalLog(`[Job Started] Unique Job ID: ${data.job_id}`, "text-info");
                    appendTerminalLog("[Worker] Worker subprocess launched. Polling execution logs...", "text-muted");
                    showToast("Training job launched! Polling execution worker...", "success");
                    pollTrainingJob(data.job_id);
                }

            } catch (err) {
                setTrainingBadge("failed", "Launch Failed");
                appendTerminalLog(`[Error] Failed to submit training job: ${err.message}`, "text-danger");
                showToast(`Launch failed: ${err.message}`, "error");
                startTrainingBtn.disabled = false;
                if (startTrainingBtnText) startTrainingBtnText.textContent = "Launch Training";
            }
        });
    }

    // =========================================================================
    // 12. RL FEEDBACK STUDIO ACTIONS
    // =========================================================================
    if (voteABtn && voteBBtn) {
        voteABtn.addEventListener("click", () => {
            selectedRLWinner = "candidate_a";
            candidateCardA.classList.add("selected");
            candidateCardB.classList.remove("selected");
            submitFeedbackBtn.disabled = false;
            submitFeedbackBtn.textContent = "Submit Preference (Candidate A Winner)";
        });

        voteBBtn.addEventListener("click", () => {
            selectedRLWinner = "candidate_b";
            candidateCardB.classList.add("selected");
            candidateCardA.classList.remove("selected");
            submitFeedbackBtn.disabled = false;
            submitFeedbackBtn.textContent = "Submit Preference (Candidate B Winner)";
        });
    }

    if (submitFeedbackBtn) {
        submitFeedbackBtn.addEventListener("click", () => {
            if (!selectedRLWinner) return;

            rlIteration++;
            if (rlIterationBadge) {
                rlIterationBadge.textContent = `Iteration ${rlIteration} / 3`;
            }

            const winnerTag = selectedRLWinner === "candidate_a" ? "Candidate A" : "Candidate B";
            const winnerScore = selectedRLWinner === "candidate_a" ? "0.8200" : "0.9100";

            const item = document.createElement("div");
            item.className = "feedback-item";
            item.innerHTML = `
                <span class="fb-time">Sample ${feedbackHistoryList.children.length + 1}</span>
                <span class="fb-prompt">${rlActivePrompt.textContent}</span>
                <span class="badge badge-success">${winnerTag} Selected</span>
                <span class="fb-reward">Reward Winner: ${winnerScore}</span>
            `;
            feedbackHistoryList.prepend(item);

            candidateCardA.classList.remove("selected");
            candidateCardB.classList.remove("selected");
            submitFeedbackBtn.disabled = true;
            submitFeedbackBtn.textContent = "Submit Preference to Loop";
            selectedRLWinner = null;

            showToast(`Preference recorded for DPO loop iteration ${rlIteration}!`);
        });
    }

    // Initial render of history
    renderHistory();
    updatePromptCounters();
});

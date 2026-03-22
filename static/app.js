const recordBtn = document.getElementById('recordBtn');
const recordLabel = document.getElementById('recordLabel');
const micIcon = document.getElementById('micIcon');
const stopIcon = document.getElementById('stopIcon');
const fileInput = document.getElementById('fileInput');
const inputArea = document.getElementById('inputArea');
const audioPreview = document.getElementById('audioPreview');
const previewPlayer = document.getElementById('previewPlayer');
const submitBtn = document.getElementById('submitBtn');
const loadingArea = document.getElementById('loadingArea');
const loadingStatus = document.getElementById('loadingStatus');
const loadingBarFill = document.getElementById('loadingBarFill');
const resultsArea = document.getElementById('resultsArea');
const transcriptionText = document.getElementById('transcriptionText');
const emotionText = document.getElementById('emotionText');
const needText = document.getElementById('needText');
const rewriteText = document.getElementById('rewriteText');
const outputPlayer = document.getElementById('outputPlayer');
const errorArea = document.getElementById('errorArea');
const originalAudioPlayer = document.getElementById('originalAudioPlayer');
const errorText = document.getElementById('errorText');
const resetBtn = document.getElementById('resetBtn');
const errorResetBtn = document.getElementById('errorResetBtn');

// Video elements
const videoCard = document.getElementById('videoCard');
const videoLoading = document.getElementById('videoLoading');
const videoStatus = document.getElementById('videoStatus');
const videoResult = document.getElementById('videoResult');
const videoPlayer = document.getElementById('videoPlayer');
const videoError = document.getElementById('videoError');

// Sidebar elements
const historyBtn = document.getElementById('historyBtn');
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const sidebarClose = document.getElementById('sidebarClose');
const sidebarList = document.getElementById('sidebarList');
const sidebarEmpty = document.getElementById('sidebarEmpty');

let mediaRecorder = null;
let recordedChunks = [];
let audioBlob = null;
let videoPollingTimer = null;

// ── IndexedDB ───────────────────────────────────

const DB_NAME = 'MirrorTalkDB';
const DB_VERSION = 1;
const STORE_NAME = 'history';

function openDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        request.onupgradeneeded = (e) => {
            const db = e.target.result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
            }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function saveHistory(record) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        const req = store.add(record);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

async function updateHistory(id, updates) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        const getReq = store.get(id);
        getReq.onsuccess = () => {
            const record = getReq.result;
            if (record) {
                Object.assign(record, updates);
                store.put(record);
            }
            resolve();
        };
        getReq.onerror = () => reject(getReq.error);
    });
}

async function getAllHistory() {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const req = store.getAll();
        req.onsuccess = () => resolve(req.result.reverse());
        req.onerror = () => reject(req.error);
    });
}

// ── Sidebar ─────────────────────────────────────

function openSidebar() {
    sidebar.classList.add('open');
    sidebarOverlay.classList.add('open');
    loadHistoryList();
}

function closeSidebar() {
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('open');
}

historyBtn.addEventListener('click', openSidebar);
sidebarClose.addEventListener('click', closeSidebar);
sidebarOverlay.addEventListener('click', closeSidebar);

async function loadHistoryList() {
    const records = await getAllHistory();
    // Clear existing cards (keep the empty message element)
    sidebarList.querySelectorAll('.history-card').forEach(el => el.remove());

    if (records.length === 0) {
        sidebarEmpty.hidden = false;
        return;
    }
    sidebarEmpty.hidden = true;

    records.forEach(record => {
        const card = document.createElement('div');
        card.className = 'history-card';
        const date = new Date(record.timestamp);
        const timeStr = date.toLocaleDateString(undefined, {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
        });
        card.innerHTML = `
            <div class="history-card-time">${timeStr}</div>
            <div class="history-card-text">${escapeHtml(record.transcription || '')}</div>
            ${record.emotion ? `<span class="history-card-emotion">${escapeHtml(record.emotion)}</span>` : ''}
        `;
        card.addEventListener('click', () => showHistoryRecord(record));
        sidebarList.appendChild(card);
    });
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function showHistoryRecord(record) {
    closeSidebar();

    // Show results area with stored data
    inputArea.hidden = true;
    loadingArea.hidden = true;
    errorArea.hidden = true;
    resultsArea.hidden = false;

    transcriptionText.textContent = record.transcription || '';
    emotionText.textContent = record.emotion || '';
    needText.textContent = record.need || '';
    rewriteText.textContent = record.rewrite || '';

    // Audio URLs (may not be available if server cleaned up)
    originalAudioPlayer.src = '';
    outputPlayer.src = record.audioUrl || '';

    // Video
    if (record.videoUrl) {
        videoLoading.hidden = true;
        videoError.hidden = true;
        videoResult.hidden = false;
        videoPlayer.src = record.videoUrl;
        videoCard.hidden = false;
    } else if (record.imageUrl) {
        // Video was still processing — show image only
        videoLoading.hidden = true;
        videoError.hidden = true;
        videoResult.hidden = false;
        videoPlayer.src = '';
        videoCard.hidden = false;
    } else {
        videoCard.hidden = true;
    }
}

// ── Recording ────────────────────────────────────
recordBtn.addEventListener('click', async () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recordedChunks = [];
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            stream.getTracks().forEach(t => t.stop());
            audioBlob = new Blob(recordedChunks, { type: 'audio/webm' });
            showPreview(audioBlob);
            recordBtn.classList.remove('recording');
            micIcon.style.display = '';
            stopIcon.style.display = 'none';
            recordLabel.textContent = 'Tap to record';
        };

        mediaRecorder.start();
        recordBtn.classList.add('recording');
        micIcon.style.display = 'none';
        stopIcon.style.display = '';
        recordLabel.textContent = 'Recording...';
    } catch (err) {
        alert('Microphone access denied. Please allow microphone access and try again.');
    }
});

// ── File upload ──────────────────────────────────
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    audioBlob = file;
    showPreview(audioBlob);
});

function showPreview(blob) {
    const url = URL.createObjectURL(blob);
    previewPlayer.src = url;
    audioPreview.hidden = false;
}

// ── Submit ───────────────────────────────────────
submitBtn.addEventListener('click', async () => {
    if (!audioBlob) return;

    inputArea.hidden = true;
    loadingArea.hidden = false;
    resultsArea.hidden = true;
    errorArea.hidden = true;

    // Animated loading steps
    const steps = [
        { text: 'Listening deeply...', pct: 15 },
        { text: 'Decoupling emotion...', pct: 40 },
        { text: 'Rewriting with care...', pct: 65 },
        { text: 'Cloning your voice...', pct: 85 },
    ];
    let stepIdx = 0;
    loadingStatus.textContent = steps[0].text;
    loadingBarFill.style.width = steps[0].pct + '%';

    const stepInterval = setInterval(() => {
        stepIdx++;
        if (stepIdx < steps.length) {
            loadingStatus.textContent = steps[stepIdx].text;
            loadingBarFill.style.width = steps[stepIdx].pct + '%';
        }
    }, 4000);

    try {
        const formData = new FormData();
        const ext = audioBlob.type.includes('webm') ? 'webm' : 'wav';
        formData.append('audio', audioBlob, `recording.${ext}`);

        const response = await fetch('/api/transform', {
            method: 'POST',
            body: formData,
        });

        clearInterval(stepInterval);

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Something went wrong');
        }

        const data = await response.json();
        console.log('API response:', JSON.stringify(data, null, 2));

        // Fill results
        transcriptionText.textContent = data.transcription || '';
        emotionText.textContent = data.emotion || '';
        needText.textContent = data.need || '';
        rewriteText.textContent = data.rewrite || '';
        originalAudioPlayer.src = URL.createObjectURL(audioBlob);
        outputPlayer.src = data.audio_url;

        // Transition: loading → results
        loadingBarFill.style.width = '100%';
        setTimeout(() => {
            loadingArea.hidden = true;
            resultsArea.hidden = false;
        }, 400);

        // Save to history
        const historyRecord = {
            timestamp: new Date().toISOString(),
            transcription: data.transcription || '',
            emotion: data.emotion || '',
            need: data.need || '',
            rewrite: data.rewrite || '',
            audioUrl: data.audio_url || '',
            videoUrl: null,
            imageUrl: null,
        };
        const recordId = await saveHistory(historyRecord);

        // Start video generation (non-blocking)
        startVideoGeneration(data.rewrite, data.emotion, recordId);

    } catch (err) {
        clearInterval(stepInterval);
        loadingArea.hidden = true;
        errorText.textContent = err.message;
        errorArea.hidden = false;
    }
});

// ── Video Generation (non-blocking) ─────────────

async function startVideoGeneration(rewrite, emotion, historyId) {
    // Reset video card state
    videoCard.hidden = false;
    videoLoading.hidden = false;
    videoResult.hidden = true;
    videoError.hidden = true;
    videoStatus.textContent = 'Generating image...';

    try {
        const resp = await fetch('/api/generate-video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rewrite, emotion }),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Video generation failed');
        }

        const data = await resp.json();
        const taskId = data.task_id;

        // Update history with image URL
        if (historyId && data.image_url) {
            await updateHistory(historyId, { imageUrl: data.image_url });
        }

        videoStatus.textContent = 'Generating video...';
        pollVideoStatus(taskId, historyId);

    } catch (err) {
        videoLoading.hidden = true;
        videoError.textContent = err.message;
        videoError.hidden = false;
    }
}

function pollVideoStatus(taskId, historyId) {
    if (videoPollingTimer) clearInterval(videoPollingTimer);

    videoPollingTimer = setInterval(async () => {
        try {
            const resp = await fetch(`/api/video-status?taskId=${encodeURIComponent(taskId)}`);
            if (!resp.ok) throw new Error('Status check failed');

            const data = await resp.json();

            if (data.status === 'completed') {
                clearInterval(videoPollingTimer);
                videoPollingTimer = null;
                videoLoading.hidden = true;
                videoResult.hidden = false;
                videoPlayer.src = data.video_url;

                // Update history with video URL
                if (historyId) {
                    await updateHistory(historyId, { videoUrl: data.video_url });
                }
            } else if (data.status === 'failed') {
                clearInterval(videoPollingTimer);
                videoPollingTimer = null;
                videoLoading.hidden = true;
                videoError.textContent = data.error || 'Video generation failed';
                videoError.hidden = false;
            }
            // else: still processing, keep polling
        } catch (err) {
            clearInterval(videoPollingTimer);
            videoPollingTimer = null;
            videoLoading.hidden = true;
            videoError.textContent = 'Lost connection to video service';
            videoError.hidden = false;
        }
    }, 3000);
}

// ── Reset ────────────────────────────────────────
function reset() {
    audioBlob = null;
    recordedChunks = [];
    audioPreview.hidden = true;
    previewPlayer.src = '';
    originalAudioPlayer.src = '';
    outputPlayer.src = '';
    fileInput.value = '';
    loadingBarFill.style.width = '0%';
    inputArea.hidden = false;
    loadingArea.hidden = true;
    resultsArea.hidden = true;
    errorArea.hidden = true;

    // Reset video state
    if (videoPollingTimer) {
        clearInterval(videoPollingTimer);
        videoPollingTimer = null;
    }
    videoCard.hidden = true;
    videoLoading.hidden = false;
    videoResult.hidden = true;
    videoError.hidden = true;
    videoPlayer.src = '';
}

resetBtn.addEventListener('click', reset);
errorResetBtn.addEventListener('click', reset);

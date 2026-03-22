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

let mediaRecorder = null;
let recordedChunks = [];
let audioBlob = null;

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
    } catch (err) {
        clearInterval(stepInterval);
        loadingArea.hidden = true;
        errorText.textContent = err.message;
        errorArea.hidden = false;
    }
});

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
}

resetBtn.addEventListener('click', reset);
errorResetBtn.addEventListener('click', reset);

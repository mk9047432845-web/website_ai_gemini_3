<script>
    // ... (Keep your existing Drag & Drop and File variables) ...
    const chatSection = document.getElementById('chat-section');
    const chatBox = document.getElementById('chat-box');

    // --- UPDATED SHOW RESULTS ---
    function showResults(data, file) {
        scanner.style.display = 'none';
        const reader = new FileReader();
        reader.onload = (e) => { document.getElementById('result-img-preview').src = e.target.result; }
        reader.readAsDataURL(file);

        const title = document.getElementById('pred-title');
        title.innerText = data.prediction.toUpperCase();
        
        if(data.prediction === 'malignant') title.style.color = '#ff0055';
        else if(data.prediction === 'benign') title.style.color = '#00ff9d';
        else title.style.color = '#00f2ff';

        document.getElementById('conf-label').innerText = `Confidence Index: ${(data.confidence * 100).toFixed(2)}%`;

        const container = document.getElementById('stats-container');
        container.innerHTML = '';
        const sorted = Object.entries(data.probabilities).sort(([,a],[,b]) => b - a);
        
        sorted.forEach(([label, score]) => {
            let color = label === 'malignant' ? '#ff0055' : (label === 'benign' ? '#00ff9d' : '#00f2ff');
            const pct = (score * 100).toFixed(1);
            const html = `<div class="stat-row">
                <div class="stat-header"><span>${label.toUpperCase()}</span><span>${pct}%</span></div>
                <div class="progress-track"><div class="progress-fill" style="width: ${pct}%; background-color: ${color}; box-shadow: 0 0 10px ${color}"></div></div>
            </div>`;
            container.insertAdjacentHTML('beforeend', html);
        });

        // Toggle Views
        uploadView.style.display = 'none';
        resultsView.style.display = 'block';
        
        // Wake up Chatbot
        chatSection.style.display = 'block';
        loadSuggestions();
    }

    // --- CHATBOT LOGIC ---
    function loadSuggestions() {
        fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: "" })
        })
        .then(res => res.json())
        .then(data => {
            const area = document.getElementById('suggestions-area');
            area.innerHTML = '';
            data.suggestions.forEach(s => {
                const btn = document.createElement('button');
                btn.className = 'suggestion-btn';
                btn.innerText = s;
                btn.onclick = () => sendChat(s);
                area.appendChild(btn);
            });
        });
    }

    async function sendChat(manualMsg = null) {
        const input = document.getElementById('user-input');
        const message = manualMsg || input.value.trim();
        if (!message) return;

        if(!manualMsg) input.value = '';

        chatBox.innerHTML += `<div class="chat-msg user-msg"><b>YOU:</b> ${message}</div>`;
        
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        const data = await response.json();
        
        chatBox.innerHTML += `<div class="chat-msg bot-msg"><b>BOT:</b> ${data.reply}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Fix for Reset Logic (Browse Trigger)
    function resetUI() {
        form.reset();
        uploadView.style.display = 'block';
        resultsView.style.display = 'none';
        chatSection.style.display = 'none';
        chatBox.innerHTML = '<div class="chat-msg bot-msg"><b>SYSTEM:</b> Analysis complete. How can I assist you with these results?</div>';
        submitBtn.innerText = "ANALYZE SCAN";
        submitBtn.disabled = true;
        scanner.style.display = 'none';
        
        document.querySelector('.upload-text-main').innerText = "Upload Medical Imagery";
        document.querySelector('.upload-icon').classList.replace('fa-check-circle', 'fa-cloud-arrow-up');
        document.querySelector('.upload-icon').style.color = 'var(--accent-cyan)';
    }
</script>
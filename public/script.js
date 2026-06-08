const searchInput = document.getElementById('searchInput');
const searchButton = document.getElementById('searchButton');
const loader = document.getElementById('loader');
const resultsContainer = document.getElementById('resultsContainer');
const docsContainer = document.getElementById('docsContainer');
const aiAnswerContainer = document.getElementById('aiAnswerContainer');
const generalAnswerContainer = document.getElementById('generalAnswerContainer');
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');

// Fetch and display source docs on load
async function loadSourceDocs() {
    try {
        const response = await fetch('/api/docs');
        if (!response.ok) throw new Error('Failed to fetch docs');
        const data = await response.json();
        
        docsContainer.innerHTML = ''; // Clear loader
        
        if (data.docs && data.docs.length > 0) {
            data.docs.forEach((docText, index) => {
                const docDiv = document.createElement('div');
                docDiv.className = 'doc-item';
                docDiv.textContent = docText;
                docDiv.style.animationDelay = `${index * 0.05}s`;
                docDiv.style.animation = 'slideUp 0.3s forwards ease-out';
                docDiv.style.opacity = '0';
                docDiv.style.transform = 'translateY(10px)';
                docsContainer.appendChild(docDiv);
            });
        } else {
            docsContainer.innerHTML = '<div style="color: var(--text-secondary);">No source data found.</div>';
        }
    } catch (error) {
        console.error('Error loading docs:', error);
        docsContainer.innerHTML = '<div style="color: #ef4444;">Failed to load source data.</div>';
    }
}

// Call on load
document.addEventListener('DOMContentLoaded', loadSourceDocs);


async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    // UI State: Loading
    resultsContainer.innerHTML = '';
    aiAnswerContainer.innerHTML = '';
    generalAnswerContainer.innerHTML = '';
    aiAnswerContainer.classList.add('hidden');
    generalAnswerContainer.classList.add('hidden');
    loader.classList.remove('hidden');

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query })
        });

        if (!response.ok) {
            throw new Error('Failed to fetch results');
        }

        const data = await response.json();
        
        // Render AI Answers
        if (data.llm_answer) {
            aiAnswerContainer.innerHTML = `
                <div class="ai-answer-title">✨ RAG Synthesized Answer</div>
                <div class="ai-answer-text">${data.llm_answer.replace(/\n/g, '<br>')}</div>
            `;
            aiAnswerContainer.classList.remove('hidden');
        }

        if (data.general_answer) {
            generalAnswerContainer.innerHTML = `
                <div class="ai-answer-title" style="color: #64748b;">🌍 General AI Knowledge</div>
                <div class="ai-answer-text">${data.general_answer.replace(/\n/g, '<br>')}</div>
            `;
            generalAnswerContainer.classList.remove('hidden');
        }

        renderResults(data.results);
    } catch (error) {
        console.error('Search error:', error);
        resultsContainer.innerHTML = `
            <div class="error-message">
                Something went wrong while searching. Ensure the backend is running.
            </div>
        `;
    } finally {
        loader.classList.add('hidden');
    }
}

function renderResults(results) {
    if (!results || results.length === 0) {
        resultsContainer.innerHTML = `
            <div style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                No matching results found.
            </div>
        `;
        return;
    }

    resultsContainer.innerHTML = `
        <h3 style="margin-top: 1rem; margin-bottom: 0.5rem; color: var(--text-primary); font-size: 1.2rem; font-weight: 600;">
            🔍 Semantic Results (Source Documents)
        </h3>
    `;
    
    results.forEach((res, index) => {
        const p = document.createElement('p');
        p.className = 'book-paragraph';
        p.style.animationDelay = `${index * 0.1}s`;
        
        const scorePercentage = (res.score * 100).toFixed(1);

        p.innerHTML = `${res.text} <span class="score-inline">[Confidence: ${scorePercentage}%]</span>`;
        resultsContainer.appendChild(p);
    });
}

// Event Listeners
searchButton.addEventListener('click', performSearch);
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        performSearch();
    }
});

if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });
}

// Accordion Logic
document.querySelectorAll('.accordion-header').forEach(button => {
    button.addEventListener('click', () => {
        const content = button.nextElementSibling;
        const icon = button.querySelector('.icon');
        
        if (content.style.maxHeight) {
            // Close it
            content.style.maxHeight = null;
            icon.textContent = '▼';
        } else {
            // Open it
            content.style.maxHeight = content.scrollHeight + "px";
            icon.textContent = '▲';
        }
    });
});

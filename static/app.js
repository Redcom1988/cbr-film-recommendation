document.addEventListener('DOMContentLoaded', () => {
    const searchBtn = document.getElementById('searchBtn');
    const queryInput = document.getElementById('queryInput');
    const userIdInput = document.getElementById('userIdInput');
    const resultsSection = document.getElementById('resultsSection');
    const moviesGrid = document.getElementById('moviesGrid');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const finishBtn = document.getElementById('finishBtn');

    let currentQuery = '';
    let acceptedIds = new Set();
    let rejectedIds = new Set();

    searchBtn.addEventListener('click', performSearch);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });

    finishBtn.addEventListener('click', submitFeedback);

    async function performSearch() {
        const query = queryInput.value.trim();
        const userId = userIdInput.value.trim();

        if (!query) return;

        currentQuery = query;
        acceptedIds.clear();
        rejectedIds.clear();
        finishBtn.classList.add('hidden');

        resultsSection.classList.remove('hidden');
        moviesGrid.innerHTML = '';
        loadingIndicator.classList.remove('hidden');
        
        // Smooth scroll to results
        setTimeout(() => {
            window.scrollTo({ top: window.innerHeight * 0.7, behavior: 'smooth' });
        }, 100);

        try {
            const res = await fetch('/api/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, user_id: userId || null })
            });
            const data = await res.json();
            
            loadingIndicator.classList.add('hidden');
            
            if (data.results && data.results.length > 0) {
                renderMovies(data.results);
                finishBtn.classList.remove('hidden');
            } else {
                moviesGrid.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: var(--text-secondary); font-size: 1.2rem;">No movies found. Try another query.</p>';
            }

        } catch (err) {
            loadingIndicator.classList.add('hidden');
            moviesGrid.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: var(--danger);">Error fetching recommendations.</p>';
        }
    }

    function renderMovies(movies) {
        moviesGrid.innerHTML = '';
        movies.forEach(movie => {
            const card = document.createElement('div');
            card.className = 'movie-card';
            
            const finalScore = (movie.score * 100).toFixed(0);
            
            card.innerHTML = `
                <div class="movie-poster-placeholder">
                    <i class="fa-solid fa-film"></i>
                    <div class="score-badge"><i class="fa-solid fa-star"></i> ${finalScore}% Match</div>
                </div>
                <div class="movie-info">
                    <div class="movie-title">${movie.title}</div>
                    <div class="movie-genres">${movie.genres || 'Unknown'}</div>
                    <div class="movie-overview">${movie.overview || 'No synopsis available.'}</div>
                    <div class="feedback-row">
                        <button class="feed-btn btn-accept" data-id="${movie.movieId}" title="Relevant"><i class="fa-solid fa-thumbs-up"></i></button>
                        <button class="feed-btn btn-reject" data-id="${movie.movieId}" title="Not Relevant"><i class="fa-solid fa-thumbs-down"></i></button>
                    </div>
                </div>
            `;

            moviesGrid.appendChild(card);
        });

        document.querySelectorAll('.btn-accept').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const id = parseInt(e.currentTarget.dataset.id);
                const rejectBtn = e.currentTarget.nextElementSibling;
                
                if (acceptedIds.has(id)) {
                    acceptedIds.delete(id);
                    e.currentTarget.classList.remove('accepted');
                } else {
                    acceptedIds.add(id);
                    rejectedIds.delete(id);
                    e.currentTarget.classList.add('accepted');
                    rejectBtn.classList.remove('rejected');
                }
            });
        });

        document.querySelectorAll('.btn-reject').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const id = parseInt(e.currentTarget.dataset.id);
                const acceptBtn = e.currentTarget.previousElementSibling;
                
                if (rejectedIds.has(id)) {
                    rejectedIds.delete(id);
                    e.currentTarget.classList.remove('rejected');
                } else {
                    rejectedIds.add(id);
                    acceptedIds.delete(id);
                    e.currentTarget.classList.add('rejected');
                    acceptBtn.classList.remove('accepted');
                }
            });
        });
    }

    async function submitFeedback() {
        if (acceptedIds.size === 0 && rejectedIds.size === 0) {
            alert('Select thumbs up/down for at least one movie first.');
            return;
        }

        const originalText = finishBtn.innerHTML;
        finishBtn.innerHTML = 'Saving <i class="fa-solid fa-spinner fa-spin"></i>';
        finishBtn.disabled = true;

        try {
            await fetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: currentQuery,
                    accepted_ids: Array.from(acceptedIds),
                    rejected_ids: Array.from(rejectedIds)
                })
            });
            
            finishBtn.innerHTML = '<i class="fa-solid fa-check"></i> Feedback Saved';
            setTimeout(() => {
                finishBtn.innerHTML = originalText;
                finishBtn.disabled = false;
            }, 2000);

        } catch (err) {
            finishBtn.innerHTML = 'Error!';
            setTimeout(() => {
                finishBtn.innerHTML = originalText;
                finishBtn.disabled = false;
            }, 2000);
        }
    }
});

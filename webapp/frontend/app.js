/* Next-movie recommender UI.
   Talks to /api/search and /api/recommend (same origin as this page). */

const searchEl = document.getElementById("search");
const resultsEl = document.getElementById("results");
const historyEl = document.getElementById("history");
const historyEmptyEl = document.getElementById("history-empty");
const clearBtn = document.getElementById("clear");
const recommendBtn = document.getElementById("recommend");
const statusEl = document.getElementById("status");
const recsPanel = document.getElementById("recs-panel");
const recsEl = document.getElementById("recs");

// History is ordered oldest -> newest, matching predict_next_items'
// "most recent last" contract.
const history = [];
let matches = [];
let activeIndex = -1;
let searchTimer = null;

// Nudge a sleeping free-tier instance awake while the user is still picking
// movies, so the first real request isn't paying the whole cold start.
fetch("/api/health").catch(() => {});

/* ---------------- search ---------------- */

searchEl.addEventListener("input", () => {
  clearTimeout(searchTimer);
  const q = searchEl.value.trim();
  if (!q) return closeDropdown();
  searchTimer = setTimeout(() => runSearch(q), 200);
});

async function runSearch(q) {
  try {
    const resp = await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=10`);
    if (!resp.ok) throw new Error(`search failed: ${resp.status}`);
    const data = await resp.json();
    matches = data.results.filter((m) => !history.some((h) => h.movie_id === m.movie_id));
    renderDropdown();
  } catch (err) {
    closeDropdown();
  }
}

function renderDropdown() {
  activeIndex = -1;
  resultsEl.innerHTML = "";

  if (matches.length === 0) {
    const li = document.createElement("li");
    li.className = "none";
    li.textContent = "No matches";
    resultsEl.appendChild(li);
    resultsEl.hidden = false;
    return;
  }

  matches.forEach((m, i) => {
    const li = document.createElement("li");
    li.textContent = m.title;
    li.addEventListener("mousedown", (e) => {
      e.preventDefault(); // keep focus in the input
      addMovie(i);
    });
    resultsEl.appendChild(li);
  });
  resultsEl.hidden = false;
}

function closeDropdown() {
  resultsEl.hidden = true;
  resultsEl.innerHTML = "";
  matches = [];
  activeIndex = -1;
}

searchEl.addEventListener("keydown", (e) => {
  if (resultsEl.hidden || matches.length === 0) return;
  const items = [...resultsEl.querySelectorAll("li:not(.none)")];
  if (items.length === 0) return;

  if (e.key === "ArrowDown") {
    e.preventDefault();
    activeIndex = (activeIndex + 1) % items.length;
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    activeIndex = (activeIndex - 1 + items.length) % items.length;
  } else if (e.key === "Enter") {
    e.preventDefault();
    addMovie(activeIndex >= 0 ? activeIndex : 0);
    return;
  } else if (e.key === "Escape") {
    closeDropdown();
    return;
  } else {
    return;
  }

  items.forEach((el, i) => el.classList.toggle("active", i === activeIndex));
  items[activeIndex].scrollIntoView({ block: "nearest" });
});

searchEl.addEventListener("blur", () => setTimeout(closeDropdown, 120));

/* ---------------- history ---------------- */

function addMovie(index) {
  const movie = matches[index];
  if (!movie) return;
  if (!history.some((h) => h.movie_id === movie.movie_id)) {
    history.push(movie);
    renderHistory();
  }
  searchEl.value = "";
  closeDropdown();
  searchEl.focus();
}

function removeMovie(movieId) {
  const i = history.findIndex((h) => h.movie_id === movieId);
  if (i !== -1) {
    history.splice(i, 1);
    renderHistory();
  }
}

function renderHistory() {
  historyEl.innerHTML = "";
  history.forEach((m) => {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = m.title;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "×";
    btn.setAttribute("aria-label", `Remove ${m.title}`);
    btn.addEventListener("click", () => removeMovie(m.movie_id));
    li.append(label, btn);
    historyEl.appendChild(li);
  });

  const has = history.length > 0;
  historyEmptyEl.hidden = has;
  clearBtn.hidden = !has;
  recommendBtn.disabled = !has;
}

clearBtn.addEventListener("click", () => {
  history.length = 0;
  renderHistory();
  recsPanel.hidden = true;
});

/* ---------------- recommend ---------------- */

recommendBtn.addEventListener("click", async () => {
  if (history.length === 0) return;

  recommendBtn.disabled = true;
  statusEl.className = "status";
  statusEl.textContent = "Thinking…";
  statusEl.hidden = false;

  // Free-tier instances sleep when idle; say so rather than looking hung.
  const slowTimer = setTimeout(() => {
    statusEl.textContent =
      "Waking up the model server — the first request after idle can take up to a minute.";
  }, 3000);

  try {
    const resp = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        history: history.map((h) => h.movie_id),
        top_k: 5,
      }),
    });
    if (!resp.ok) throw new Error(`request failed: ${resp.status}`);
    const data = await resp.json();
    renderRecs(data.recommendations);
    statusEl.hidden = true;
  } catch (err) {
    statusEl.className = "status error";
    statusEl.textContent = "Something went wrong. Try again in a moment.";
  } finally {
    clearTimeout(slowTimer);
    recommendBtn.disabled = history.length === 0;
  }
});

function renderRecs(recs) {
  recsEl.innerHTML = "";

  if (!recs || recs.length === 0) {
    recsPanel.hidden = true;
    statusEl.className = "status";
    statusEl.textContent = "No recommendations for that history.";
    statusEl.hidden = false;
    return;
  }

  // Raw logits aren't on a fixed scale, so bars are normalised within this
  // result set purely for visual comparison — the printed number is the real
  // score.
  const scores = recs.map((r) => r.score);
  const max = Math.max(...scores);
  const min = Math.min(...scores);
  const span = max - min || 1;

  recs.forEach((r) => {
    const li = document.createElement("li");

    const title = document.createElement("div");
    title.className = "rec-title";
    title.textContent = r.title;

    const barWrap = document.createElement("div");
    barWrap.className = "rec-bar-wrap";

    const bar = document.createElement("div");
    bar.className = "rec-bar";
    const fill = document.createElement("span");
    // Floor at 18% so the lowest-ranked item still reads as a bar.
    fill.style.width = `${18 + ((r.score - min) / span) * 82}%`;
    bar.appendChild(fill);

    const score = document.createElement("div");
    score.className = "rec-score";
    score.textContent = r.score.toFixed(2);

    barWrap.append(bar, score);
    li.append(title, barWrap);
    recsEl.appendChild(li);
  });

  recsPanel.hidden = false;
  recsPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

renderHistory();

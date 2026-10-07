/* next-movie — talks to /api/search and /api/recommend (same origin). */

const searchEl   = document.getElementById("search");
const resultsEl  = document.getElementById("results");
const historyEl  = document.getElementById("history");
const emptyEl    = document.getElementById("history-empty");
const runBtn     = document.getElementById("recommend");
const runLabel   = document.getElementById("run-label");
const recsEl     = document.getElementById("recs");
const placeholder= document.getElementById("recs-placeholder");
const badgeEl    = document.getElementById("out-badge");
const liveDot    = document.getElementById("live-dot");

// Oldest -> newest, matching predict_next_items' "most recent last" contract.
const history = [];
let matches = [];
let activeIndex = -1;
let searchTimer = null;
let hasRun = false;

/* Wake a sleeping free-tier instance while the user is still picking movies,
   so the first real request isn't paying the whole cold start. */
fetch("/api/health")
  .then((r) => { if (r.ok) liveDot.classList.add("on"); })
  .catch(() => {});

/* ───────── search ───────── */

searchEl.addEventListener("input", () => {
  clearTimeout(searchTimer);
  const q = searchEl.value.trim();
  if (!q) return closeDropdown();
  searchTimer = setTimeout(() => runSearch(q), 200);
});

async function runSearch(q) {
  try {
    const resp = await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=10`);
    if (!resp.ok) throw new Error(String(resp.status));
    const data = await resp.json();
    matches = data.results.filter(
      (m) => !history.some((h) => h.movie_id === m.movie_id)
    );
    renderDropdown();
  } catch {
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
      e.preventDefault();          // keep focus in the input
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

/* ───────── history ───────── */

function addMovie(i) {
  const movie = matches[i];
  if (!movie) return;
  if (history.length >= 63) return;          // API caps history at 63
  if (!history.some((h) => h.movie_id === movie.movie_id)) {
    history.push(movie);
    renderHistory();
  }
  searchEl.value = "";
  closeDropdown();
  searchEl.focus();
}

function removeMovie(id) {
  const i = history.findIndex((h) => h.movie_id === id);
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
  emptyEl.hidden = has;
  runBtn.disabled = !has;
  runLabel.textContent = hasRun ? "Run model again" : "Run model";
}

/* ───────── recommend ───────── */

runBtn.addEventListener("click", async () => {
  if (history.length === 0) return;

  runBtn.disabled = true;
  runLabel.textContent = "Running…";
  clearError();
  badgeEl.hidden = true;

  // Free-tier instances sleep when idle; say so rather than looking hung.
  const slowTimer = setTimeout(() => {
    runLabel.textContent = "Waking the server…";
  }, 3000);

  const started = performance.now();

  try {
    const resp = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        history: history.map((h) => h.movie_id),
        top_k: 5,
      }),
    });
    if (!resp.ok) throw new Error(String(resp.status));
    const data = await resp.json();
    const ms = Math.round(performance.now() - started);
    renderRecs(data.recommendations, ms);
    hasRun = true;
    liveDot.classList.add("on");
  } catch {
    showError("Couldn't reach the model. Give it a moment and try again.");
  } finally {
    clearTimeout(slowTimer);
    runBtn.disabled = history.length === 0;
    runLabel.textContent = hasRun ? "Run model again" : "Run model";
  }
});

function renderRecs(recs, ms) {
  recsEl.innerHTML = "";

  if (!recs || recs.length === 0) {
    placeholder.hidden = false;
    badgeEl.hidden = true;
    return;
  }

  recs.forEach((r) => {
    const li = document.createElement("li");

    const main = document.createElement("div");
    const title = document.createElement("div");
    title.className = "rec-title";
    title.textContent = r.display || r.title;
    main.appendChild(title);

    const bits = [r.year, ...(r.genres || [])].filter(Boolean);
    if (bits.length) {
      const meta = document.createElement("div");
      meta.className = "rec-meta";
      meta.textContent = bits.join(" · ");
      main.appendChild(meta);
    }

    const score = document.createElement("div");
    score.className = "rec-score";
    score.textContent = r.score.toFixed(2);

    li.append(main, score);
    recsEl.appendChild(li);
  });

  placeholder.hidden = true;
  badgeEl.textContent = `COMPLETE · ${ms}MS`;
  badgeEl.hidden = false;
}

/* ───────── errors ───────── */

function showError(msg) {
  clearError();
  const p = document.createElement("p");
  p.className = "err";
  p.id = "err";
  p.textContent = msg;
  runBtn.insertAdjacentElement("afterend", p);
}

function clearError() {
  const existing = document.getElementById("err");
  if (existing) existing.remove();
}

renderHistory();

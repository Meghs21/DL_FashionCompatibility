// State Management
const wardrobePools = {
  top: [],
  bottom: [],
  footwear: [],
  accessories: []
};

let activePortraitFile = null;
let cachedBlogs = [];
let currentCategoryFilter = "all";

const MULTI_PRESETS = {
  casual: {
    top: ["sample_images/casual_top.png", "sample_images/street_top.png"],
    bottom: ["sample_images/casual_bottom.png", "sample_images/formal_bottom.png"],
    footwear: ["sample_images/casual_shoes.png"],
    accessories: ["sample_images/casual_acc.png"]
  },
  formal: {
    top: ["sample_images/formal_top.png", "sample_images/casual_top.png"],
    bottom: ["sample_images/formal_bottom.png", "sample_images/casual_bottom.png"],
    footwear: ["sample_images/formal_shoes.png", "sample_images/casual_shoes.png"],
    accessories: ["sample_images/formal_acc.png"]
  },
  street: {
    top: ["sample_images/street_top.png", "sample_images/casual_top.png"],
    bottom: ["sample_images/street_bottom.png", "sample_images/formal_bottom.png"],
    footwear: ["sample_images/street_shoes.png"],
    accessories: ["sample_images/street_acc.png", "sample_images/casual_acc.png"]
  }
};

// Initialize App
document.addEventListener("DOMContentLoaded", () => {
  fetchFashionBlogs();
});

// Tab Navigation
function switchTab(tabId) {
  const tabs = document.querySelectorAll('.tab-content');
  const navBtns = document.querySelectorAll('.nav-item');

  tabs.forEach(tab => tab.classList.remove('active'));
  navBtns.forEach(btn => btn.classList.remove('active'));

  document.getElementById(tabId).classList.add('active');

  if (tabId === 'studioTab') navBtns[0].classList.add('active');
  else if (tabId === 'trendsTab') navBtns[1].classList.add('active');
  else if (tabId === 'colorTab') navBtns[2].classList.add('active');
  else if (tabId === 'stylistTab') navBtns[3].classList.add('active');
}

// -------------------------------------------------------------
// FUNCTIONAL PERSONAL COLOR & UNDERTONE ANALYZER
// -------------------------------------------------------------
function handlePortraitSelect(event) {
  const file = event.target.files[0];
  if (file) {
    activePortraitFile = file;
    const imgEl = document.getElementById("portraitPreview");
    const placeholder = document.getElementById("portraitPlaceholder");

    imgEl.src = URL.createObjectURL(file);
    imgEl.style.display = "block";
    placeholder.style.display = "none";
  }
}

async function runPersonalColorAnalysis() {
  if (!activePortraitFile) {
    alert("Please select or drop a clear portrait / selfie photo first!");
    return;
  }

  const analyzeBtn = document.getElementById("analyzeColorBtn");
  const resultsPanel = document.getElementById("portraitResultsPanel");

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = `<span>Sampling Facial Pixels & Analyzing Undertone...</span>`;

  const formData = new FormData();
  formData.append("portrait", activePortraitFile);

  try {
    const response = await fetch("/api/analyze_personal_color", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.error || "Personal color analysis failed");
    }

    const data = await response.json();
    renderPersonalColorResults(data);

  } catch (err) {
    console.error(err);
    alert(`Analysis Failed: ${err.message}`);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = `<span>Analyze My Personal Color & Palette ✨</span>`;
  }
}

function renderPersonalColorResults(data) {
  const resultsPanel = document.getElementById("portraitResultsPanel");
  resultsPanel.style.display = "block";
  resultsPanel.scrollIntoView({ behavior: 'smooth' });

  document.getElementById("resultSkinCircle").style.background = data.skin_hex;
  document.getElementById("resultSkinHex").innerText = data.skin_hex;
  document.getElementById("resultUndertoneBadge").innerText = data.undertone;
  document.getElementById("resultSeason").innerText = data.season;

  // Best colors chips
  const bestList = document.getElementById("bestColorsList");
  bestList.innerHTML = "";
  data.best_garment_colors.forEach(c => {
    const chip = document.createElement("div");
    chip.className = "g-chip";
    chip.innerHTML = `
      <div class="g-color" style="background: ${c.hex}"></div>
      <span>${c.name}</span>
    `;
    bestList.appendChild(chip);
  });

  // Avoid colors chips
  const avoidList = document.getElementById("avoidColorsList");
  avoidList.innerHTML = "";
  data.avoid_colors.forEach(c => {
    const chip = document.createElement("div");
    chip.className = "g-chip";
    chip.innerHTML = `
      <div class="g-color" style="background: ${c.hex}"></div>
      <span>${c.name}</span>
    `;
    avoidList.appendChild(chip);
  });

  document.getElementById("jewelryText").innerText = data.jewelry_recommendation;
  document.getElementById("stylistAdviceText").innerText = data.stylist_advice;
}

// -------------------------------------------------------------
// REAL-TIME FASHION BLOGS API INTEGRATION
// -------------------------------------------------------------
async function fetchFashionBlogs() {
  try {
    const response = await fetch("/api/fashion_blogs");
    if (!response.ok) throw new Error("Failed to fetch fashion blogs");
    const data = await response.json();
    cachedBlogs = data.articles;
    renderBlogs(cachedBlogs);
  } catch (err) {
    console.error("Error loading real-time fashion blogs:", err);
  }
}

function renderBlogs(articles) {
  const grid = document.getElementById("blogsGrid");
  if (!grid) return;
  grid.innerHTML = "";

  if (articles.length === 0) {
    grid.innerHTML = `
      <div style="grid-column: 1/-1; text-align:center; padding:3rem; color:var(--text-muted)">
        <p style="font-size:1.2rem">No editorial trends found matching your criteria.</p>
      </div>
    `;
    return;
  }

  articles.forEach(article => {
    const card = document.createElement("div");
    card.className = "blog-card";
    card.onclick = () => openBlogModal(article.id);

    card.innerHTML = `
      <div class="blog-img-container">
        <img src="${article.image}" alt="${article.title}">
        <span class="blog-cat-badge">${article.category}</span>
      </div>
      <div class="blog-body">
        <h3 class="blog-title">${article.title}</h3>
        <p class="blog-summary">${article.summary}</p>
        <div class="blog-footer">
          <span>✍️ ${article.author}</span>
          <span>⏱️ ${article.read_time}</span>
        </div>
      </div>
    `;
    grid.appendChild(card);
  });
}

function filterBlogs(category) {
  currentCategoryFilter = category;
  const chips = document.querySelectorAll('.filter-chip');
  chips.forEach(c => c.classList.remove('active'));

  const activeChip = Array.from(chips).find(c => c.innerText.toLowerCase().includes(category.toLowerCase()) || (category === 'all' && c.innerText.includes('All')));
  if (activeChip) activeChip.classList.add('active');

  applyFilters();
}

function searchBlogs() {
  applyFilters();
}

function applyFilters() {
  const searchQuery = (document.getElementById("blogSearchInput")?.value || "").toLowerCase();
  
  let filtered = cachedBlogs;

  if (currentCategoryFilter !== 'all') {
    filtered = filtered.filter(b => b.category.toLowerCase().includes(currentCategoryFilter.toLowerCase()));
  }

  if (searchQuery.trim() !== "") {
    filtered = filtered.filter(b => 
      b.title.toLowerCase().includes(searchQuery) || 
      b.summary.toLowerCase().includes(searchQuery) ||
      b.category.toLowerCase().includes(searchQuery) ||
      b.content.toLowerCase().includes(searchQuery)
    );
  }

  renderBlogs(filtered);
}

function openBlogModal(articleId) {
  const article = cachedBlogs.find(b => b.id === articleId);
  if (!article) return;

  document.getElementById("modalCategory").innerText = article.category;
  document.getElementById("modalTitle").innerText = article.title;
  document.getElementById("modalMeta").innerText = `${article.author} • ${article.date} • ${article.read_time}`;
  document.getElementById("modalImg").src = article.image;
  document.getElementById("modalContent").innerHTML = article.content;

  document.getElementById("articleModal").classList.add("active");
}

function closeBlogModal() {
  document.getElementById("articleModal").classList.remove("active");
}

// -------------------------------------------------------------
// MULTI-ITEM WARDROBE POOL HANDLERS
// -------------------------------------------------------------
function handleMultiFileSelect(event, category) {
  const files = Array.from(event.target.files);
  files.forEach(file => {
    wardrobePools[category].push(file);
  });
  renderThumbnails(category);
}

function renderThumbnails(category) {
  const strip = document.getElementById(`${category}Thumbnails`);
  strip.innerHTML = "";

  wardrobePools[category].forEach((fileItem, idx) => {
    const thumbDiv = document.createElement("div");
    thumbDiv.className = "thumb-item";

    const img = document.createElement("img");
    if (typeof fileItem === "string") {
      img.src = fileItem;
    } else {
      img.src = URL.createObjectURL(fileItem);
    }

    const removeBtn = document.createElement("button");
    removeBtn.className = "thumb-remove";
    removeBtn.innerText = "✕";
    removeBtn.onclick = (e) => {
      e.stopPropagation();
      wardrobePools[category].splice(idx, 1);
      renderThumbnails(category);
    };

    thumbDiv.appendChild(img);
    thumbDiv.appendChild(removeBtn);
    strip.appendChild(thumbDiv);
  });
}

function clearAllPools() {
  ["top", "bottom", "footwear", "accessories"].forEach(cat => {
    wardrobePools[cat] = [];
    renderThumbnails(cat);
    const input = document.getElementById(`${cat}Input`);
    if (input) input.value = "";
  });
  document.getElementById("combosSection").style.display = "none";
}

async function loadMultiPreset(presetKey) {
  clearAllPools();
  const preset = MULTI_PRESETS[presetKey];
  if (!preset) return;

  for (const [category, pathArray] of Object.entries(preset)) {
    for (const path of pathArray) {
      try {
        const response = await fetch(path);
        const blob = await response.blob();
        const filename = path.split("/").pop();
        const file = new File([blob], filename, { type: "image/png" });
        wardrobePools[category].push(file);
      } catch (err) {
        console.error(`Error loading preset ${presetKey} ${category}:`, err);
      }
    }
    renderThumbnails(category);
  }

  runMixAndMatch();
}

async function runMixAndMatch() {
  if (wardrobePools.top.length < 1 || wardrobePools.bottom.length < 1) {
    alert("Please upload at least 1 Top Wear and 1 Bottom Wear item in your wardrobe pool!");
    return;
  }

  const mixBtn = document.getElementById("mixMatchBtn");
  const combosSection = document.getElementById("combosSection");

  mixBtn.disabled = true;
  mixBtn.innerHTML = `<span>Evaluating Wardrobe Permutations...</span>`;

  const formData = new FormData();

  ["top", "bottom", "footwear", "accessories"].forEach(cat => {
    wardrobePools[cat].forEach(fileItem => {
      formData.append(cat, fileItem);
    });
  });

  try {
    const response = await fetch("/mix_and_match", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.error || "Combinatorial pipeline error");
    }

    const data = await response.json();
    renderCombos(data);

  } catch (error) {
    console.error(error);
    alert(`Mix & Match Failed: ${error.message}`);
  } finally {
    mixBtn.disabled = false;
    mixBtn.innerHTML = `<span>Generate & Rank Outfit Combos ✨</span>`;
  }
}

function renderCombos(data) {
  const combosSection = document.getElementById("combosSection");
  const combosGrid = document.getElementById("combosGrid");
  const statsText = document.getElementById("summaryStatsText");

  combosSection.style.display = "block";
  combosSection.scrollIntoView({ behavior: 'smooth' });

  statsText.innerText = `Evaluated ${data.total_combos_generated} candidate combinations across your wardrobe pool. Here are the top ranked matches:`;
  combosGrid.innerHTML = "";

  data.top_outfits.forEach((outfit) => {
    const card = document.createElement("div");
    card.className = `combo-card ${outfit.rank === 1 ? 'rank-1' : ''}`;

    let badgeClass = "bronze";
    let badgeLabel = `#${outfit.rank} COMBINATION`;
    if (outfit.rank === 1) {
      badgeClass = "gold";
      badgeLabel = "🏆 #1 BEST MATCH";
    } else if (outfit.rank === 2) {
      badgeClass = "silver";
      badgeLabel = "🥈 #2 RUNNER UP";
    } else if (outfit.rank === 3) {
      badgeClass = "bronze";
      badgeLabel = "🥉 #3 BRONZE MATCH";
    }

    const itemsStripHtml = outfit.items.map(item => `
      <div class="item-badge">
        <div class="cat">${item.category}</div>
        <div class="swatch" style="background: ${item.hex}" title="${item.hex}"></div>
      </div>
    `).join("");

    card.innerHTML = `
      <span class="rank-tag ${badgeClass}">${badgeLabel}</span>
      
      <div class="combo-top">
        <div>
          <h4 style="font-family:var(--font-serif); font-size:1.2rem">Outfit Look #${outfit.combo_id}</h4>
          <span style="font-size:0.8rem; color:var(--text-muted)">Color Harmony: ${outfit.color_harmony}%</span>
        </div>
        <div class="score-val">${outfit.compatibility}%</div>
      </div>

      <div class="item-badge-strip">
        ${itemsStripHtml}
      </div>

      <p style="font-size:0.9rem; margin-bottom:0.75rem; font-weight:700; color:var(--accent-sunflower)">
        ${outfit.feedback_summary}
      </p>

      <div class="reason-box">
        ${outfit.comparison_reasoning}
      </div>
    `;

    combosGrid.appendChild(card);
  });
}
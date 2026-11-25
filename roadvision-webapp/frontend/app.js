(() => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const dropzoneEmpty = document.getElementById("dropzoneEmpty");
  const dropzonePreview = document.getElementById("dropzonePreview");
  const previewImg = document.getElementById("previewImg");
  const previewFileName = document.getElementById("previewFileName");
  const previewFileSize = document.getElementById("previewFileSize");
  const changeImgBtn = document.getElementById("changeImgBtn");
  const removeImgBtn = document.getElementById("removeImgBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");

  const confSlider = document.getElementById("confSlider");
  const confValue = document.getElementById("confValue");
  const presetBtns = document.querySelectorAll(".preset-btn");

  const errorBanner = document.getElementById("errorBanner");
  const loadingPanel = document.getElementById("loadingPanel");
  const loadingStatus = document.getElementById("loadingStatus");
  const loadingStepEls = document.querySelectorAll(".lstep");

  const resultsDashboard = document.getElementById("resultsDashboard");
  const metricTotal = document.getElementById("metricTotal");
  const metricClasses = document.getElementById("metricClasses");
  const metricLatency = document.getElementById("metricLatency");
  const metricModel = document.getElementById("metricModel");

  const tabResult = document.getElementById("tabResult");
  const tabOriginal = document.getElementById("tabOriginal");
  const resultImg = document.getElementById("resultImg");
  const downloadBtn = document.getElementById("downloadBtn");

  const distBars = document.getElementById("distBars");
  const anaTotal = document.getElementById("anaTotal");
  const anaUnique = document.getElementById("anaUnique");
  const anaAvgConf = document.getElementById("anaAvgConf");
  const anaTime = document.getElementById("anaTime");

  const detectionsTbody = document.getElementById("detectionsTbody");
  const tableCountNote = document.getElementById("tableCountNote");
  const samplesGrid = document.getElementById("samplesGrid");

  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");

  const CLASS_COLORS = {
    0: "#00D4FF",
    1: "#FF8C42",
    2: "#10B981",
    3: "#F43F5E",
  };

  let selectedFile = null;
  let lastResult = null;
  let activeView = "result";
  let loadingInterval = null;

  const LOADING_MESSAGES = [
    "Analyzing road surface...",
    "Running multi-view detection...",
    "Combining predictions...",
    "Preparing final results...",
  ];

  // Health polling
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();
      if (data.model_loaded) {
        statusDot.className = "status-dot online";
        statusText.textContent = `AI MODEL STATUS: ONLINE (${String(data.device).toUpperCase()})`;
      } else {
        statusDot.className = "status-dot offline";
        statusText.textContent = "AI MODEL STATUS: OFFLINE";
      }
    } catch {
      statusDot.className = "status-dot offline";
      statusText.textContent = "AI MODEL STATUS: OFFLINE";
    }
  }

  checkHealth();
  setInterval(checkHealth, 15000);

  // Load curated samples
  async function loadSamples() {
    try {
      const res = await fetch("/api/sample-images");
      if (!res.ok) return;
      const data = await res.json();
      const samples = data.samples || [];
      if (!samples.length) {
        samplesGrid.innerHTML = `<p class="muted-text">No sample images available. Please upload an image above.</p>`;
        return;
      }
      samplesGrid.innerHTML = samples
        .map(
          (s) => `
          <button type="button" class="sample-card" data-sample-id="${s.id}">
            <img src="${s.url}" alt="${s.title} sample" loading="lazy" />
            <div class="sample-body">
              <div class="sample-code">${s.class_code}</div>
              <div class="sample-title">${s.title}</div>
              <div class="sample-sub">${s.subtitle}</div>
            </div>
          </button>`
        )
        .join("");

      samplesGrid.querySelectorAll(".sample-card").forEach((card) => {
        card.addEventListener("click", () => {
          const sid = card.getAttribute("data-sample-id");
          runSampleInference(sid);
        });
      });
    } catch {
      samplesGrid.innerHTML = `<p class="muted-text">Unable to load sample images.</p>`;
    }
  }

  loadSamples();

  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.classList.remove("hidden");
  }

  function clearError() {
    errorBanner.textContent = "";
    errorBanner.classList.add("hidden");
  }

  function updateConfLabel(val) {
    const num = Number(val);
    const suffix = Math.abs(num - 0.237) < 0.005 ? " (F1-Optimal)" : "";
    confValue.textContent = `${num.toFixed(3).replace(/0$/, "")}${suffix}`;
    presetBtns.forEach((b) => {
      b.classList.toggle("active", Math.abs(Number(b.dataset.conf) - num) < 0.005);
    });
  }

  confSlider.addEventListener("input", () => updateConfLabel(confSlider.value));
  presetBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      confSlider.value = btn.dataset.conf;
      updateConfLabel(btn.dataset.conf);
    });
  });
  updateConfLabel(confSlider.value);

  function setSelectedFile(file) {
    clearError();
    if (!file) return;

    const validExts = [".jpg", ".jpeg", ".png"];
    const nameLower = (file.name || "").toLowerCase();
    const hasValidExt = validExts.some((ext) => nameLower.endsWith(ext));
    const hasValidMime = ["image/jpeg", "image/jpg", "image/png"].includes(file.type);

    if (!hasValidExt && !hasValidMime) {
      showError("Invalid file type. Please upload a JPG, JPEG, or PNG image.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showError("File size exceeds 10 MB limit.");
      return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewFileName.textContent = file.name;
      previewFileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
      dropzoneEmpty.classList.add("hidden");
      dropzonePreview.classList.remove("hidden");
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    previewImg.src = "";
    dropzonePreview.classList.add("hidden");
    dropzoneEmpty.classList.remove("hidden");
    analyzeBtn.disabled = true;
    clearError();
  }

  dropzone.addEventListener("click", (e) => {
    if (e.target === removeImgBtn || e.target === changeImgBtn) return;
    fileInput.click();
  });

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  changeImgBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  removeImgBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      setSelectedFile(fileInput.files[0]);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  });

  function startLoadingUI() {
    clearError();
    loadingPanel.classList.remove("hidden");
    analyzeBtn.disabled = true;
    let step = 0;
    const setStep = (idx) => {
      loadingStatus.textContent = LOADING_MESSAGES[idx];
      loadingStepEls.forEach((el, i) => el.classList.toggle("active", i <= idx));
    };
    setStep(0);
    loadingInterval = setInterval(() => {
      step = Math.min(step + 1, LOADING_MESSAGES.length - 1);
      setStep(step);
    }, 650);
  }

  function stopLoadingUI() {
    if (loadingInterval) {
      clearInterval(loadingInterval);
      loadingInterval = null;
    }
    loadingPanel.classList.add("hidden");
    analyzeBtn.disabled = !selectedFile;
  }

  async function runInference() {
    if (!selectedFile) return;
    startLoadingUI();
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const conf = Number(confSlider.value).toFixed(3);
      const res = await fetch(`/api/detect?conf_threshold=${conf}`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || data.detail || "Detection failed.");
      }
      renderResults(data);
    } catch (err) {
      showError(err.message || "Unable to reach backend inference service.");
    } finally {
      stopLoadingUI();
    }
  }

  async function runSampleInference(sampleId) {
    document.getElementById("detect").scrollIntoView({ behavior: "smooth" });
    startLoadingUI();
    try {
      const conf = Number(confSlider.value).toFixed(3);
      const res = await fetch(`/api/detect/sample/${encodeURIComponent(sampleId)}?conf_threshold=${conf}`, {
        method: "POST",
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || data.detail || "Sample detection failed.");
      }
      // Show sample in preview too
      previewImg.src = `data:image/jpeg;base64,${data.original_image_base64}`;
      previewFileName.textContent = `${sampleId}.jpg (Validation Sample)`;
      previewFileSize.textContent = `${data.image_size[0]} × ${data.image_size[1]} px`;
      dropzoneEmpty.classList.add("hidden");
      dropzonePreview.classList.remove("hidden");
      renderResults(data);
    } catch (err) {
      showError(err.message || "Unable to run sample inference.");
    } finally {
      stopLoadingUI();
    }
  }

  analyzeBtn.addEventListener("click", runInference);

  function renderResults(data) {
    lastResult = data;
    resultsDashboard.classList.remove("hidden");

    const analytics = data.analytics || {};
    metricTotal.textContent = data.total_detections;
    metricClasses.textContent = analytics.unique_classes ?? 0;
    metricLatency.textContent = `${data.inference_time_ms.toFixed(1)} ms`;
    metricModel.textContent = data.model_info?.name || "YOLOv8n + 3-View TTA + WBF";

    activeView = "result";
    updateViewerImage();

    // Distribution bars
    const dist = analytics.distribution || [];
    const maxCount = Math.max(1, ...dist.map((d) => d.count));
    distBars.innerHTML = dist
      .map((d) => {
        const pct = Math.round((d.count / maxCount) * 100);
        return `
          <div class="dist-row">
            <div class="dist-row-top">
              <span><strong>${d.class_code}</strong> ${d.damage_type}</span>
              <strong class="mono">${d.count}</strong>
            </div>
            <div class="dist-track">
              <div class="dist-fill" style="width:${pct}%; background:${d.color}"></div>
            </div>
          </div>`;
      })
      .join("");

    anaTotal.textContent = analytics.total_objects ?? data.total_detections;
    anaUnique.textContent = analytics.unique_classes ?? 0;
    anaAvgConf.textContent = `${((analytics.average_confidence || 0) * 100).toFixed(1)}%`;
    anaTime.textContent = `${data.inference_time_ms.toFixed(1)} ms`;

    // Table
    tableCountNote.textContent = `${data.total_detections} shown (conf ≥ ${data.conf_threshold.toFixed(3)}) • ${data.raw_detections_count} raw WBF candidates`;
    if (data.detections.length === 0) {
      detectionsTbody.innerHTML = `
        <tr>
          <td colspan="4" class="muted-text">No road damage detected above confidence threshold ${data.conf_threshold.toFixed(3)}.</td>
        </tr>`;
    } else {
      detectionsTbody.innerHTML = data.detections
        .map((det) => {
          const color = CLASS_COLORS[det.class_id] || "#00D4FF";
          const bboxStr = det.bbox.map((v) => Math.round(v)).join(", ");
          return `
            <tr>
              <td><span class="class-pill" style="background:${color}">${det.class_code}</span></td>
              <td>${det.damage_type}</td>
              <td class="mono">${(det.confidence * 100).toFixed(1)}%</td>
              <td class="mono">[${bboxStr}]</td>
            </tr>`;
        })
        .join("");
    }

    resultsDashboard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function updateViewerImage() {
    if (!lastResult) return;
    tabResult.classList.toggle("active", activeView === "result");
    tabOriginal.classList.toggle("active", activeView === "original");
    const b64 = activeView === "result" ? lastResult.annotated_image_base64 : lastResult.original_image_base64;
    resultImg.src = `data:image/jpeg;base64,${b64}`;
  }

  tabResult.addEventListener("click", () => {
    activeView = "result";
    updateViewerImage();
  });

  tabOriginal.addEventListener("click", () => {
    activeView = "original";
    updateViewerImage();
  });

  downloadBtn.addEventListener("click", () => {
    if (!lastResult || !lastResult.annotated_image_base64) return;
    const link = document.createElement("a");
    link.href = `data:image/jpeg;base64,${lastResult.annotated_image_base64}`;
    link.download = "roadvision_annotated_detection.jpg";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });
})();

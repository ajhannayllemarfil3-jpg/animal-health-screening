import { clipPrompts, clipDecide, BASELINE, validateFile, validateCard, scaleParams, qualityCheck, isEmergency, decide, EMERGENCY } from "./logic.js";
const $ = (s, r = document) => r.querySelector(s), app = $("#app");
const SYM = ["Loss of appetite", "Weakness or lethargy", "Coughing or sneezing", "Vomiting", "Diarrhea", "Skin changes or hair loss", "Eye or nose discharge", "Behavior change", ...EMERGENCY];
const KEY = "ahs_history_v1";
const hist = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } };
const save = (h) => { try { localStorage.setItem(KEY, JSON.stringify(h)); return true; } catch { return false; } };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const LBL = { prediction: "Model prediction (not a diagnosis)", inconclusive: "Inconclusive: low confidence", unavailable: "Disease model not installed", unsupported_species: "Species not supported by model", failed: "Analysis failed", clip: "Experimental visual-sign match (not a diagnosis)", clip_inconclusive: "Experimental screen: inconclusive" };

let card = null, cardState = "loading", model = null, st = { file: null, url: "", sym: {} };
async function loadCard() {
  try {
    const r = await fetch("model/model-card.json"); if (!r.ok) throw 0;
    const c = await r.json(), miss = validateCard(c);
    if (miss.length) { cardState = "Model card is missing: " + miss.join(", "); return; }
    card = c; cardState = "ok";
  } catch { cardState = "No model installed (model/model-card.json not found)."; }
}
async function loadModel() {
  if (model) return model;
  if (!window.tf) throw new Error("TensorFlow.js failed to load (offline?).");
  model = await (card.format === "graph" ? tf.loadGraphModel : tf.loadLayersModel)("model/model.json");
  return model;
}
const loadImg = (url) => new Promise((ok, no) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => no(new Error("Could not decode this image. It may be corrupt.")); i.src = url; });
function stats(img) {
  const c = document.createElement("canvas"); c.width = c.height = 64; const x = c.getContext("2d"); x.drawImage(img, 0, 0, 64, 64);
  const d = x.getImageData(0, 0, 64, 64).data, L = []; for (let i = 0; i < d.length; i += 4) L.push(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
  const m = L.reduce((a, b) => a + b, 0) / L.length; return { meanLum: m, stdLum: Math.sqrt(L.reduce((a, b) => a + (b - m) ** 2, 0) / L.length) };
}
function thumb(img) { const c = document.createElement("canvas"); c.width = 120; c.height = Math.round(120 * img.naturalHeight / img.naturalWidth) || 120; c.getContext("2d").drawImage(img, 0, 0, c.width, c.height); return c.toDataURL("image/jpeg", 0.6); }
let clip = null;
async function clipRun(species) {
  if (!clip) { const { pipeline } = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0"); clip = await pipeline("zero-shot-image-classification", "Xenova/clip-vit-base-patch32"); }
  const { prefix, prompts } = clipPrompts(species), out = await clip(st.url, prompts, { hypothesis_template: "{}" });
  return clipDecide(out, prefix);
}
async function infer(img) {
  const m = await loadModel(), { mul, add } = scaleParams(card.input.scale), s = card.input.size;
  const out = tf.tidy(() => { const t = tf.browser.fromPixels(img).resizeBilinear([s, s]).toFloat().mul(mul).add(add).expandDims(0); return m.predict(t); });
  const data = Array.from(await out.data()); out.dispose();
  if (data.length !== card.labels.length) throw new Error(`Model output size (${data.length}) does not match labels (${card.labels.length}).`);
  return data;
}

function view() {
  const h = location.hash.slice(1) || "/";
  if (h === "/dashboard") return dash(); if (h === "/history") return history(); if (h === "/about") return about();
  if (h.startsWith("/r/")) return report(h.slice(3)); return screen();
}
function screen() {
  app.innerHTML = `<h2>Screen an animal</h2><div class="al ${card ? "inf" : "err"}">${card ? `Model: <b>${esc(card.name)}</b> (${esc(card.license)}). Supports: ${esc(card.species.join(", "))}.` : "<b>No verified disease model installed.</b> The experimental CLIP visual-sign screen will be used instead (general model, not veterinary-trained, internet needed). " + esc(cardState)}</div>
  <div id="em"></div><div class="card"><h3>1. Photo</h3>
  <button class="alt" id="up">${st.file ? "Replace" : "Upload"} image</button><button class="alt" id="cam">Use camera</button>${st.file ? '<button class="bad" id="rm">Remove</button>' : ""}
  <input id="f" type="file" hidden accept="image/jpeg,image/png,image/webp"><div id="cv"></div><div id="msg"></div>${st.url ? `<img class="p" src="${st.url}" alt="Selected animal">` : ""}
  <label for="sp">Species</label><select id="sp"><option value="">Select…</option>${["Dog", "Cat", "Chicken", "Goat", "Cattle", "Pig", "Sheep", "Horse", "Other"].map((s) => `<option>${s}</option>`).join("")}</select></div>
  <div class="card"><h3>2. Symptoms you observed (separate from AI)</h3>${SYM.map((k) => `<label class="chk"><input type="checkbox" data-s="${esc(k)}" ${st.sym[k] ? "checked" : ""}>${esc(k)}</label>`).join("")}
  <label for="nt">Notes / duration</label><textarea id="nt" rows="2"></textarea></div>
  <button id="go" ${st.file ? "" : "disabled"}>Analyze image</button> <span id="busy"></span>`;
  $("#f").onchange = (e) => pick(e.target.files[0]); $("#up").onclick = () => $("#f").click(); $("#cam").onclick = camera;
  if ($("#rm")) $("#rm").onclick = () => { st.file = null; st.url = ""; screen(); };
  app.querySelectorAll("[data-s]").forEach((c) => (c.onchange = () => { st.sym[c.dataset.s] = c.checked; emg(); }));
  $("#go").onclick = analyze; emg();
}
function emg() { const el = $("#em"); if (el) el.innerHTML = isEmergency(Object.keys(st.sym).filter((k) => st.sym[k])) ? '<div class="al emg" role="alert">Possible emergency signs. Contact a veterinarian urgently now.</div>' : ""; }
function pick(f) {
  const e = validateFile(f); if (e) return ($("#msg").innerHTML = `<div class="al err" role="alert">${e}</div>`);
  if (st.url) URL.revokeObjectURL(st.url); st.file = f; st.url = URL.createObjectURL(f); screen();
}
async function camera() {
  const box = $("#cv");
  if (!navigator.mediaDevices?.getUserMedia) return ($("#f").setAttribute("capture", "environment"), $("#f").click());
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    box.innerHTML = '<video autoplay playsinline></video><br><button id="snap">Capture</button><button class="alt" id="x">Cancel</button>'; const v = $("video", box); v.srcObject = s;
    const stop = () => { s.getTracks().forEach((t) => t.stop()); box.innerHTML = ""; };
    $("#x").onclick = stop; $("#snap").onclick = () => { const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight; c.getContext("2d").drawImage(v, 0, 0);
      c.toBlob((b) => { stop(); pick(new File([b], "camera.jpg", { type: "image/jpeg" })); }, "image/jpeg", 0.92); };
  } catch (e) { $("#msg").innerHTML = `<div class="al err">${e.name === "NotAllowedError" ? "Camera permission was denied. Allow it in browser settings or upload a file." : "Camera unavailable on this device. Upload a file instead."}</div>`; }
}
let running = false;
async function analyze() {
  if (running || !st.file) return; running = true; $("#go").disabled = true; $("#busy").textContent = "Analyzing…";
  const species = $("#sp").value, reported = Object.keys(st.sym).filter((k) => st.sym[k]), notes = $("#nt").value;
  const rec = { id: Date.now().toString(36), ts: new Date().toISOString(), species, reported, notes, emergency: isEmergency(reported), warnings: [], top: null, status: "unavailable" };
  try {
    const img = await loadImg(st.url); rec.thumb = thumb(img); rec.warnings = qualityCheck({ width: img.naturalWidth, height: img.naturalHeight, ...stats(img) });
    if (!species) { $("#msg").innerHTML = '<div class="al err">Select the species first.</div>'; return; }
    let raw = null; if (card && card.species.map((s) => s.toLowerCase()).includes(species.toLowerCase())) raw = await infer(img);
    const d = decide({ card, species, rawOutput: raw }); rec.status = d.status; rec.top = d.top || null; rec.supported = d.supported || null;
    if (["unavailable", "unsupported_species"].includes(d.status)) { $("#busy").textContent = "Loading experimental model (first time downloads a large file)…"; const c = await clipRun(species); rec.status = c.status; rec.top = c.top; rec.baseline = c.baseline; }
  } catch (e) { rec.status = "failed"; rec.error = /fetch|import|network|load/i.test(e.message) ? "The experimental model could not load. Check your internet and try again." : e.message; }
  finally { running = false; }
  const h = hist(); h.unshift(rec); save(h); location.hash = "/r/" + rec.id;
}
function report(id) {
  const r = hist().find((x) => x.id === id); if (!r) return (app.innerHTML = "<p>Record not found.</p>");
  const pred = r.status === "prediction" || r.status === "inconclusive";
  app.innerHTML = `<div class="np"><button onclick="print()">Print report</button><a href="#/"><button class="alt">New screening</button></a><a href="#/history"><button class="alt">History</button></a></div>
  <h2>Screening report</h2><div class="card"><p>Ref ${esc(r.id)} · ${new Date(r.ts).toLocaleString()} · Species: ${esc(r.species)}</p>${r.thumb ? `<img class="p" src="${r.thumb}" alt="Uploaded animal (small preview)">` : ""}</div>
  ${r.emergency ? '<div class="al emg">Emergency signs were reported. Seek urgent veterinary care.</div>' : ""}
  <div class="card"><h3>AI result: ${LBL[r.status]}</h3>
  ${r.status === "unavailable" ? "<p>No verified disease model is installed, so no disease analysis was performed. Nothing here indicates the animal is healthy or sick.</p>" : ""}
  ${r.status === "unsupported_species" ? `<p>The installed model only covers: ${esc((r.supported || []).join(", "))}. No analysis was performed.</p>` : ""}
  ${["clip", "clip_inconclusive"].includes(r.status) ? `<p>General-purpose CLIP model (Xenova/clip-vit-base-patch32), not veterinary-trained. Relative match among a fixed list of visible-sign descriptions. These are <b>not</b> disease probabilities:</p>${r.top.map((t) => `<p>${esc(t.label)}: ${(t.score * 100).toFixed(1)}%<span class="bar"><i style="width:${t.score * 100}%"></i></span></p>`).join("")}<div class="al ${r.status === "clip" ? "inf" : "err"}">${r.status === "clip_inconclusive" ? "No description matched clearly. Inconclusive. " : ""}${r.baseline ? "The closest match was \"no visible abnormality\", which does NOT mean the animal is healthy. " : ""}It can miss real problems and flag normal animals. Only a veterinarian can diagnose.</div>` : ""}
  ${r.status === "failed" ? `<p>${esc(r.error)}</p>` : ""}
  ${pred ? `<p>Model scores (softmax output of "${esc(card?.name || "model")}"; <b>not</b> a clinical probability):</p>${r.top.map((t) => `<p>${esc(t.label)}: ${(t.score * 100).toFixed(1)}%<span class="bar"><i style="width:${t.score * 100}%"></i></span></p>`).join("")}
  <div class="al ${r.status === "inconclusive" ? "err" : "inf"}">${r.status === "inconclusive" ? "Top score is below the confidence threshold. Treat as inconclusive. " : ""}Model scores can be wrong, especially for photos unlike its training data. Not a diagnosis. A normal-looking result never proves an animal is healthy.</div>` : ""}
  ${r.warnings.length ? `<p><b>Image quality notes:</b></p><ul>${r.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}</div>
  <div class="card"><h3>Symptoms reported by user (not AI)</h3>${r.reported.length ? `<ul>${r.reported.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>` : "<p>None selected.</p>"}<p>${esc(r.notes)}</p></div>
  <div class="al inf">Next step: a veterinary physical exam, history and laboratory tests are needed to confirm any disease. Do not start medication based on this screening.</div>`;
}
function dash() {
  const h = hist(), c = (f) => h.filter(f).length, sp = {}; h.forEach((x) => (sp[x.species] = (sp[x.species] || 0) + 1));
  app.innerHTML = `<h2>Dashboard</h2><div class="grid">${[["Screenings", h.length], ["Model / experimental results", c((x) => ["prediction", "clip"].includes(x.status))], ["Inconclusive", c((x) => ["inconclusive", "clip_inconclusive"].includes(x.status))], ["No model / unsupported", c((x) => ["unavailable", "unsupported_species"].includes(x.status))], ["Failed", c((x) => x.status === "failed")]].map(([l, n]) => `<div class="card"><span class="n">${n}</span>${l}</div>`).join("")}</div>
  <div class="card"><h3>Species</h3>${Object.keys(sp).length ? Object.entries(sp).map(([k, v]) => `<p>${esc(k)}: ${v}</p>`).join("") : "<p>No data yet.</p>"}</div>`;
}
function history() {
  const draw = () => {
    const q = $("#q").value.toLowerCase(), s = $("#fs").value, h = hist().filter((x) => x.species.toLowerCase().includes(q) && (!s || x.status === s));
    $("#rows").innerHTML = h.map((x) => `<tr><td>${new Date(x.ts).toLocaleString()}</td><td>${esc(x.species)}</td><td>${LBL[x.status]}</td><td><a href="#/r/${x.id}">View</a> <button class="bad" data-d="${x.id}">Delete</button></td></tr>`).join("") || '<tr><td colspan="4">No records.</td></tr>';
    app.querySelectorAll("[data-d]").forEach((b) => (b.onclick = () => { if (confirm("Delete this record?")) { save(hist().filter((x) => x.id !== b.dataset.d)); draw(); } }));
  };
  app.innerHTML = `<h2>History</h2><div class="card"><input id="q" placeholder="Search species" aria-label="Search species"><select id="fs" aria-label="Status"><option value="">All results</option>${Object.entries(LBL).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select>
  <button class="bad" id="clr">Clear all</button></div><div class="card t"><table><thead><tr><th>Date</th><th>Species</th><th>Result</th><th></th></tr></thead><tbody id="rows"></tbody></table></div>`;
  $("#q").oninput = $("#fs").onchange = draw; $("#clr").onclick = () => { if (confirm("Delete ALL history?")) { save([]); draw(); } }; draw();
}
function about() {
  app.innerHTML = `<h2>Model and privacy</h2><div class="card"><p><b>Status:</b> ${esc(cardState === "ok" ? "Installed" : cardState)}</p>${card ? `<p>${esc(card.name)} · ${esc(card.license)} · <a href="${esc(card.source)}">source</a><br>Training data: ${esc(card.trainingData)}<br>Classes: ${esc(card.labels.join(", "))}</p>` : "<p>See README for how to install a model.</p>"}</div>
  <div class="card"><h3>Privacy</h3><p>Images are processed in your browser and never uploaded. History (with a tiny preview) is stored only in this browser's localStorage. Only TensorFlow.js is downloaded from a CDN.</p></div>`;
}
addEventListener("hashchange", view); loadCard().then(view);

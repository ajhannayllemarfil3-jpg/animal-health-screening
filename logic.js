// Pure logic (no DOM / no TensorFlow) so it can be unit-tested in Node.
export const MAX_BYTES = 8 * 1024 * 1024;
export const TYPES = ["image/jpeg", "image/png", "image/webp"];
export const MIN_CONF = 0.7;
export const EMERGENCY = ["Difficulty breathing", "Collapse or cannot stand", "Seizures", "Suspected poisoning", "Heavy bleeding"];

export function validateFile(f) {
  if (!f) return "No file selected.";
  if (!TYPES.includes(f.type)) return "Unsupported file type. Use JPG, PNG or WebP.";
  if (f.size === 0) return "The file is empty.";
  if (f.size > MAX_BYTES) return "Image is larger than 8 MB.";
  return null;
}
export function validateCard(c) {
  const miss = [];
  if (!c || typeof c !== "object") return ["model-card.json"];
  for (const k of ["name", "source", "license", "format", "trainingData"]) if (!c[k] || typeof c[k] !== "string") miss.push(k);
  if (!c.input || !(c.input.size > 0) || !["0-1", "-1-1", "0-255"].includes(c.input.scale)) miss.push("input.size/scale");
  if (!["probabilities", "logits"].includes(c.outputs)) miss.push("outputs");
  if (!["graph", "layers"].includes(c.format)) miss.push("format(graph|layers)");
  if (!Array.isArray(c.labels) || !c.labels.length) miss.push("labels");
  if (!Array.isArray(c.species) || !c.species.length) miss.push("species");
  return miss;
}
export function scaleParams(mode) {
  return { "0-1": { mul: 1 / 255, add: 0 }, "-1-1": { mul: 1 / 127.5, add: -1 }, "0-255": { mul: 1, add: 0 } }[mode];
}
export function softmax(a) { const m = Math.max(...a), e = a.map((x) => Math.exp(x - m)), s = e.reduce((p, c) => p + c, 0); return e.map((x) => x / s); }
export function topK(scores, labels, k = 3) {
  return scores.map((s, i) => ({ label: labels[i] ?? `class ${i}`, score: s })).sort((a, b) => b.score - a.score).slice(0, k);
}
export function qualityCheck({ width, height, meanLum, stdLum }) {
  const w = [];
  if (Math.min(width, height) < 224) w.push("Image is small (under 224 px); results may be unreliable.");
  if (meanLum < 40) w.push("Image is very dark.");
  if (meanLum > 225) w.push("Image is overexposed.");
  if (stdLum < 12) w.push("Image has very little detail (blurry or blank?).");
  return w;
}
export function isEmergency(reported) { return (reported || []).some((s) => EMERGENCY.includes(s)); }
// status: unavailable | unsupported_species | inconclusive | prediction
export function decide({ card, species, rawOutput }) {
  if (!card || validateCard(card).length) return { status: "unavailable" };
  if (!species) return { status: "need_species" };
  if (!card.species.map((s) => s.toLowerCase()).includes(species.toLowerCase())) return { status: "unsupported_species", supported: card.species };
  const scores = card.outputs === "logits" ? softmax(rawOutput) : rawOutput;
  const top = topK(scores, card.labels, 3);
  return { status: top[0].score >= (card.minConfidence ?? MIN_CONF) ? "prediction" : "inconclusive", top };
}
// Experimental CLIP visual-sign prompts (general model, NOT veterinary-trained)
export const SIGNS = ["with no visible abnormality", "with skin lesions, sores or scabs", "with hair loss or bald patches", "with swollen lumps or nodules on the skin", "with eye or nasal discharge", "that looks very thin or underweight", "that looks lethargic or lying down"];
export const BASELINE = SIGNS[0];
export function clipPrompts(sp) { const n = (sp || "animal").toLowerCase() === "other" ? "animal" : sp.toLowerCase(); const prefix = `a photo of a ${n} `; return { prefix, prompts: SIGNS.map((s) => prefix + s) }; }
export function clipDecide(out, prefix, min = 0.5) {
  const top = out.slice().sort((a, b) => b.score - a.score).slice(0, 3).map((o) => ({ label: o.label.replace(prefix, ""), score: o.score }));
  return { status: top[0].score >= min ? "clip" : "clip_inconclusive", top, baseline: top[0].label === BASELINE };
}

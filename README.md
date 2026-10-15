# Animal Health Screening (browser-only, GitHub Pages ready)

HTML + CSS + JavaScript + TensorFlow.js. No backend, no API key. Images never leave the browser.

## Experimental CLIP mode (added)
When no verified disease model is installed, the app uses `Xenova/clip-vit-base-patch32` (ONNX port of OpenAI CLIP, loaded in-browser via Transformers.js 3.0.0 from jsDelivr). It scores the photo against fixed visible-sign descriptions (skin lesions, hair loss, lumps, discharge, thin, lethargic). **Not veterinary-trained, not a diagnosis.** Check the license on the model page. First run downloads a large model and needs internet.
**Not tested here:** model download and real inference (no network in the build sandbox). Test in your browser.

## Model research result (checked Oct 2026)
**No verified, ready-to-use, free, browser-compatible, multi-species animal disease model was found, so none is bundled.** Nothing was invented.
- Only candidate found: `SrimathiE21ALR044/Cattle_Skin_Disease` (Hugging Face, ViT, cattle skin disease). Problems: PyTorch weights only (not TF.js), no license shown on its page, class list not on the model card, and a self-reported accuracy of 1.000 (suspiciously perfect; likely overfit or leaked test data). I could not download or test it. Use only after you verify its classes and license and convert it.
- General classifiers (MobileNet/ImageNet, cat-vs-dog demos) are NOT disease models and are deliberately not used.

**Trade-off:** the app ships with the model slot empty. Without a model it records photos and symptoms and says plainly that no disease analysis happened. It will not show disease results until a complete `model/model-card.json` exists.

## Install a model (when you have a verified one)
1. Convert to TF.js (`pip install tensorflowjs`, `tensorflowjs_converter ...`) and put `model.json` + shard files in `model/`.
2. Copy `model/model-card.example.json` to `model/model-card.json` and fill every field from the author's documentation (name, source, license, training data, species, labels in output order, input size/scale).
3. Anything missing, or species not covered, disables predictions. Scores below `minConfidence` (default 0.7) show as inconclusive.

## Run locally
`python -m http.server 8000` in this folder, open http://localhost:8000 (camera needs localhost or HTTPS).
Tests: `node --test tests/logic.test.mjs`

## Publish on GitHub Pages
1. Create a GitHub repo, upload all files (keep `index.html` at the root).
2. Settings > Pages > Deploy from branch > `main` / root.
3. Open the URL shown. Check the About page shows your model status. Confirm the license permits redistribution of the weights.

## Tests actually performed (in the build sandbox)
Executed and passed (16): file validation, model-card validation, preprocessing parameters, softmax, decision logic (no model, bad card, unsupported species, high/low confidence), image quality warnings, emergency detection. JavaScript syntax check of app.js passed.
**Not performed (no network/weights/browser):** real model loading, real inference, camera, UI in a browser, GitHub Pages deployment, mobile layout. Test these yourself.

## Limitations
Screening aid only, not a diagnosis. Never reports "healthy". The original brief ended mid-sentence ("Make system..."); tell me what was meant.

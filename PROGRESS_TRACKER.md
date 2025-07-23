# 🧪 Real-Time Raman Spectral Classifier - Progress Tracker

This file tracks the development progress and todos of the research project:  
**疾患検出のためのリアルタイム・ラマン分光分類ソフトウェアの開発**

---

## 🗂️ Icon Legend

| Icon     | Meaning                     |
|----------|-----------------------------|
| ✅       | Task Completed              |
| 🟡       | In Progress                 |
| ❌       | Not Started / Canceled      |
| 📅       | Date                        |
| 🧠       | Notes or observations       |

---

## 1. 📊 Algorithm Testing

| Task | Status | 📅 Date | 🧠 Notes |
|------|--------|---------|----------|
| Implement baseline LDA | ✅ | 2025-06-02 | Good baseline accuracy (~70%) |
| SVM + PCA | 🟡 | 2025-06-08 | Tune `C`, `gamma`, try RBF |
| 1D-CNN v1 (raw) | ❌ | — | Start after SVM done |
| Compare CPU vs GPU runtime | ❌ | — | Needed for optimization |
| Noise-augmented test | ❌ | — | Use noise1.csv–noise3.csv |

---

## 2. 🧩 Function Development

| Function | Status | 📅 Date | 🧠 Notes |
|----------|--------|---------|----------|
| `baseline_correction()` | ✅ | 2025-05-20 | Using ALS (airPLS alternative) |
| `load_model(path)` | 🟡 | 2025-06-01 | Must support .pt / .onnx |
| `infer_spectrum()` | ❌ | — | Real-time wrapper for models |
| `plot_result()` | 🟡 | 2025-06-10 | Matplotlib or PyQtGraph |
| `export_result()` | ❌ | — | Export to CSV/log file |

---

## 3. 🖥️ GUI Creation

| Component | Status | 📅 Date | 🧠 Notes |
|-----------|--------|---------|----------|
| Main Window | 🟡 | 2025-06-15 | DearPyGUI or PyQt5 |
| Drag & Drop (spectrum) | ❌ | — | Accept .csv/.txt |
| Model Loader UI | ❌ | — | Plug-and-play design |
| Result Display (graph) | ❌ | — | Overlay prediction label |
| Settings Tab | ❌ | — | Allow algorithm switching |

---

## 4. 🧪 App Testing / Validation

| Test | Status | 📅 Date | 🧠 Notes |
|------|--------|---------|----------|
| Case-wise LOOCV | ❌ | — | Required for paper section |
| Runtime Benchmarks | ❌ | — | Target: <500ms inference |
| Smoothing + noise combo | ❌ | — | Test stability under noise |
| Sample GUI testing | ❌ | — | Simulate .csv import & classify |

---

## 5. 🚀 Deployment / Packaging

| Task | Status | 📅 Date | 🧠 Notes |
|------|--------|---------|----------|
| PyInstaller `.exe` build | ❌ | — | For Windows GUI distribution |
| ONNX export | ❌ | — | Model conversion for CPU |
| Create installer (NSIS) | ❌ | — | Add GUI, model files, readme |
| Upload to GitHub Releases | ❌ | — | Tagged `v0.1-beta` |
| Final GUI packaging (Win/Linux) | ❌ | — | Add documentation |

---
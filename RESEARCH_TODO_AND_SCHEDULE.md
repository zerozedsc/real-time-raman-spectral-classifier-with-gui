# 📅 Research Schedule & Weekly TODO Tracker
*Project:* **Real-Time Raman Spectral Classifier Software for Disease Detection**
*Start Date:* Mid April 2025  
*Current Date:* 2025-04-25  
*Focus:* Core functions first — data loading, preprocessing, ML pipelines  
*GUI will begin in October*

---

## 🔧 Weekly TODOs

### ✅ Week of May 20
- [x] Fix Kaggle loader function to support flatten + merged modes
- [ ] Setup FFT smoothing test function for noisy Raman spectrum
- [ ] Start assembling Hirschsprung + Kaggle unified test set

### 📌 Week of May 27
- [ ] Finish `train_on_case12_test_on_case3()` ML wrapper
- [ ] Write function to extract best region from grid search
- [ ] Begin exporting confusion matrix and classification report as `.png` or `.csv`
- [ ] Create baseline PCA visualizer (label + chemistry)

---

## 📆 Monthly Schedule Overview

### 🗓 May 2025 – Setup + Loaders
- [x] Load Hirschsprung (case1-3), Kaggle, assign labels
- [x] Build preprocessing pipeline (ASPLS, SavGol, Vector)


### 🗓 June – ML + Evaluation
- [ ] Train SVM on case1+2, test on case3
- [ ] Region-wise grid search per disease
- [ ] Score analysis (accuracy, precision, recall)
- [ ] Save best model per region/dataset

### 🗓 July – PCA, SHAP, Interpretability
- [ ] Plot PCA (colored by label/chemistry)
- [ ] Endmember extraction (NFINDR)
- [ ] Add SHAP for SVM feature importance

### 🗓 August – Real-time Packaging
- [ ] Build `predict_one_spectrum()` wrapper
- [ ] Evaluate model on Surface Pro (CPU only)
- [ ] Start CLI tool for single spectrum test

### 🗓 October – GUI Phase
- [ ] Choose frontend framework: Tauri or Streamlit
- [ ] Add spectrum input + file drop support
- [ ] Display spectrum, baseline, prediction result

---

## 📦 Folder Structure
```
real-time-raman-spectrum-classifier-with-gui/
┣ data/              # case1-3, kaggle, rod
┣ scripts/           # preprocessing, ML models, visualizers
┣ results/           # PCA plots, confusion matrices
┣ gui/               # GUI code (starts Oct)
┣ paper/             # Thesis doc + images
┣ README.md
```

---


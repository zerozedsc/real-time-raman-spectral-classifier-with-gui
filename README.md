# Real-Time Raman Spectral Classifier for Disease Detection

> 疾患検出のためのリアルタイム・ラマン分光分類ソフトウェアの開発

This repository contains the official implementation of a BSc graduation research project conducted at the **Clinical Photonic Information Engineering Laboratory (臨床光情報工学研究室), University of Toyama**. The project explores real-time Raman spectroscopy analysis using machine learning and deep learning models for disease detection.

## 📚 Project Overview

The main objective of this research is to build a modular and CPU-optimised software system capable of analysing Raman spectral data in real-time and classifying biological tissue states (e.g., diseased or normal) using pre-trained ML/DL models.

### 🔬 Key Features
- Support for traditional ML (SVM, LDA, Logistic Regression) and deep learning (1d-CNN)
- Modular plug-and-play architecture for loading disease-specific models
- Preprocessing tools: baseline correction, normalisation, smoothing
- GUI prototype with drag-and-drop interface (PyQt5, DearPyGUI, or other)
- ONNX-compatible for deployment on CPU or GPU

---

## 🧠 Models & Algorithms

| Model                  | Purpose                       | CPU-Friendly | Description |
|-----------------------|-------------------------------|--------------|-------------|
| LDA / Logistic Reg.   | Baseline classifiers          | ✅            | Lightweight, interpretable models |
| SVM                   | Accurate nonlinear classifier | ✅ (linear)   | Uses kernel tricks for complex data |
| 1D-CNN                | Deep learning-based method    | ⚠️ Yes (with quantization) | Learns features from raw spectra |
| PCA                   | Feature compression           | ✅            | Improves model performance |

---

## 📁 Repository Structure


---

## 💻 How to Run

### Requirements
- Python 3.12+
- PyTorch / scikit-learn / onnxruntime
- PyQt5 / matplotlib / numpy / scipy

### Example (command line inference):


### Example (GUI mode):


---

## 🧪 Dataset

The data used in this study comes from **intestinal Raman spectra** from animal models and patients (e.g., Hirschsprung’s disease cases), divided into:
- `normal/` – control tissues
- `disease/` – diseased tissues
- `noiseX.csv` – for data augmentation testing

> ✳️ The full dataset is not publicly released, and the dataset used is based on the University of Toyama Database.

---

## 📖 Research Information

- 🎓 **Student:** Muhammad Helmi, 4th Year BSc, Toyama University
- 🧪 **Lab:** Clinical Photonic Information Engineering Lab (臨床光情報工学研究室)
- 📅 **Period:** May 2025 – February 2026
- 🧠 **Keywords:** Raman spectroscopy, machine learning, deep learning, pre-disease detection, 1d-CNN, real-time diagnosis

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

> ⚠️ For any commercial or clinical use, please contact the lab advisor or author.

---

## 🙏 Acknowledgments

- Supervisor: Prof. [大嶋　佑介], University of Toyama
- Thanks to my lab senpai for dataset contributions and prior work inspiration
- Supported by Toyama University Engineering Faculty

---

## 🔗 Related Work
- [SHAP](https://github.com/slundberg/shap) – Explainable ML for interpretability
- [SpectralTransformer](https://arxiv.org/abs/2304.01427) – Transformer for spectral data
- [Raman spectroscopy in biomedical research](https://www.nature.com/articles/s41551-019-0379-6)

---

## 📬 Contact
If you have questions or wish to collaborate:
- Email: [s2270294@ems.u-toyama.ac.jp]
- GitHub: [zerozedsc]

---

Made with 💡 and 🧪 in Toyama

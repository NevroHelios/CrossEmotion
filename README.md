# Cross-Dataset Emotion Recognition Benchmark (MELD vs IFEED)

**Author**: NevroHelios 
**Last Updated**: 07 June 2025

## 🎯 Objective
Compare video (MELD) and image (IFEED) emotion recognition performance through:
1. Modality-specific baseline models
2. Cross-dataset transfer learning experiments
3. Character-wise analysis (Friends TV cast)

## 📥 Data Preparation

### MELD Dataset (Video)
```
# Download from official source
git clone https://github.com/declare-lab/MELD.git
mv MELD/data/MELD.Raw data/meld_raw
```

### IFEED Dataset (Images)
```
# Request dataset from original paper authors
wget researchlab2.iiit.ac.in/ifeed/IFEED_170x140_v3.tar.gz
tar -xzf IFEED_170x140_v3.tar.gz -C data/ifeed_raw
```

## 🛠️ Installation
```
conda create -n emotion python=3.12
conda activate emotion
pip install -r requirements.txt  
```



### Preliminary MELD Benchmarks
| Model | Val Accuracy | F1-Score | Inference Speed |
|-------|--------------|----------|-----------------|
| CNN-LSTM (Baseline) | 63.2% | 0.61 | 87ms/video |
| Custom 3D-CNN | 67.8% | 0.65 | 104ms/video |

## 🚧 Future Work
- [ ] **IFEED Integration**
  - [x] Adapt dataloader for 170x140px images
  - [ ] Implement ResNet-50 baseline
    - [ ] Training pipeline
    - [ ] Inference pipeline
  - [ ] Benchmark ResNet-50 on IFEED

- [ ] **MELD Improvements**
  - [ ] Train Custom 3D-CNN
  - [ ] Inference with Custom 3D-CNN
  - [ ] Benchmark 3D-CNN on MELD

- [ ] **Multimodal Fusion**
  - [ ] Implement FusionModel (image + text)
    - [ ] Integrate ResNet-50 image encoder
    - [ ] Integrate BERT text encoder
    - [ ] Add attention-based fusion layer
  - [ ] Train FusionModel
  - [ ] Inference with FusionModel
  - [ ] Benchmark FusionModel

- [ ] **Deployment**
  - [ ] Web demo using Gradio
  - [ ] SaaS API endpoint (FastAPI)

## 📊 Expected Benchmarks
| Metric | MELD Target | IFEED Target |
|--------|-------------|--------------|
| Accuracy | 72% | 85% |
| F1-Score | 0.71 | 0.87 |
| Inference Speed |  **Note**: Dataset licenses 
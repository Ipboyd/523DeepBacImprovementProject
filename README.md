# 🧬 **Computer Vision for Bacteria Counting**

## 🚀 **Introduction**  
This project is an extension of the **2022 DeepBacs paper**, focusing on enhancing **cell recall** in dense, high-contrast regions. Our approach combines advanced deep learning methodologies to address the challenges posed in densely populated, high-contrast images.

### Key Highlights:
- **Data**: Available in the `Data` folder on SCC, labeled according to figures in the original DeepBacs paper.  
- **Approach**:
  - Pre-task training with a **Convolutional Neural Network (CNN)** to extract feature representations.
  - A **Transformer-based segmentation model** that utilizes the pre-task CNN features and the original grayscale image to produce precise segmentations.
- **Goal**: Improve upon **StarDist2D's generalist model** and the methodologies in DeepBacs for segmenting cells in challenging environments.

---

## 🛠️ **Setup Instructions**

### 1. **Dependencies**  
Ensure the following libraries are installed:  
```bash
!pip install numpy matplotlib random scikit-image torch pillow einops opencv-python
```
Alternatively, list them in a requirements file:
```bash
pip install -r requirements.txt
```

### 2. **Primary Results**  
To reproduce our results, follow these steps:  
- **Run the Transformer-CNN pipeline**:
   ```bash
   jupyter notebook TransformerCNN_masks.ipynb
   ```
   - **Input**: Raw grayscale images of *S. Aureus* dataset (available at [Zenodo](https://doi.org/10.5281/zenodo.5550933)).  
   - **Output**: 
     - Individual instance masks for each bacterium.  
     - Combined binary masks (sum across all instance masks).  

- **Augmented Dataset Analysis**:  
   ```bash
   jupyter notebook Augmented_training_pipeline.ipynb
   ```

### 3. **Output Example**  
Below is an example of the segmentation output, showcasing individual bacterial masks:

![Segmentation Example](https://github.com/user-attachments/assets/b743e9aa-56f0-4233-bd0b-d6e2f4aae666)

---

## 📊 **Work Log**

| **Date**       | **Task**                                                                                          | **Contributor(s)**       |
|-----------------|--------------------------------------------------------------------------------------------------|--------------------------|
| 10/10/2024     | Uploaded StarDist demo for initial analysis.                                                     | Isaac                    |
| 11/19/2024     | Added Transformer framework class.                                                               | Omar                     |
| 11/20/2024     | Integrated StarDist2D generalist model and updated the `Data` folder.                            | Isaac                    |
| 11/20/2024     | Added CNN framework class.                                                                       | Mohi                     |
| 12/06/2024     | Integrated pre-trained ResNet for CNN features.                                                  | Mohi / Berk              |
| 12/07/2024     | Initial version of the Transformer-based model.                                                  | Isaac / Omar             |
| 12/07 → 12/15  | Developed multiple Transformer model variations.                                                 | Everyone              |
| 12/15/2024     | Finalized model selection, cleaned GitHub structure, and organized main notebooks into folders.  | Everyone                 |

---

## 🧪 **Results Summary**  
- **Target Metric**: Improved recall compared to StarDist2D.  
- **Approach**: Transformer + CNN pre-task framework.  
- **Key Focus**: Dense, high-contrast regions with challenging segmentation cases.  

---

🌟 **For any questions, please feel free to contact our team or refer to the detailed notebooks in the `Notebooks` folder.**

---

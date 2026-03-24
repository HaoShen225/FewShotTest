# Some Ideas 

## Benchmark
Few-shot segmentation (FSS) paradigm method: use one DINOv2 encoder fine-tuned on other dataset (ACDC/AMOS) with self-supervised training to encode the
SPIDER bone MRI set. Identify the possible foreground by compairing the semantical similarity between query set feature and foreground feature.

Replicate this: L. Ayzenberg, R. Giryes and H. Greenspan, "DINOv2 Based Self Supervised Learning for Few Shot Medical Image Segmentation," 2024 IEEE International Symposium on Biomedical Imaging (ISBI), Athens, Greece, 2024, pp. 1-5, doi: 10.1109/ISBI56570.2024.10635439.

Challenges: domain shift, uncontinuous prediction mask, optimal transmission with spatial constraint.

## Tasks
I. In the first stage, 
## Proposed Method 1
Dictionary-learning-based sparse encoding: Use 2 layers of dictionaries to encode the input image in two abstract degrees, appearance encoding and structural encoding. Finally, foreground/background decision is made on the structural sparce code. The encoding process can be represented as followed:

I. Appearance encoding:

```math
\min_{D_1,\{\mathbf{w}_{1,i}\}}
\sum_i \|\mathbf{x}_i - \mathbf{D}_1 \mathbf{w}_{1,i}\|_2^2 + \lambda_1 \|\mathbf{w}_{1,i}\|_1
```

This step aims to learn a dictionary used to encode the local patch appearance into sparce coding. We expect that each kind of appearance can be represented by a sparce code. The first term is the reconstruction error, which enforces the dictionary, $D_{1}$, to learn the patterns on the image, and the second term enforces the dictionary to represent each appearance sparsely.

II. Structural Ensambling  
This step aims to represent structure information with the sparec code combination from the first step. Assume that the $j^{th}$ sensing field is $\\{(x_{j},y_{j})|x\in \Omega_{x,j},y\in \Omega_{y,j}\\}$, the local scructure code is $\tilde{\mathbf{w}}_{1,j}=[\mathbf{w}_{1,j}]$ (stack all the sparce code in the field).  

III. Structural Encoding  
This step aims to learn a dictionary used to encode the structures into sparce coding. We expect to represent each special structure in human body into a sparce vector. This code enables structural-level segmentation. We expect the sparce code to be discriminative, so a linear segmentation head is added.

```math
\min_{D_2,\{\mathbf{w}_{2,j}\}}
\sum_j \|\tilde{\mathbf{w}}_{1,j} - \mathbf{D}_2 \mathbf{w}_{2,j}\|_2^2 + \lambda_1 \|\mathbf{w}_{2,j}\|_1 + \mu crossEntropy(y_{j}-\mathbf{d}\mathbf{w}_{2,j})
```

# Task: Train a model from beginning with a few shots (not necessarily 5, can be 5-20), avoid over-fitting and domain collapse.
Data cleaning code: SPIDER_Data_Cleaning_and_Preprocessor.py

# Literature Review
# Rethinking Few-Shot Medical Segmentation: A Vector Quantization View
This study reformulates prototype learning in few-shot medical image segmentation as a vector quantization problem, and proposes a three-stage mechanism: Grid Feature Vector Quantization (GFVQ) for prototype generation via grid average pooling, Self-Organizing Vector Quantization (SOVQ) for self-organizing clustering and local mapping, and Residual Offset Vector Quantization (ROVQ) for residual fine-tuning of prototypes during inference.  
Problems:  
It does not explicitly address the problems of hard negatives and feature inseparability in the foreground feature space.  
S. Huang, T. Xu, N. Shen, F. Mu and J. Li, "Rethinking Few-Shot Medical Segmentation: A Vector Quantization View," 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, BC, Canada, 2023, pp. 3072-3081, doi: 10.1109/CVPR52729.2023.00300. keywords: {Training;Extraterrestrial phenomena;Vector quantization;Magnetic resonance imaging;Prototypes;Feature extraction;Pattern recognition;Medical and biological vision;cell microscopy},
# PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment



# Method 1 (Evaluate Now)
Few-shot segmentation (FSS) paradigm method: use one DINOv2 encoder fine-tuned on other dataset (ACDC/AMOS) with self-supervised training to encode the
SPIDER bone MRI set. Identify the possible foreground by compairing the semantical similarity between query set feature and foreground feature.
Replicate this: L. Ayzenberg, R. Giryes and H. Greenspan, "DINOv2 Based Self Supervised Learning for Few Shot Medical Image Segmentation," 2024 IEEE International Symposium on Biomedical Imaging (ISBI), Athens, Greece, 2024, pp. 1-5, doi: 10.1109/ISBI56570.2024.10635439.

Possible challenges: domain shift, uncontinuous prediction mask, optimal transmission with spatial constraint.

# Method 2 (Under Experiment)
Dictionary-learning-based sparse encoding: Use 2 layers of dictionaries to encode the input image in two abstract degrees, appearance encoding and structural encoding. Finally, foreground/background decision is made on the structural sparce code. The encoding process can be represented as followed:  
I. Appearance encoding: 

```math
\min_{D_1,\{\mathbf{w}_{1,i}\}}
\sum_i \|\mathbf{x}_i - \mathbf{D}_1 \mathbf{w}_{1,i}\|_2^2 + \lambda_1 \|\mathbf{w}_{1,i}\|_1

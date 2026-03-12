# Literature Review
# Rethinking Few-Shot Medical Segmentation: A Vector Quantization View
This study reformulates prototype learning in few-shot medical image segmentation as a vector quantization problem, and proposes a three-stage mechanism: Grid Feature Vector Quantization (GFVQ) for prototype generation via grid average pooling, Self-Organizing Vector Quantization (SOVQ) for self-organizing clustering and local mapping, and Residual Offset Vector Quantization (ROVQ) for residual fine-tuning of prototypes during inference.
Problems:  
It does not explicitly address the problems of hard negatives and feature inseparability in the foreground feature space.


# Method 1 (Evaluate Now)
Few-shot segmentation (FSS) paradigm method: use one DINOv2 encoder fine-tuned on other dataset (ACDC/AMOS) with self-supervised training to encode the
SPIDER bone MRI set. Identify the possible foreground by compairing the semantical similarity between query set feature and foreground feature.
Replicate this: L. Ayzenberg, R. Giryes and H. Greenspan, "DINOv2 Based Self Supervised Learning for Few Shot Medical Image Segmentation," 2024 IEEE International Symposium on Biomedical Imaging (ISBI), Athens, Greece, 2024, pp. 1-5, doi: 10.1109/ISBI56570.2024.10635439.

Possible challenges: domain shift, uncontinuous prediction mask, optimal transmission with spatial constraint.

# Method 2 (Under Experiment)

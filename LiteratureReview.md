---
title: Literature Review
bibliography: references.bib
---
# Literature Review
## Traditional Few Shot Segmentation
Conventional few-shot segmentation (FSS) paradigms typically rely on a pre-trained encoder with semantic discrimination capability to jointly encode the support set and query set. These methods are built on the core assumption that foreground regions from the support and query sets lie close to each other in the semantic space, whereas a significant semantic gap exists between the foreground and background. Under this assumption, the final mask prediction for segmentation can be produced by measuring the distance between the encoded query features and the foreground features derived from the support set.
### representative studies
#### PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment [@wang2019panet]  
**Contributions:** Prototype Alignment Regularization is proposed. During training, the predicted mask of the query is inverted and used as an additional support, which is then employed to perform segmentation on the original support. This strategy forces the support and query prototypes to be more consistent within the same embedding space.  
**Problems in FS medical image tasks:** I. Representing an object category predominantly via the mean of a handful of prototypes will inevitably discard the intra-class diversity of the category. Specifically, in the 1-shot or 5-shot settings, the multi-modal feature structure arising from variations in pose, scale, local deformation and occlusion is highly prone to being smoothed out by the averaging operation, given only 1 or 5 support samples are provided. II. Patch-wise nearest prototype matching is inherently limited by the absence of structural prior regularization and its independence from stable structural topology. This intrinsic drawback makes the method highly vulnerable to performance instability in the presence of appearance domain shift.

# Literature Review
## Traditional Few Shot Segmentation
Conventional few-shot segmentation (FSS) paradigms typically rely on a pre-trained encoder with semantic discrimination capability to jointly encode the support set and query set. These methods are built on the core assumption that foreground regions from the support and query sets lie close to each other in the semantic space, whereas a significant semantic gap exists between the foreground and background. Under this assumption, the final mask prediction for segmentation can be produced by measuring the distance between the encoded query features and the foreground features derived from the support set.
### representative studies
#### PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment
**Contributions:** Prototype Alignment Regularization is proposed. During training, the predicted mask of the query is inverted and used as an additional support, which is then employed to perform segmentation on the original support. This strategy forces the support and query prototypes to be more consistent within the same embedding space.

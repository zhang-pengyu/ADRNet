# ADRNet
The official code and results for IJCV paper: [Learning Adaptive Attribute-Driven Representation for Real-Time RGB-T Tracking](https://link.springer.com/article/10.1007/s11263-021-01495-3)

## Framework
![avatar](https://github.com/zhang-pengyu/ADRNet/blob/main/Framework.png)

- **Attribute-Driven Residual Branch (ADRB)** aims to build a robust and discriminative appearance model when RGB or T modality is not reliable for tracking. Four heterogeneous attributes, including extreme illumination (EI), occlusion (OCC), motion blur (MB), and thermal crossover (TC), are labeled.
- **Attribute Ensemble Network (AENet)** aggregates the residual features from different attributes in channel and spatial levels, which consists of two subnetworks: Channel Ensemble Network (CENet) and Spatial Ensemble Network (SENet).

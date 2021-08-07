# ADRNet
The official code and results for IJCV paper: [Learning Adaptive Attribute-Driven Representation for Real-Time RGB-T Tracking](https://github.com/zhang-pengyu/ADRNet/blob/main/Zhang_IJCV2021_ADRNet.pdf)

## Framework
![avatar](https://github.com/zhang-pengyu/ADRNet/blob/main/Framework.png)

- **Attribute-Driven Residual Branch (ADRB)** aims to build a robust and discriminative appearance model when RGB or T modality is not reliable for tracking. Four heterogeneous attributes, including extreme illumination (EI), occlusion (OCC), motion blur (MB), and thermal crossover (TC), are labeled.
- **Attribute Ensemble Network (AENet)** aggregates the residual features from different attributes in channel and spatial levels, which consists of two subnetworks: Channel Ensemble Network (CENet) and Spatial Ensemble Network (SENet).

## Experiments
### Comparison with SOTA on GTOT and RGBT234.
| Tracker | GTOT(MSR/MPR) | RGBT234(MSR/MPR)|
| ------ | ------ | ------ |
| **ADRNet** | **73.9/90.4** | **57.1/80.9** |
| CAT(ECCV 20') | 71.7/88.9 | 56.1/80.4 |
| MaCNet(Sensors 20') | 71.4/88.0 | 55.4/79.0 |
| MANet(ICCVW 19') | 72.4/89.4 | 53.9/77.7 |
| DAPNet(ACM MM 19') | 70.7/88.2 | 53.7/76.6 |

### Comparison with VOT2019-RGBT competitors.
| Tracker | EAO | Acc | R |
| ------ | ------ | ------ | ------ |
| JMMAC | 0.4826 | 0.6649 | 0.8211 |
| **ADRNet** | 0.3959 | 0.6218 | 0.7567 |
| SiamDW-T | 0.3925 | 0.6158 | 0.7839 | 
| mfDiMP | 0.3879 | 0.6019 | 0.8036 |
| FSRPN | 0.3553 | 0.6362 | 0.7069 |
| MANet | 0.3463 | 0.5823 | 0.7010 |
| MPAT | 0.3180 | 0.5723 | 0.7242 |
| CISRDCF | 0.2923 | 0.5215 | 0.6904 |
| GESBTT | 0.2896 | 0.6163 | 0.6350 |

## Get Started
### Set up Anaconda environment
```
conda create -n ADRNet python=3.7
conda activate ADRNet
cd $Path_to_ADRNet$
bash install.sh
```
### Run Demo sequence
```
cd $Path_to_ADRNet$
python Run_test.py
```
### Run RGBT234 and GTOT
```
cd $Path_to_ADRNet$
python Run_RGBT234.py
python Run_GTOT.py
```

# Source and model provenance

This repository contains a paper-guided reimplementation for the manuscript **CSFINet-Based Multimodal MRI Image Segmentation and Survival Prediction** by Jia-Lien Hsu and Pei-Yu Yu. It includes an additional five-channel survival-regression workflow. It is not a recovered copy of the manuscript's historical code, data split or checkpoints, and is not the original CSFINet authors' implementation.

The CSFINet architecture is described by Yu Feng, Yuhao Zhan and Hao Zeng, “CSFINet: Cross-scale Feature Interaction for Medical Image Segmentation,” ICSIP 2023, pp. 207–211, [DOI: 10.1109/ICSIP57908.2023.10270856](https://doi.org/10.1109/ICSIP57908.2023.10270856). The [authors' repository](https://github.com/CSFINet/CSFINet/tree/836dcca3f88ba65f99417566fa1c35767f508533), commit `836dcca3f88ba65f99417566fa1c35767f508533`, was consulted. Upstream code and figures are not redistributed here. Swin components are imported from torchvision.

The release viewer is independently implemented and reads locally generated artifacts. The legacy website code and its externally supplied helpers are not included. Consequently the local viewer need not have the same layout as the separately hosted demonstration.

Dependencies, including PyTorch, torchvision, Captum and Streamlit, remain subject to their respective licenses. This repository does not bundle their source. BraTS data and metadata are downloaded separately and remain subject to their source terms; model availability does not grant rights to redistribute the dataset.

## License status

The source and weights are publicly available. No additional general reuse or redistribution license is granted by this release; rights remain with the respective rights holders, subject to applicable law and GitHub's terms. An open-source license has not yet been selected. Do not interpret the absence of an upstream license, or the availability of a download, as an MIT license.

## Cite the work

Cite the accompanying manuscript by its unchanged title and authors above. A publication DOI is not asserted in this release. For the exact software version, cite the repository and its `v2.0.0` release. Cite the original CSFINet architecture and the BraTS dataset sources separately when relevant.

# Source and model provenance

This repository contains a paper-guided reimplementation for the manuscript **CSFINet-Based Multimodal MRI Image Segmentation and Survival Prediction** by Jia-Lien Hsu and Pei-Yu Yu. It includes an additional five-channel survival-regression workflow. It is not the original CSFINet authors' implementation.

The CSFINet architecture is described by Yu Feng, Yuhao Zhan and Hao Zeng, “CSFINet: Cross-scale Feature Interaction for Medical Image Segmentation,” ICSIP 2023, pp. 207–211, [DOI: 10.1109/ICSIP57908.2023.10270856](https://doi.org/10.1109/ICSIP57908.2023.10270856). The [authors' repository](https://github.com/CSFINet/CSFINet/tree/836dcca3f88ba65f99417566fa1c35767f508533), commit `836dcca3f88ba65f99417566fa1c35767f508533`, was consulted. Upstream code and figures are not redistributed here. Swin components are imported from torchvision.

The release viewer is independently implemented and reads locally generated artifacts. The legacy website code and its externally supplied helpers are not included. Consequently the local viewer need not have the same layout as the separately hosted demonstration.

Dependencies, including PyTorch, torchvision, Captum and Streamlit, remain subject to their respective licenses. This repository does not bundle their source. BraTS data and metadata are downloaded separately and remain subject to their source terms; model availability does not grant rights to redistribute the dataset.

## License and scope

Project-authored source code, documentation and configurations, together with the six project-trained checkpoint files identified in `release-manifest.json`, are licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for the complete terms and [NOTICE](NOTICE) for attribution notices. The grant covers rights held by the project contributors; it does not relicense the original CSFINet implementation, third-party dependencies, BraTS images or clinical metadata.

Each weight archive in the licensed release includes `LICENSE` and `NOTICE`. Redistribution must comply with Apache 2.0, including its applicable license, notice and modification-marking requirements. The research-use description in the documentation communicates the scope of experimental validation and does not impose an additional noncommercial restriction on the Apache-licensed project materials.

Dataset access, use and citation remain governed by the original providers' terms. [Data-source licensing notes](docs/data-licensing.md) record the sources checked and distinguish the repository's license from dataset permissions. Requests to cite the accompanying manuscript are scholarly attribution guidance, not an additional condition of Apache 2.0.

## Cite the work

Cite the accompanying manuscript by its unchanged title and authors above. A publication DOI is not asserted in this release. For the exact software version, cite the repository and its `v2.0.2` release. Cite the original CSFINet architecture and the BraTS dataset sources separately when relevant.

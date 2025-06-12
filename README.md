## Notes
Please visit our new repositories for **AbX** at <https://github.com/CarbonMatrixLab/AbX> and **CarbonNovo** at <https://github.com/CarbonMatrixLab/carbonnovo>. This repository will continue to host the code for CarbonDesign.

## CarbonDesign
**Contact**\
Haicang Zhang (zhanghaicang@sjtu.edu.cn) 


**Citation**\
Accurate and robust protein sequence design with CarbonDesign.  M. Ren, C. Yu, D. Bu, H. Zhang. Nature Machine Intelligence. 6, 536–547 (2024). https://doi.org/10.1038/s42256-024-00838-2

**Contributions**\
H.Z. conceived the ideas and implemented the CarbonDesign model and algorithms. H.Z. and M.R. designed the experiments, and M.R. conducted the main experiments and analysis. M.R. wrote the manuscript. H.Z., D.B. and C.Y. revised the manuscript.

**Acknowledgements**\
We acknowledge the financial support from the National Natural Science Foundation of China (grant no. 32370657) and the Project of Youth Innovation Promotion Association CAS to H.Z. We also acknowledge the financial support from the Development Program of China (grant no. 2020YFA0907000) and the National Natural Science Foundation of China (grant nos. 32271297 and 62072435). We thank Beijing Paratera Co., Ltd and the ICT Computing-X Center, Chinese Academy of Sciences, for providing computational resources.



**Installation**  

1. CarbonDesign relies on the ESM2 language model. You can install ESM2 using the following command:  `pip install fair-esm`.
2. Install other required libraries by running `bash install.sh`.

**Model weights**
1. Download CarbonDesign model weights from <https://carbondesign.s3.amazonaws.com/params.tar>, and place them in the ./params directory.
2. Download the ESM2 model weights from <https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt> and <https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt>, and place them in the `./params` directory. 


**Usage**\
You are required to input the **PDB** file (**--data_dir**) of the protein backbone structures. CarbonDesign will subsequently output the designed protein sequence (**--output_dir**). Additionally, CarbonDesign supports the prediction of the side chain structures of the designed sequences (**--save_sidechain**).  
Example,

```
python -u run_carbondesign.py --data_dir ./data/pdbs  --output_dir ./results  --name_idx ./data/pdbs/name.idx
```

Main arguments:  
data_dir: input directory of pdb files  
output_dir: output directory  
name_idx: list of pdb ids whose pdb files have been put in the input directory  


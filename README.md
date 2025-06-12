## Notes
Please visit our new repositories for **AbX** at <https://github.com/CarbonMatrixLab/AbX> and **CarbonNovo** at <https://github.com/CarbonMatrixLab/carbonnovo>. This repository will continue to host the code for CarbonDesign.


## CarbonMatrix Team
**Aims**\
We aim to develop large-scale generative AI models for protein structure prediction and protein design. We have already released CarbonDesign, and we plan to release more models soon.

**Contact**\
Haicang Zhang (zhanghaicang@sjtu.edu.cn) 

**News**\
**11 June 2024**: CarbonNovo and AbX are available online in the ICML 2024.  
CarbonNovo:  <https://openreview.net/attachment?id=FSxTEvuFa7&name=pdf>  
AbX: <https://openreview.net/pdf?id=1YsQI04KaN> 


**23 May 2024**: Our paper, CarbonDesign, is now available online in Nature Machine Intelligence. <https://www.nature.com/articles/s42256-024-00838-2>.  


**2 May 2024**: Our two papers, CarbonNovo and AbX, have been accepted by ICML 2024. We will release the source codes in this repository soon.  
- CarbonNovo: Joint Design of Protein Structure and Sequence Using a Unified Energy-based Model. M. Ren, T. Zhu, H. Zhang#.
- Antibody Design Using a Score-based Diffusion Model Guided by Evolutionary, Physical and Geometric Constraints. T. Zhu, M. Ren, H. Zhang#.


**9 Jan 2024**: Our paper, CarbonDesign, has been accepted by Nature Machine Intelligence.  
Accurate and robust protein sequence design with CarbonDesign. <https://www.biorxiv.org/content/10.1101/2023.08.07.552204v1> 

**24 Oct 2024**
We resolved the issue with using the PDB file residue numbers. CarbonDesign can now generate a mapping file that links the designed residue types with the corresponding residue numbers in the PDB file (**--save_map**). We used mmCIF files for training, as their mapping relationships allow for more accurate sequence design.


## CarbonDesign
Protein sequence design is critically important for protein engineering. Despite recent advancements in deep learning-based methods, achieving accurate and robust sequence design remains a challenge. Here we present CarbonDesign, an approach that draws inspiration from successful ingredients of AlphaFold and which has been developed specifically for protein sequence design. At its core, CarbonDesign introduces Inverseformer, which learns representations from backbone structures and an amortized Markov random fields model for sequence decoding. Moreover, we incorporate other essential AlphaFold concepts into CarbonDesign: an end-to-end network recycling technique to leverage evolutionary constraints from protein language models and a multitask learning technique for generating side-chain structures alongside designed sequences. CarbonDesign outperforms other methods on independent test sets including the 15th Critical Assessment of protein Structure Prediction (CASP15) dataset, the Continuous Automated Model Evaluation (CAMEO) dataset and de novo proteins from RFDiffusion. Furthermore, it supports zero-shot prediction of the functional effects of sequence variants, making it a promising tool for applications in bioengineering.

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

**Citation**\
Accurate and robust protein sequence design with CarbonDesign.  M. Ren, C. Yu, D. Bu, H. Zhang. Nature Machine Intelligence. 6, 536–547 (2024). https://doi.org/10.1038/s42256-024-00838-2




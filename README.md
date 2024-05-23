## CarbonMatrix Team
**Aims**\
We aim to develop large-scale generative AI models for protein structure prediction and protein design. We have already released CarbonDesign, and we plan to release more models soon.

**Current members**\
Haicang Zhang (zhanghaicang@ict.ac.cn), Team leader\
Milong Ren, Tian Zhu, Zaikai He, Siyuan Tao

**News**\
**23 May 2024**: Our paper, CarbonDesign, is now available online in Nature Machine Intelligence. <https://www.nature.com/articles/s42256-024-00838-2>.  


**2 May 2024**: Our two papers, CarbonNovo and AbX, have been accepted by ICML 2024. We will release the source codes in this repository soon.  
- CarbonNovo: Joint Design of Protein Structure and Sequence Using a Unified Energy-based Model. M. Ren, T. Zhu, H. Zhang#.
- Antibody Design Using a Score-based Diffusion Model Guided by Evolutionary, Physical and Geometric Constraints. T. Zhu, M. Ren, H. Zhang#.


**9 Jan 2024**: Our paper, CarbonDesign, has been accepted by Nature Machine Intelligence.  
Accurate and robust protein sequence design with CarbonDesign. <https://www.biorxiv.org/content/10.1101/2023.08.07.552204v1> 



## CarbonDesign
Protein sequence design is critically important for protein engineering. Despite recent advancements in deep learning-based methods, achieving accurate and robust sequence design remains a challenge. Here we present CarbonDesign, an approach that draws inspiration from successful ingredients of AlphaFold and which has been developed specifically for protein sequence design. At its core, CarbonDesign introduces Inverseformer, which learns representations from backbone structures and an amortized Markov random fields model for sequence decoding. Moreover, we incorporate other essential AlphaFold concepts into CarbonDesign: an end-to-end network recycling technique to leverage evolutionary constraints from protein language models and a multitask learning technique for generating side-chain structures alongside designed sequences. CarbonDesign outperforms other methods on independent test sets including the 15th Critical Assessment of protein Structure Prediction (CASP15) dataset, the Continuous Automated Model Evaluation (CAMEO) dataset and de novo proteins from RFDiffusion. Furthermore, it supports zero-shot prediction of the functional effects of sequence variants, making it a promising tool for applications in bioengineering.

**Installation**\
Please install the required libraries using install.sh.

**Model weights**\
The model weights can be downloaded from <https://carbondesign.s3.amazonaws.com/params.tar> .

**Usage**\
You are required to input the **PDB** file (**--data_dir**) of the protein backbone structures. CarbonDesign will subsequently output the designed protein sequence (**--output_dir**). Additionally, CarbonDesign supports the prediction of the side chain structures of the designed sequences (**--save_sidechain**).
````python
python -u run_carbondesign.py 
--model ./params/carbondesign_default.ckpt ## model name
--model_features ./config/config_data_mrf2.json ## feature config
--model_config ./config/config_model_mrf_pair_enable_esm_sc.json ## model config
--data_dir ../data/pdbs ## input pdbs path
--output_dir ../ results ## results path
--device gpu ## use GPU
--gpu_idx 0 ## gpu index
--name_idx ../data/pdbs/name.idx ## list of sequences that need to be designed is required
--temp 0.01 ##sampling temperature
````
**Citation**\
Accurate and robust protein sequence design with CarbonDesign.  M. Ren, C. Yu, D. Bu, H. Zhang. Nature Machine Intelligence. 6, 536–547 (2024). https://doi.org/10.1038/s42256-024-00838-2

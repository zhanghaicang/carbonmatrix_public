## CarbonMatrix Team
**Aims**\
We aim to develop large-scale generative AI models for protein structure prediction and protein design. We have already released CarbonDesign, and we plan to release more models soon.

**Current members**\
Haicang Zhang (zhanghaicang@ict.ac.cn) \
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
--data_dir ./data/pdbs ## input pdbs path
--output_dir ./results ## results path
--device gpu ## use GPU
--gpu_idx 0 ## gpu index
--name_idx ./data/pdbs/name.idx ## list of sequences that need to be designed is required
--temp 0.0 ##sampling temperature
````
**Citation**\
Accurate and robust protein sequence design with CarbonDesign.  M. Ren, C. Yu, D. Bu, H. Zhang. Nature Machine Intelligence. 6, 536–547 (2024). https://doi.org/10.1038/s42256-024-00838-2

## CarbonNovo
De novo protein design aims to create novel protein structures and sequences unseen in nature. Recent structure-oriented design methods typically employ a two-stage strategy, where structure design and sequence design modules are trained separately, and the backbone structures and sequences are generated sequentially in inference. While diffusion-based generative models like RFdiffusion show great promise in structure design, they face inherent limitations within the two-stage framework. First, the sequence design module risks overfitting, as the accuracy of the generated structures may not align with that of the crystal structures used for training. Second, the sequence design module lacks interaction with the structure design module to further optimize the generated structures. To address these challenges, we propose CarbonNovo, a unified energy-based model for jointly generating protein structure and sequence. Specifically, we leverage a score-based generative model and Markov Random Fields for describing the energy landscape of protein structure and sequence. In CarbonNovo, the structure and sequence design module communicates at each diffusion step, encouraging the generation of more coherent structure-sequence pairs. Moreover, the unified framework allows for incorporating the protein language models as evolutionary constraints for generated proteins. The rigorous evaluation demonstrates that CarbonNovo outperforms two-stage methods across various metrics, including designability, novelty, sequence plausibility, and Rosetta Energy.

**Installation**\
We are working on organizing the code and will release the software soon.

**Citation**\
CarbonNovo: Joint Design of Protein Structure and Sequence Using a Unified Energy-based Model. M. Ren, T. Zhu, H. Zhang. ICML 2024.

## AbX
Antibodies are central proteins in adaptive immune responses, responsible for protecting against viruses and other pathogens. Rational antibody design has proven effective in the diagnosis and treatment of various diseases like cancers and virus infections. While recent diffusion-based generative models show promise in designing antigen-specific antibodies, the primary challenge lies in the scarcity of labeled antibody-antigen complex data and binding affinity data. We present AbX, a new score-based diffusion generative model guided by evolutionary, physical, and geometric constraints for antibody design. These constraints serve to narrow the search space and provide priors for plausible antibody sequences and structures. Specifically, we leverage a pre-trained protein language model as priors for evolutionary plausible antibodies and introduce additional training objectives for geometric and physical constraints like van der Waals forces. Furthermore, as far as we know, AbX is the first score-based diffusion model with continuous timesteps for antibody design, jointly modeling the discrete sequence space and the SE(3) structure space. Evaluated on two independent testing sets, we show that AbX outperforms other published methods, achieving higher accuracy in sequence and structure generation and enhanced antibody-antigen binding affinity. Ablation studies highlight the clear contributions of the introduced constraints to antibody design.

**Installation**\
We are working on organizing the code and will release the software soon.

**Citation**\
Antibody Design Using a Score-based Diffusion Model Guided by Evolutionary, Physical and Geometric Constraints. T. Zhu, M. Ren, H. Zhang. ICML 2024.



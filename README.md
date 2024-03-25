## CarbonDesign

**Installation**\


**Usage**\
You are required to input the **PDB** file (**--data_dir**) of the protein backbone structures. CarbonDesign will subsequently output the designed protein sequence (**--output_dir**). Additionally, CarbonDesign supports the prediction of the side chain structures of the designed sequences (**--save_sidechain**).
````python
python -u run_carbondesign.py 
--model ../data/models/default_model.ckpt ## model name
--model_features ./config/config_data_mrf2.json ## feature config
--model_config ./config/config_model_mrf_pair_enable_esm_sc.json ## model config
--data_dir ../data/pdbs ## input pdbs path
--output_dir ../ results ## results path
--device gpu ## use GPU
--gpu_idx 0 ## gpu index
--name_idx ../data/pdbs/name.idx ## list of sequences that need to be designed is required
--temp 0.01 ##sampling temperature
````




## CarbonMatrix Team
We aim to develop large-scale deep learning and generative AI models for protein structure prediction and protein design. We are going to release more models.

Current members:\
Haicang Zhang, team leader\
Milong Ren, Tian Zhu, Zaikai He, Siyuan Tao

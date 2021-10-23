# Neighborhood Reconstructing Autoencoders
The official repository for \<Neighborhood Reconstructing Autoencoders\> (Lee, Kwon, and Park, NeurIPS 2021).

> This paper proposes Neighborhood Reconstructing Autoencoders (NRAE), which is a **graph**-based autoencoder that explicitly accounts for the **local connectivity and geometry** of the data, and consequently learns a **more accurate data manifold and representation**.

Arxiv: [TBU]()  
15-mins video: [TBU]()

# Preview
<center>
<figure>
<img src="./results/AE_toy_denoising/AE_training.gif" alt="drawing" style='width:30%;'/>
<img src="./results/NRAEL_toy_denoising/NRAEL_training.gif" alt="drawing" style='width:30%;'/>
<img src="./results/NRAEQ_toy_denoising/NRAEQ_training.gif" alt="drawing" style='width:30%;'/>
<figcaption><I>Figure 1: De-noising property of the NRAE.</I></figcaption>
</figure>
<figure>
<img src="./results/AE_toy_geometry_preserving/AE_training.gif" alt="drawing" style='width:30%;'/>
<img src="./results/NRAEL_toy_geometry_preserving/NRAEL_training.gif" alt="drawing" style='width:30%;'/>
<img src="./results/NRAEQ_toy_geometry_preserving/NRAEQ_training.gif" alt="drawing" style='width:30%;'/>
<figcaption><I>Figure 2: Correct local connectivity learned by the NRAE.</I></figcaption>
</figure>
</center>

# Environment

The project is developed under a standard PyTorch environment.
- python 3.8.8
- numpy 
- matplotlib 
- imageio 
- argparse 
- yaml 
- omegaconf 
- torch 1.8.0
- CUDA 11.1

# Running
```
python train_and_eval.py --config configs/config_name.yml --device 0
```
The `config_name.yml` is one of the files in `configs/` (e.g., `AE_toy_denoising.yml`). 

# Playing with the code
- You can add a new type of 2d dataset in `loader.synthetic_dataset.SyntheticData.get_data` (currently, we have `sincurve` and `swiss_roll`).
- The most significant parameters requiring tuning include: i) the number of nearest neighbors for graph construction `num_nn` and ii) kernel parameter `lambda` (you can find these parameters in `configs/NRAEL_toy_denoising.yml` for example). 
- We empirically observe that setting as `include_center=True` (when defining data loader) has performance advantange. 

# Citation
To be updated.
# Neighborhood Reconstructing Autoencoders
The official repository for \<Neighborhood Reconstructing Autoencoders\> (Lee, Kwon, and Park, NeurIPS 2021).

> This paper proposes Neighborhood Reconstructing Autoencoders (NRAE), which is a **graph**-based autoencoder that explicitly accounts for the **local connectivity and geometry** of the data, and consequently learns a **more accurate data manifold and representation**.

- *Paper: [TBU]()*  
- *15-mins video: [TBU]()*  
- *Slides: [./doc/NRAE_talk.pdf](./doc/NRAE_talk.pdf)*  
- *Poster: [TBU]()*  
- *OpenReview: [TBU]()*  

## Preview (synthetic data)
<center>
<div class="imgCollage">
<span style="width: 31.8%"><img src="./results/AE_toy_denoising/AE_training.gif" width="250 height="190"/></span>
<span style="width: 31.8%"><img src="./results/NRAEL_toy_denoising/NRAEL_training.gif" width="250 height="190"/> </span>
<span style="width: 31.8%"><img src="./results/NRAEQ_toy_denoising/NRAEQ_training.gif" width="250 height="190"/> </span>
</div>
  <I>Figure 1: De-noising property of the NRAE (Left: Vanilla AE, Middle: NRAE-L, Right: NRAE-Q). </I>
</center>

<center>
<div class="imgCollage">
<span style="width: 31.8%"><img src="./results/AE_toy_geometry_preserving/AE_training.gif" width="250 height="190"/></span>
<span style="width: 31.8%"><img src="./results/NRAEL_toy_geometry_preserving/NRAEL_training.gif" width="250 height="190"/> </span>
<span style="width: 31.8%"><img src="./results/NRAEQ_toy_geometry_preserving/NRAEQ_training.gif" width="250 height="190"/> </span>
</div>
  <I>Figure 2: Correct local connectivity learned by the NRAE (Left: Vanilla AE, Middle: NRAE-L, Right: NRAE-Q).</I>
</center>

## Preview (rotated/shifted MNIST)
<center>
<div class="imgCollage">
<span style="width: 100%"><img src="./results/AE_mnist_rotated/AE.gif" width="750 height="190"/></span>
<span style="width: 100%"><img src="./results/NRAEL_mnist_rotated/NRAE.gif" width="750 height="190"/> </span>
<span style="width: 100%"><img src="./results/NRAEQ_mnist_rotated/NRAE.gif" width="750 height="190"/> </span>
</div>
  <I>Figure 3: Generated sequences of rotated images by travelling the 1d latent spaces (Top: Vanilla AE, Middle: NRAE-L, Bottom: NRAE-Q). </I>
</center>

<center>
<div class="imgCollage">
<span style="width: 100%"><img src="./results/AE_mnist_shifted/AE.gif" width="750 height="190"/></span>
<span style="width: 100%"><img src="./results/NRAEL_mnist_shifted/NRAE.gif" width="750 height="190"/> </span>
<span style="width: 100%"><img src="./results/NRAEQ_mnist_shifted/NRAE.gif" width="750 height="190"/> </span>
</div>
  <I>Figure 3: Generated sequences of shifted images by travelling the 1d latent spaces (Top: Vanilla AE, Middle: NRAE-L, Bottom: NRAE-Q). </I>
</center>

## Environment

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

## Running (synthetic data)
```
python train_synthetic.py --config configs/config_name.yml --device 0
```
The `config_name.yml` is one of the files in `configs/` (e.g., `AE_toy_denoising.yml`). 

### Playing with the code
- You can add a new type of 2d dataset in `loader.synthetic_dataset.SyntheticData.get_data` (currently, we have `sincurve` and `swiss_roll`).
- The most significant parameters requiring tuning include: i) the number of nearest neighbors for graph construction `num_nn` and ii) kernel parameter `lambda` (you can find these parameters in `configs/NRAEL_toy_denoising.yml` for example). 
- We empirically observe that setting as `include_center=True` (when defining data loader) has performance advantange. 

## Running (rotaed/shifted MNIST)

## Citation
To be updated.

# Object Genration Experiments for Riemannian LoRA

**Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models** <br>
*Fangzhao Zhang, Mert Pilanci* <br>
Paper: [https://arxiv.org/abs/2402.02347](https://arxiv.org/abs/2402.02347) <br>

This repository builds on the [custom diffsuon](https://github.com/cloneofsimo/lora)  project. 

<p>
<img src="figures/vase.png" width="800" >
</p>

We also generate figures with target objects chair and dog, with training images stored in [contents/](contents).

## Repository Overview
* [contents/](contents) contains the training images of different objects.
* [lora_diffusion/](lora_diffusion) contains the code for lora fine-tuning custom-diffusion model.
* [training_scripts/](training_scripts) contains scripts code for training and testing.

## Requirements
See the [custom diffsuon](https://github.com/cloneofsimo/lora) repository for requirements.
```
pip install -r requirements.txt
python setup.py develop
 ```

## Training
```bash
cd training_scripts
sh run_lora_db_w_text.sh scaled_adamw 1e-2 1e6
 ```
Here <code>sgd, scaled_gd, adamw, scaled_adamw</code> are all valid choices for command line optimizer arguments, which specify the optimizer to be used. <code>1e-2</code> specifies learning rate to be used. <code>1e6</code> specifies preconditioner regularization to be used, which is only needed for <code>scaled_gd</code> and <code>scaled_adamw</code> optimizer. The tuned model path is stored in <code>OUTPUT_DIR</code> in <code>run_lora_db_w_text.sh</code> file (here is <code>exps/examples/vase/lora_weight.safetensors</code> in default code).

## Testing
```bash
python test.py
 ```
Generated images are saved at <code>exps/examples/vase</code> in default code.

## Parameter Reference
Unet training is fixed to have learning rate <code>1e-4</code> and preconditioner regularization <code>0</code>. We only change text encoder training parameters. To reproduce Figure 1 result, we set


| optimizer  |learning rate (preconditioner regularization) | 
| ------------- | ------------- |
| (scaled) adamw  | 1e-2 (1e6)  |
| (scaled) adamw  | 1e-3 (1e8)  |
| (scaled) adamw  | 5e-5 (1e6)  |
| (scaled) adamw  | 1e-6 (1e8)  |

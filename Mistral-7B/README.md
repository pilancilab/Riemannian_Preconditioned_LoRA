# Mistral-7B experiments for Riemannian LoRA



**Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models** <br>
*Fangzhao Zhang, Mert Pilanci* <br>
Paper: XXX <br>

<p>
<img src="figures/score_mistral.png" width="800" >
</p>

## Repository Overview
* [mistral/](mistral) contains the source code used for optimizers and trainer which overwrites HuggingFace Transformers trainer class.
* [mistral_glue.py](mistral_glue.py) contains main training code.

## Requirements
Install all required dependencies by
```bash
 pip install -r requirement.txt
 ```

## Training and Evaluating
 Run the following code
 ```bash
python mistral_glue.py --optimizer scaled_adamw
 ```
Here <code>sgd, scaled_gd, adamw, scaled_adamw</code> are all valid choices for <code>--optimizer</code>.

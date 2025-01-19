# TOAD-GAN: Procedural Generator of 2d Platform Game Levels

### Table of Contents
- [1. Overview](#overview)
- [2. Installation](#installation)
- [3. Usage](#usage)
  - [i. Training](#training)
  - [ii. Evaluation](#evaluation)
- [4. Results](#results)
- [5. Authors](#authors)


## Overview

This project implements a **Generative Adversarial Network (GAN)** using a modified version of **Toad GAN** tailored to generate **2D levels** for the **Sonic the hedgehog** game. 

The primary goal of this repository is to create a user-friendly workflow for generating diverse and coherent 2D levels inspired by the iconic Sonic platformer games, with a focus on making it easy for beginners to use and adapt the technology.

The model utilizes a multi-scale approach to generate game levels progressively, learning patterns from only one example. Additionally, reinforcement learning (RL) techniques are integrated to evaluate and refine generated levels to ensure their playability and coherence.

### Reference
The core methodology of this project is based on the work of:

```
@inproceedings{awiszus2020toadgan,
  title={TOAD-GAN: Coherent Style Level Generation from a Single Example},
  author={Awiszus, Maren and Schubert, Frederik and Rosenhahn, Bodo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment},
  year={2020}
}
```

### Documentation
More details about the project can be found [here]()

## Installation

### Prerequisites:
To run this project, you need the following dependencies:

- **Python 3.x** (preferably Python 3.7 or above) - see [Python 3](https://www.python.org/downloads)
- **PyTorch** (with GPU support if available)
- **wandb** for logging and visualization.
- **tqdm** for progress bars.
- Other Python dependencies can be found in `requirements.txt`.

## Usage

### Training

### Step 1: Clone the repository

```bash
git clone https://github.com/vsx23733/SONIC-GAN.git

cd SONIC-GAN
```

### Step 2 (Optional): Create environment

We recommend setting up a [virtual environment with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
and installing the packages there.

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you have a **GPU**, make sure it is usable with [PyTorch](https://pytorch.org) to speed up training. You can modify the dependency in the `requirements.txt` file to install GPU-optimized libraries.

### Step 4: Training

Once all prerequisites are installed, TOAD-GAN can be trained by running `main.py`. Make sure you are using the python installation you installed the prerequisites into.

For example, to train a 3-layer Toad GAN on level 1-1 of Sonic with 4000 iterations per scale, use the following command:

```
$ python main.py --game sonic --not_cuda  --input-dir input\sonic --input-name lvl_1-1.txt --alpha 200 --nfc 128 --out output
```

#### Command-line explanation

There are several command line options available for training. These are defined in `config.py`.

See below an explanation of the different hyperparameters of the TOAD-GAN model, their importance and our recommendation for utilization.

**1.	--game**
- **Description**: Specifies the game type (e.g., mario, mariokart, sonic).
- **Importance**: Determines the context for the generated levels or tasks.
- **Recommendation**: Choose based on the game you're targeting.

**2.	--not_cuda**
- **Description**: Disables CUDA (GPU computation).
- **Importance**: Use only if you don’t have a GPU or encounter compatibility issues.
- **Recommendation**: Keep GPU enabled (do not set --not_cuda) for better performance if available.

**3.	--manualSeed**
- **Description**: Allows setting a specific seed for reproducibility.
- **Importance**: Critical for debugging and deterministic results.
- **Recommendation**: Set a value when experimenting or debugging, leave empty to use a random seed.

**File Paths**

**4.	--netG and --netD**
- **Description**: Paths to pre-trained generator (netG) or discriminator (netD) models for resuming training.
- **Importance**: Load these if continuing training or reusing a model.
- **Recommendation**: Provide paths only if resuming training.

**5.	--out, --input-dir, --input-name**
- **Description**: Controls where results are saved (--out), input directory (--input-dir), and specific file (--input-name).
- **Importance**: Defines input and output file locations.
- **Recommendation**: Adjust paths for your project structure.

**Network Hyperparameters**

**6.	--nfc**
- **Description**: Number of convolutional filters.
- **Importance**: Affects model capacity; higher values can improve results but require more memory.
- **Recommendation**: We setted this hyperparameter to 128 instead of default 64 because of the complexity of the game.

**7.	--ker_size**
- **Description**: Kernel size for convolutional layers.
- **Importance**: Affects how much context each filter sees.
- **Recommendation**: Stick to 3 (default) unless you have a specific reason to change.

**8.	--num_layer**
- **Description**: Number of convolutional layers in the network.
- **Importance**: Affects model depth; deeper networks may capture more details but can overfit.
- **Recommendation**: Default (3) is sufficient for most cases.

**Scaling Parameters**

**9.	--scales**
- **Description**: Descending scale factors for multiscale generation.
- **Importance**: Defines the level of detail and scale transitions in the output.
- **Recommendation**: Use default unless you want finer or coarser scaling.

**10.	--noise_update**
- **Description**: Weight for added noise during training.
- **Importance**: Helps prevent overfitting and adds diversity.
- **Recommendation**: Stick with default (0.1) for stable training.

**11.	--pad_with_noise**
- **Description**: Adds random noise padding around inputs.
- **Importance**: Can make edge outputs more diverse.
- **Recommendation**: Use cautiously as it may introduce randomness in edges.

**Optimization Parameters**

**12.	--niter**
- **Description**: Number of training epochs per scale.
- **Importance**: Longer training can improve quality but takes more time.
- **Recommendation**: Use default (4000) for most tasks.

**13.	--gamma**
- **Description**: Learning rate scheduler decay factor.
- **Importance**: Controls how learning rate reduces over time.
- **Recommendation**: Default (0.1) works well.

**14.	--lr_g, --lr_d**
- **Description**: Learning rates for generator and discriminator.
- **Importance**: Controls training speed and stability.
- **Recommendation**: Default values (0.0005) are fine for most cases.

**15.	--beta1**
- **Description**: Adam optimizer's momentum parameter.
- **Importance**: Affects stability and convergence speed.
- **Recommendation**: Default (0.5) is standard for GANs.

**16.	--Gsteps, --Dsteps**
- **Description**: Number of generator/discriminator updates per iteration.
- **Importance**: Balances training between generator and discriminator.
- **Recommendation**: Use default (3).

**17.	--lambda_grad**
- **Description**: Weight for gradient penalty in discriminator.
- **Importance**: Ensures discriminator stability during training.
- **Recommendation**: Stick with default (0.1).

**18.	--alpha**
- **Description**: Weight for reconstruction loss.
- **Importance**: Controls tradeoff between realism and fidelity to input.
- **Recommendation**: Adjust depending on output requirements (we modified this parameter to 200 (default is 100)).

**Experimental**

**19.	--token_insert**
- **Description**: Determines the layer for splitting token groupings.
- **Importance**: Experimental; may impact training stability.
- **Recommendation**: Use default (-2) unless experimenting.

### Generating samples

If you want to use your trained TOAD-GAN to generate more samples, use `generate_samples.py`.
Make sure you define the path to a pretrained TOAD-GAN and the correct input parameters it was trained with.

```
$ python generate_samples.py  --out_ path/to/pretrained/TOAD-GAN --input-dir input --input-name lvl_1-1.txt --num_layer 3 --alpha 100 --niter 4000 --nfc 64
```

## Evaluation

## Results

## Authors

This project is developed by [ISART Digital](https://www.isart.fr/?utm_source=google&utm_medium=cpc&utm_campaign=datashake%20-%20Search%20-%20Marque&utm_term=isart%20digital&hsa_acc=4905923708&hsa_cam=11636978144&hsa_grp=146613250209&hsa_ad=636937715424&hsa_src=g&hsa_tgt=kwd-1000928741934&hsa_kw=isart%20digital&hsa_mt=e&hsa_net=adwords&hsa_ver=3&gad_source=1&gclid=Cj0KCQiAv628BhC2ARIsAIJIiK-e6QnV7auBhkLWBcnB59CPKfb0ZWwGC7LRSxpgaXPQRuYSwnw1TUMaAsDuEALw_wcB) in collaboration with [Aivancity](https://www.aivancity.ai/?utm_term=aivancity&utm_campaign=Acquisition+%5BSF,+Search,+FR%5D&utm_source=adwords&utm_medium=ppc&hsa_acc=4115683429&hsa_cam=16679639181&hsa_grp=135347317816&hsa_ad=695767871507&hsa_src=g&hsa_tgt=kwd-1074062695173&hsa_kw=aivancity&hsa_mt=p&hsa_net=adwords&hsa_ver=3&gad_source=1&gclid=Cj0KCQiAv628BhC2ARIsAIJIiK_OyiZbkRQ6wy9TjJ-WIyVrhvQUkaAPNKpG3hlbuYktgNwONTSmeToaAq2VEALw_wcB) 3rd year students :

* **[Axel ONOBIONO](https://www.linkedin.com/in/axel-onobiono/)**
* **[Florian HOUNKPATIN](https://www.linkedin.com/in/florian-hounkpatin/)**
* **[Noémi DOMBOU](https://www.linkedin.com/in/noemi-dombou/)**
* **[Asser OMAR](https://www.linkedin.com/in/asseromar/)**
* **[Ephraim KOSSONOU](https://www.linkedin.com/in/ephraïm-kossonou/)**

## Copyright

This code is not endorsed by Sega and is only intended for research purposes. 
Sonic is a Sega character which the authors don’t own any rights to. 

Sega is also the sole owner of all the graphical assets in the game.
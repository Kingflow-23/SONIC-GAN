# Toad GAN for Sonic 2D Level Generation

This project implements a **Generative Adversarial Network (GAN)** using a modified version of **Toad GAN** for generating **2D levels** for the **Sonic** game. Toad GAN is a variant of GAN designed for generating game levels from scratch. It learns to produce new levels based on examples, progressing through multiple scales during training.

The goal of this project is to generate coherent and diverse 2D Sonic levels through an unsupervised learning framework.

### Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
    1. [Training](#training)
    2. [Configuration](#configuration)


## Overview

This repository implements **Toad GAN** for generating **2D Sonic levels**. The architecture of Toad GAN has been adapted to generate levels that resemble the style and design patterns seen in Sonic's 2D platformer games.

### Key Features:
- **Multi-scale GAN architecture**: The model progressively refines the generated levels through various scales.
- **Game-specific tokenization**: The levels are tokenized into meaningful entities (e.g., platforms, enemies, obstacles).
- **Training pipeline**: Designed for efficient training with the capability to handle different scales of generation.
- **Integration with **wandb**: For logging and visualizing training metrics and generated levels.

This project is inspired by **Toad GAN**, a GAN variant for level generation, but it's specifically tailored for generating **Sonic 2D levels**.

---

## Installation

### Prerequisites:
To run this project, you need the following dependencies:

- **Python 3.x** (preferably Python 3.7 or above)
- **PyTorch** (with GPU support if available)
- **wandb** for logging and visualization.
- **tqdm** for progress bars.
- Other Python dependencies can be found in `requirements.txt`.

### Step 1: Clone the repository

```bash
git clone https://github.com/vsx23733/SONIC-GAN.git

cd SONIC-GAN
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

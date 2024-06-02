
# Physically Constrained 3D Diffusion for Inverse Design of Fiber-reinforced Polymer Composite Materials

_*Abstract*_ -- Designing fiber-reinforced polymer composites (FRPCs) with a tailored nonlinear stress-strain response can enable innovative applications across various industries. However, the design space of FRPCs is inherently 3D topological nature with physical constraints. No previous efforts have successfully achieved the inverse design of FRPCs that target the entire stress-strain curve. Here, we develop PC3D_Diffusion, a diffusion model designed for the inverse design of FRPCs. We simulate 1.3 million FRPCs and calculate their stress-strain curve for training. We adopt two distinct forms of diffusion models to process the position and orientation of fibers in 3D space. Moreover, we propose a loss-guided, learning-free approach to apply physical constraints during generation. PC3D_Diffusion can generate high-quality designs with tailored mechanical behaviors while guaranteeing to satisfy the physical constraints. PC3D_Diffusion advances FRPC inverse design and may facilitate the inverse design of other 3D materials, offering potential applications in industries reliant on materials with custom mechanical properties.

## Dataset

We provide our generated training and testing data [here](https://drive.google.com/drive/folders/1ezahBsw5ogX1JilRAmnFWFdTE1KVkiGz?usp=drive_link).

## Code Usage

### Train

    python train.py \
        --train <training data files> \
        --test <valuating data files> \
        --ckpt <target checkpoint directory> \
        --workers <number of workers for distributed training> \
        --rank <rank of the current work> \
        --device <device id>

### Test

    python eval.py \
        --ckpt <checkpoint directory or file> \
        --n <number of fibers> \
        --length <fiber length> \
        --diameter <fiber diameter> \
        --a0 <first order coefficient of SS curve> \
        --a1 <second order coefficient of SS curve> \
        --a2 <third order coefficient of SS curve> \



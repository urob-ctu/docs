---
title: GPU FEL Remote
layout: default
has_children: false
nav_order: 8
mathjax: true
---


# GPU FEL Remote Access Instructions

## Overview
This page provides instructions for accessing GPU FEL cluster systems remotely. It covers the necessary steps to connect to the systems. GPU FEL information can be found at: [GPU FEL](https://gpu.fel.cvut.cz/wiki/).
## Prerequisites
- SSH client installed on your local machine (e.g., OpenSSH, PuTTY).
- Access credentials (username and password or SSH key).
- VPN access for remote connections, if required.

## Steps to Access GPU FEL Remotely
1. **Establish VPN Connection (if not on faculty network)**:
   - Connect to the VPN.
   - How to connect to VPN: [VPN Instructions](https://svti.fel.cvut.cz/en/services/vpn.html).
  
2. **Open Terminal or SSH Client**:
    - For Linux/Mac: Open the terminal application.
    - `ssh username@gpu.fel.cvut.cz`
    - Provide your CTU password when prompted (KOS password).
    - Optionally, you can use an SSH key for authentication.
        -  generate an SSH key pair using `ssh-keygen`. e.g., `ssh-keygen -t ed25519 -f ~/.ssh/NAME_OF_KEY`
        -  copy the contents of the public key (`~/.ssh/NAME_OF_KEY.pub`) to the server `~/.ssh/authorized_keys` on the server (you can use `ssh-copy-id -i ~/.ssh/NAME_OF_KEY.pub username@gpu.fel.cvut.cz`).
     -  Optionally, create or edit the SSH config file (`~/.ssh/config`) to simplify the connection:
        ```
        Host gpu.fel
            HostName gpu.fel.cvut.cz
            User your_username
            IdentityFile ~/.ssh/NAME_OF_KEY
        ```
        Then connect using `ssh gpu.fel`.
3. **Set Up Remote Deployment in VS Code (Optional)**:
    - Install SFTP by Natizyskunk extension in VS Code.
    - Create a configuration file (`sftp.json`) in your project directory and subdirectory `.vscode` with the following content:
        ```json
        {
            "host": "gpu.fel.cvut.cz",
            "username": "your_username",
            "remotePath": "/home.nfs/your_username/path_to_your_project",
            "openSsh": true,
            "uploadOnSave": true,
            "ignore": [".git", ".venv", "__pycache__", "*.pt", "*.pth",".vscode"],
            "concurrency": 4
        }

        ```
    - cmd+shift+p -> SFTP: Sync Local -> Remote
    - After that, every time you save a file, it will be uploaded to the server.
4. **Copy Files to/from the Server**:
    - Use `scp` or `rsync` to copy files between your local machine and the server.
    - Example using `scp`:
        - To upload a file: `scp /path/to/local/file username@gpu.fel.cvut.cz:/home.nfs/username/path_to_your_project`
        - To download a file: `scp username@gpu.fel.cvut.cz:/home.nfs/username/path_to_your_project/file /path/to/local/destination`

## Run Jobs on GPU FEL
- Once connected, you can submit jobs to the GPU FEL cluster using the appropriate job scheduler commands (e.g., `sbatch`, `srun`).
- On the cluster, we use SLURM as a job scheduler. You can find more information about SLURM commands and usage in the [SLURM documentation](https://slurm.schedmd.com/documentation.html).
- !!! DO NOT run heavy computations directly on the login node. Always submit jobs to the compute nodes using the job scheduler. !!!
- To load models use: `module load` or simply `ml` command.
- To see avalilable modules use: `module avail` or `ml av`.
- To see exact model use: `ml spider <model_name>`.
- To start an interactive session on a compute node, use:
    ```
    srun -p fast --gres=gpu:1 --pty bash -i
    ```
- If you want to run a script, create a SLURM job script (e.g., `job.sh`) and submit it using:
    ```
    sbatch job.sh
    ```
- We provide a submit script `train_sbatch.sh` that you can modify and use to submit your training jobs. Make sure to adjust the parameters according to your requirements and also set the correct path to your Python environment.
- To check the status of your jobs, use:
    ```
    squeue -u your_username
    ```
- To cancel a job, use:
    ```
    scancel job_id
    ```

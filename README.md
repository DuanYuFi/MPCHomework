# MPC Framework Homework

## Prerequisite

- Install miniconda, see [Quick command line install - miniconda](https://docs.anaconda.com/free/miniconda/#quick-command-line-install):

  ```bash
  mkdir -p ~/miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  rm -rf ~/miniconda3/miniconda.sh

  export PATH=$PATH:~/miniconda3/bin
  conda --version
  ```

- prepare the python env and install python code formatter:

  ```bash
  # python3.12
  conda create --prefix ./venv python=3.12
  conda init zsh # for zsh shell
  conda activate ./venv

  pip3 install --upgrade pip
  
  # check python code format
  pip3 install pre-commit
  pre-commit install
  pre-commit run --all-files
  ```

- install python package dependencies

  ```bash
  pip3 install torch torchvision
  ```

## LeNet-5 Privacy Machine Learning

- Try to train LeNet-5 in local to get weights and parameters, these params will used in MPC.

  ```bash
  python3 src/lenet_torch_train.py
  ```

- Try to infer LeNet-5 in MPC enviroments, only Player-0 will load the picture as model private input.

  ```bash
  cd src/
  bash ./run_script.py ./lenet_mpc_infer.py
  ```

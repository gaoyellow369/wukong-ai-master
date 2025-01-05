## Installation

#### 1. Create a Conda environment
```shell
conda create --name wukong python=3.10
conda activate wukong
```

#### 2. First install major packages with Conda, then use `pip` to install other libraries

```shell
# Use Conda to install major libraries
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Use pip to install the remaining libraries
pip install --upgrade pip

pip install -r requirements.txt

```

#### 3. Check the installation results
After installation, confirm that all libraries are correctly installed:

```shell
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print('GPU is', 'available' if torch.cuda.is_available() else 'not available')"

```

## Main File Descriptions
- window.py: Definitions of rectangle coordinates for various health bars on the screen
- judge.py: Calculation of reward points
- restart.py: Logic for automatically returning to the boss from the land temple after death
- training.py: Training script, press T to pause or resume training

## Starting Training
1. Set your game resolution to 1280 * 720, move the game window to the most top left corner of your screen.
2. Use `python -m utils.find_health_location` to check if your health bar of the player displays correctly. 
2. Start the script
```shell
python training.py
```
Alternatively, you may use our combat simulator for a quick training to see the results:
```shell
python combat_env_train.py
```

## Some code is from the following repositories, thanks for open sourcing
- https://github.com/analoganddigital/DQN_play_sekiro
- https://github.com/Sentdex/pygta5
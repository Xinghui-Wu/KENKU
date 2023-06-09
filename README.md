# KENKU

This repository is the core implementation of the black-box audio adversarial attack framework proposed by our paper (KENKU: Towards Efficient and Stealthy Black-box Adversarial Attacks against ASR Systems).

## Installation

1. Clone this repo.

```bash
git clone https://github.com/Xinghui-Wu/KENKU.git
cd KENKU
```

2. Create a virtual environment running Python 3.7 interpreter of later.
3. Install the dependencies.

```bash
pip install -r requirements.txt
```

Notice that we used a GPU server equipped with a NVIDIA GeForce RTX 3090 card and leveraged the PyTorch framework on the CUDA 11.0 platform to solve the optimization problems.


## Usage

1. Register the ASR cloud services provided by the target manufacturers and fill in the relevant information in the account.json file.
2. Create two folders, namely songs/ and commands/, under the root directory of the KENKU project.
3. Prepare a few song clips in the WAV format and place them in the songs/ folder.
4. Specify the desired target command texts in commands/commands.txt (one line one sentence) and use text_to_speech.py to synthesize audio files for those target commands.
```bash
python text_to_speech.py
```
5. Launch the hidden voice command attack or the integrated command attack supported by KENKU.
```bash
python hidden_voice_command_attacks.py
python integrated_command_attack.py
```
6. Test the generated audio adversarial examples on the black-box commercial ASR platforms. Provide the generated csv file in step 5 to the black_box_asr.py script as the input.csv file. The output.csv file will contain the transcribed results.
```bash
python black_box_asr.py -i input.csv -o output.csv
```

Notice that we used the two folder names, songs/ and commands/, to configure the default command line arguments for all the Python scripts in this project. 
If you would like to rename the corresponding folders or files, you must provide the correct values for those related command line arguments.

In addition, the two attack scripts involve many configurable hyperparameters that can be fine-tuned to improve the attack performance.
We do not guarantee the default settings are the best and we encourage the project users to test more cases.

Please use the following commands for more help.

```bash
python text_to_speech.py -h
python hidden_voice_command_attacks.py -h
python integrated_command_attack.py -h
```


## Citation

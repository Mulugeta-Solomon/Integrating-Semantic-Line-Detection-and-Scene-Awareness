# Integrating Semantic Line Detection and Scene Awareness
Implementation of  "Integrating Semantic Line Detection and Scene-Awareness" RSJ2024

![image](asset/architecture.png)

## Data Downloading
The training and testing datasets (including the Wireframe dataset and YorkUrban dataset) can be downloaded via [Google Drive](https://drive.google.com/file/d/134L-u9pgGtnzw0auPv8ykHqMjjZ2claO/view). Many thanks to the authors of these excellent datasets!



For annotating the dataset, please refer to the annotation tool available at [this link](https://github.com/Mulugeta-Solomon/LineAnnotationTool).
The full annotated data will released soon. 

## Installation
### Setting Up the Virtual Environment
Clone the repository:
```sh
git clone https://github.com/Mulugeta-Solomon/Integrating-Semantic-Line-Detection-and-Scene-Awareness.git
cd Integrating-Semantic-Line-Detection-and-Scene-Awareness
```
Install ninja-build by: 
``` 
sudo apt install ninja-build.
```

Create and activate a virtual environment:
```sh
python -m venv torchenv
source torchenv/bin/activate  # On Windows use: venv\Scripts\activate
```
Install pytorch:
```
# Install pytorch, please be careful for the version of CUDA on your machine
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116 

```

Install other dependencies:
```sh
pip install --upgrade pip
pip install -r requirements.txt
```
Verify the installation.
```
python -c "import torch; print(torch.cuda.is_available())" # Check if the installed pytorch supports CUDA.
```

## Citation 
If you find this work helpful in your research, please cite it as follows:

```bibtex
@inproceedings{RSJ2024AC1G1-02,
  title     = {Integrating Semantic Line Detection and Scene-Awareness},
  author    = {Mulugeta Solomon Abate and Tian Yang and Kazuhiro Shimonomura},
  booktitle = {Proceedings of RSJ2024},
  year      = {2024}
}
```

## Acknowledgement
This project is built on [hawp](https://github.com/cherubicXN/hawp). We acknowledge [Dolphin Mulugeta](https://dododoyo.github.io/) for his contribution to the annotation tool. We also acknowledge the author of the wireframe dataset.


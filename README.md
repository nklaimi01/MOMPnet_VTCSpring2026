# Physically constrained unfolded multi-dimensional OMP for large MIMO systems

Implementation of the methods proposed in the paper:

>📄 [Physically constrained unfolded multi-dimensional OMP for large MIMO systems](https://arxiv.org/pdf/2601.10771)  
> Nay Klaimi, Clément Elvira, Philippe Mary, Luc Le Magoarou  
> VTC Spring 2026
## Getting Started
⚠️ This project requires **Python 3.9**.

If you are using a different Python version, make sure to downgrade to Python 3.9 in your environment to avoid compatibility issues.

### Setting Up the Environment
It is **recommended** to create a virtual environment before installing dependencies.
#### 🧪 Option 1: Using `venv` (Standard Python Virtual Environment)
```bash
# Create virtual environment with Python 3.9
python3.9 -m venv myenv

# Activate the environment
source myenv/bin/activate      # Unix/macOS
myenv\Scripts\activate         # Windows
```
#### 🐍 Option 2: Using `conda`
```bash
# Create and activate a conda environment with Python 3.9
conda create -n myenv python=3.9
conda activate myenv
```
### 📦 Install Dependencies
To install all required Python packages, run the following command:
```bash
pip install -r requirements.txt
```
### 🔧 Generating Simulation Data  
We use the **[Sionna Ray Tracing library](https://nvlabs.github.io/sionna/)** to generate the channel data required for training and testing.  

Run the script [`HD_data_gen.py`](./HD_data_gen.py):  
```bash
python HD_data_gen.py
```

The script [`saved_data_loader.py`](./saved_data_loader.py) imports the previously generated data to all scripts containing 
```bash 
from saved_data_loader import *
```
## 📚 Citation
Please consider citing the original paper if this code contributes to your work.
```bibtex
@misc{klaimi2026physicallyconstrainedunfoldedmultidimensional,
      title={Physically constrained unfolded multi-dimensional OMP for large MIMO systems}, 
      author={Nay Klaimi and Clément Elvira and Philippe Mary and Luc Le Magoarou},
      year={2026},
      eprint={2601.10771},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2601.10771}, 
}
```
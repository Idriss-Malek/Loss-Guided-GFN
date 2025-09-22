
# Instructions
 
## Installation
 
To install the required dependencies, run:
 
```bash
git clone --branch dag_gfn --single-branch https://github.com/sharmaabhijith/torchgfn.git

cd torchgfn

pip install -e ".[all]"

pip install transformers

```
 
## Project Structure
 
```plaintext

.

├── src

│   ├── algos       # Training methods for GFlowNets

│   ├── envs        # Environments required for experiments

│   ├── gflownet    # Modified trajectory balance implementation

│   └── utils       # Utilities for training and evaluation

├── examples        # Scripts to train a GFlowNet on each environment

└── notebooks       # Jupyter notebook demonstrating learning in the hypergrid

```
 
## Usage
 
1. Navigate to the project root:

   ```bash

   cd torchgfn

   ```

2. Launch one of the example scripts:

   ```bash

   python examples/<environment>.py

   ```

3. To explore training behavior on the hypergrid:

   ```bash

   jupyter notebook notebooks/hypergrid_example.ipynb

   ```

 

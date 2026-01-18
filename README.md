# Project DFB: Motor Sound Anomaly Detection

**Students:** Rafael Pardo Rueda, Juan Luis Torres Ramos  
**Context:** Project DFB (Extends Lab 2) 


## Overview
An unsupervised system to detect faults in electrical motors using audio analysis. It learns the normal operational signature and flags deviations as anomalies.

## Pipeline

The complete pipeline is structured as follows:

1. **Audio Preprocessing**  
   Raw motor sound recordings are converted into time–frequency representations (Mel-spectrograms).

2. **Latent Feature Extraction (Autoencoder)**  
   A Convolutional Autoencoder (CAE) is trained to reconstruct spectrograms corresponding to healthy motor operation.  
   The encoder compresses each input into a low-dimensional latent representation, reducing dimensionality and filtering noise.

3. **Anomaly Detection**  
   Latent vectors are used as input for two anomaly detection algorithms:
   - **Isolation Forest**
   - **One-Class Support Vector Machine (OC-SVM)**

4. **Evaluation and Comparison**  
   The models are evaluated using standard anomaly detection metrics (precision, recall, F1-score, FPR, FNR), and their performance is compared under identical test conditions.

## Project Structure
```text
.
├── data/               # Raw audio & processed numpy arrays
├── models/             # Saved Autoencoder (.h5) & ML models (.pkl)
├── src/
│   ├── preprocessing.py                 # Audio preprocessing utilities
│   ├── Autoencoder.ipynb                # Latent feature extraction (CAE)
│   ├── Anomaly_Detection_IF_OCSVM.ipynb # Anomaly detection and evaluation
│   └── Test/                            # Auxiliary scripts used for preliminary testing
│       ├── ad_models.py
│       └── pipeline.py
├── Dockerfile          # Environment configuration
└── requirements.txt    # Python dependencies
```

Note:
Data and trained models are not included in the repository due to size constraints.

## How to Run
We are going to use Docker Engine.

build it take 3-8 mins (just wait)

```bash
docker build -t dfb-project . 
docker run -it -v ${PWD}:/app dfb-project

root@2e1b13483c26:/app# ls
Dockerfile  LICENSE  README.md  data  documents  models  notebooks  requirements.txt  src
```

with jupyter to work 
```bash
docker run -it -p 8888:8888 -v $(pwd):/app dfb-project
root@2e1b13483c26:/app#  jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

use the url to connect it 

goes inside the container 


the place where you are going to try the scripts 

another terminal 
```bash
docker ps
docker exec -it <ID> bash
```

## License
Academic Use / MIT License.

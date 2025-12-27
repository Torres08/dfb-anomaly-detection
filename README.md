# Project DFB: Motor Sound Anomaly Detection

**Students:** Rafael Pardo Rueda, Juan Luis Torres Ramos  
**Context:** Project DFB (Extends Lab 2) | **Deadline:** Jan 17, 2025


## Overview
An unsupervised system to detect faults in electrical motors using audio analysis. It learns the normal operational signature and flags deviations as anomalies.

**Pipeline:**
1.  **Latent Representation (Part A):** A Convolutional Autoencoder (CAE) compresses raw audio spectrograms into a compact feature vector.
2.  (Juan Luis)
3.  **Anomaly Detection (Part B):** An Isolation Forest (or SVM) classifies these vectors to identify non-nominal behaviors. (Rafa)


## Project Structure
```text
.
├── data/               # Raw audio & processed numpy arrays
├── models/             # Saved Autoencoder (.h5) & ML models (.pkl)
├── src/
│   ├── preprocessing.py    # Audio -> Mel-Spectrograms
│   ├── autoencoder.py      # Deep Learning Architecture
│   ├── ad_models.py        # Isolation Forest / SVM Logic
│   └── pipeline.py         # Main execution script
├── Dockerfile          # Environment configuration
└── requirements.txt    # Dependencies
```

Data and models does not upload to github.

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
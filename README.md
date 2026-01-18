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

**Note:**
Data and trained models are not included in the repository due to size constraints.

## Docker Environment and Execution

To ensure reproducibility and avoid dependency conflicts, the entire project is executed inside a **Docker container**. The container includes all required Python libraries and system dependencies needed to run the notebooks and scripts.

### Build the Docker Image

From the root directory of the repository, build the Docker image using:

```bash
docker build -t dfb-project .
```
The build process may take several minutes, depending on the host system and network speed.

### Run the Docker Container (Interactive Mode)
```bash
docker run -it -v ${pwd}:/app dfb-project
```
This command mounts the project directory into the container, allowing all source files and notebooks to be accessed and modified from within the Docker environment.

### Run Jupyter Lab Inside Docker
To work with the Jupyter notebooks, the container must expose a network port:
```bash
docker run -it -p 8888:8888 -v $(pwd):/app dfb-project
```

Once inside the container, launch Jupyter Lab with:
```bash
jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```
Jupyter will display a URL in the terminal. Open this URL in a web browser on the host machine to access the notebook interface.

### Access a Running Container from Another Terminal

If the container is already running and a new terminal session is required, first list active containers:

```bash
docker ps
```

Then attach to the desired container using:

```bash
docker exec -it <CONTAINER_ID> bash
```

This allows multiple terminals to interact with the same running Docker environment.

### Execution Notes

- All notebooks must be executed **from within the Docker container**.
- No manual dependency installation is required outside Docker.
- Audio data and trained models must be placed in the appropriate directories before execution.


## License
Academic Use / MIT License.

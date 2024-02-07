# CUDA-GSSL

CUDA-GSSL is a library for Graph Semi-Supervised Learning (GSSL) implemented in CUDA.


## Installation

To compile CUDA-GSSL, follow these steps (Assuming make, Cudatoolkit already present):

1. Clone the repository:

    ```bash
    git clone https://github.com/aromalma/cuda-gssl.git
    ```

2. Install Pybind11:

    ```bash
    pip install pybind11
    ```

3. Navigate to the cloned repository:

    ```bash
    cd cuda-gssl
    ```

4. Build the library:

    ```bash
    make
    ```

## Demo

To run the demo, execute the following command:

```bash
python3 demo.py

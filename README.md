# MultiLayerPerceptrons_Workshop

A hands-on workshop on **Neural Networks and Multi-Layer Perceptrons (MLPs)** for CSCN8010 (Foundations of Machine Learning).

The notebook starts with a single artificial neuron (the perceptron) and builds up to a full MLP. It then:

1. **Challenge 1:** compares the same MNIST digit classifier written in **Keras**, **PyTorch** and **TensorFlow (Core API)**, in terms of accuracy and speed.
2. **Challenge 2:** builds a neural network **from scratch with NumPy**, with forward propagation, backpropagation and gradient descent, and runs experiments on activation functions, learning rates, depth, vanishing/exploding gradients and weight initialisation.
3. **Extension:** uses the from-scratch network to solve **XOR** and classify **MNIST digits**, then compares it with the frameworks.

---

## 📁 Repository contents

| File / folder | Description |
|---|---|
| `MultiLayerPerceptrons_Workshop.ipynb` | **Main notebook**: theory, the three framework implementations, and the completed Challenge 1, Challenge 2, reflection and Extension |
| `MultiLayeredPerceptrons.ipynb` | Companion tutorial notebook (perceptron, MLP, activation functions, feedforward, backprop, hyperparameters) |
| `requirements.txt` | Python packages needed to run the notebooks |
| `data/` | *(created automatically, not committed)* MNIST files downloaded by `torchvision` |

---

## 🧠 Summary of the work

### Challenge 1: Keras vs PyTorch vs TensorFlow on MNIST

All three implementations train the **same** network, `784 → 128 (ReLU) → 64 (ReLU) → 10`, for 5 epochs with batch size 64 and the Adam optimizer.

| Framework | Speed (original run) | Test accuracy | Pros | Cons |
|---|---|---|---|---|
| Keras | 7.42 s | 0.9720 | Least code, built-in validation & metrics, fast | Hides the training loop; less flexible |
| PyTorch | 32.25 s | 0.9647 | Explicit, Pythonic, easy to debug; research standard | More code; slow here because of the per-image `transforms` data pipeline |
| TensorFlow Core | 8.84 s | 0.9748 | Full control via `GradientTape`; `@tf.function` keeps it fast | Most verbose; graph tracing can be confusing |

Key findings:
- **Accuracy is effectively a tie** (~96–98%). The differences come from random initialisation, normalisation and data splits, not from the frameworks.
- **PyTorch's slow time was caused by data loading, not the framework.** When PyTorch uses the same pre-loaded arrays as Keras, it trains about **5× faster** (≈7.6 s, 0.974 accuracy).
- **Batch size** has the biggest effect on speed: batch 512 trained about 6× faster than batch 32, but reached lower accuracy in the same number of epochs.
- **Preferred for MNIST: Keras.** It is simple, fast, accurate and has built-in validation. PyTorch is the better choice for learning/research, and TF Core for custom training steps.

### Challenge 2: Forward & backpropagation from scratch (NumPy only)

- A `NumpyMLP` class that implements $Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$, $A^{[l]} = g(Z^{[l]})$, cross-entropy loss, backpropagation ($dZ$, $dW$, $db$) and gradient descent updates.
- A **gradient check** that compares backprop to numerical derivatives (relative error ≈ 1e-10, which confirms the maths is correct).
- Training on the non-linear **two-moons** dataset (95% test accuracy) with a loss curve and decision-boundary plot.
- Experiments and visualisations:

| Experiment | What we saw |
|---|---|
| Activation functions | ReLU/tanh learn fast; sigmoid is much slower (loss 0.29 vs 0.09) |
| Learning rate | 0.001 too slow; 0.1–1.0 good; 5.0 overshoots → dead ReLUs, stuck at chance |
| Depth | A 3rd hidden layer helps only slightly on a simple 2D problem |
| Vanishing gradients | In an 8-hidden-layer sigmoid network, layer-1 gradients are >100,000× smaller than the last layer's, so the network learns nothing. ReLU trains fine |
| Initialisation | Tiny weights → signal vanishes; large weights → signal explodes (`nan`); He initialisation keeps it stable |

- Plots of the **forward flow of activations**, the **backward flow of gradients**, and a **diagram** of how data and gradients move through the layers.
- A written **reflection** (150–200 words) on learning dynamics, initialisation and activations.

### Extension: XOR and MNIST with the NumPy network

- **XOR** is solved perfectly (outputs ≈ 0.0002 / 0.9997).
- **MNIST** with the same 784-128-64-10 architecture and plain mini-batch SGD reaches **≈97.4% test accuracy in ~5 s**, on par with the frameworks. This shows that the frameworks do the same maths; what they add is automatic differentiation, GPU support and tooling.

> ℹ️ Timings depend on your computer, and accuracies change slightly on every run because weights start randomly.

---

## ▶️ How to run the notebook

### 1. Prerequisites
- **Python 3.10 or newer** (Python 3.12/3.13 is the safest choice for TensorFlow + PyTorch).
- **Git**
- **VS Code** with the *Python* and *Jupyter* extensions, **or** JupyterLab / Jupyter Notebook.

### 2. Clone the repository

```bash
git clone https://github.com/<your-username>/MultiLayerPerceptrons_Workshop.git
```

```bash
cd MultiLayerPerceptrons_Workshop
```

### 3. Create a virtual environment and install the packages

**Option A: using `pip`**

Windows (PowerShell):
```bash
python -m venv .venv
```
```bash
.venv\Scripts\Activate.ps1
```
```bash
pip install -r requirements.txt
```

macOS / Linux:
```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

**Option B: using [`uv`](https://docs.astral.sh/uv/)**
```bash
uv venv
```
```bash
uv pip install -r requirements.txt
```

### 4. Open and run the notebook

- **VS Code:** open the folder, open `MultiLayerPerceptrons_Workshop.ipynb`, click **Select Kernel** (top right) and choose the `.venv` interpreter. Then click **Run All**.
- **Jupyter:** run the command below, then open the notebook in the browser and choose **Run → Run All Cells**.

```bash
jupyter notebook
```

Run the cells **from top to bottom**. Later cells re-use variables from earlier ones (for example `x_train_flat`, the training times and the `NumpyMLP` class).

### 5. What to expect
- The first run **downloads MNIST** (~11 MB for Keras plus ~60 MB for `torchvision` into `data/`). After that it is cached.
- Running everything takes about **2–4 minutes on a typical laptop CPU**. No GPU is needed.
- Keras/TensorFlow print some informational log messages. These are normal.

### Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure the notebook kernel is the `.venv` where you ran `pip install`. |
| TensorFlow fails to install | Use Python 3.10–3.13. TensorFlow wheels can lag behind the newest Python release. |
| `LLVM ERROR: ... registered more than once` (Linux) | A version clash between TensorFlow and PyTorch wheels. Install `torch==2.9.*` and `torchvision==0.24.*`. |
| MNIST download fails | Check your internet connection, or delete the `data/` folder and re-run. |

---

## 📚 References
- [3Blue1Brown: Neural Networks](https://www.3blue1brown.com/lessons/neural-networks)
- [3Blue1Brown: What is backpropagation really doing?](https://www.3blue1brown.com/lessons/backpropagation)
- [CSCN8010 course notebooks](https://github.com/CSCN8010/CSCN8010)

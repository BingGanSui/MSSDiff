# MSSDiff

This is the official code repository for the paper **'MSSDiff: Diffusion-based Super-Resolution and Bias Correction for Sub-seasonal to Seasonal Ensemble Forecasts'**.

## Training MSSDiff

### 1. Pre-train RRDBNet
For the configuration used in the paper, refer to `conf/rrdb_config.yaml`.
Start the RRDBNet training using the following command:

```bash
python main.py train --config conf/rrdb_config.yaml <experiment name>
```

### 2. Extract RRDBNet Parameters
You can extract the RRDBNet parameters for a specific epoch by modifying the `pairs` list in `extract.py`. This specifies the experiment name, dataset name, and the target epoch.

For example, to extract parameters from experiment `exp_Asia_cur` on dataset `Asia_cur` at epoch 375, modify the `pairs` list as follows:

```python
pairs = [
    ['exp_Asia', 'Asia', '375'],
]
```

Then, run `extract.py`:

```bash
python extract.py
```

### 3. Train MSSDiff
For the configuration used in the paper, refer to `conf/mssdiff_config.yaml`.
Start the MSSDiff training using the following command:

```bash
python main.py train --config conf/mssdiff_config.yaml <experiment name>
```

## Demo

The `sample` directory provides sample data and scripts for testing and reference.

### Contents
- `20190205.pt`: Sub-seasonal forecast data from the FGOALS-f numerical model for February 5, 2019 (Internal ID: 05).
- `mssdiff_sample.pt`: Pre-trained MSSDiff model parameters.
- `pretrained-rrdbnet-Asia_cur.pt`: Pre-trained RRDBNet parameters.

### Running the Demo
Follow these steps to perform inference and visualization:

1. **Run Inference**:
   This generates the super-resolution result using the sample data and pre-trained models.
   ```bash
   python sample/inference_paper.py
   ```

2. **Visualize Results**:
   This generates visualization plots for the Super-Resolution (SR), Low-Resolution (LR), and High-Resolution (HR) data.
   ```bash
   python sample/visualize_result.py
   ```

After running these scripts, you will find the generated images (`visualization_SR.png`, `visualization_LR.png`, `visualization_HR.png`) in the `sample` directory.

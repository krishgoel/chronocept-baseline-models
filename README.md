# Chronocept Baseline Models
> **Publication**: [Chronocept: Instilling a Sense of Time in Machines](https://arxiv.org/abs/2505.07637)  
**Authors:** Krish Goel, Sanskar Pandey, KS Mahadevan, Harsh Kumar, and Vishesh Khadaria  

> **Dataset**: [huggingface/krishgoel/chronocept](https://huggingface.co/datasets/krishgoel/chronocept)  

This repository contains baseline implementations for Chronocept, the world's first benchmark for modeling validity of textual information as continuous probability distributions over time. The models predict three parameters (location ξ, scale ω, skewness α) that characterize the temporal relevance of textual information using a skew-normal distribution, over a logarithmically transformed time axis.

## Repository Structure

The repository is organized to retain both the original `v1` baselines and the improved `v2` baselines:

- **`v1/`**: Original baseline implementations.
  - `models/`: Model definitions (e.g., SBERT, BiLSTM, FFNN).
  - `utils/`: DataLoader and metric utilities.
  - `experiments/`: Jupyter notebooks containing all experiment code for `v1` models. 
- **`v2/`**: Improved baselines introduced (e.g., MT-DNN, Skew-Normal NLL variants).
  - `models/`: Improved modular model components (encoders, heads, pooling).
  - `utils/`: Updated DataLoader and training utilities.
  - `experiments/`: Jupyter notebooks for ablation studies and benchmarks using the `v2` codebase. 

## `DataLoader` Implementation
The `DataLoader` class is used to load and preprocess the data. It supports multiple embedding methods and data splits. Depending on the baseline version, use either `from v1.utils import DataLoader` or `from v2.utils import ImprovedDataLoader`.

### Parameters (v1 `DataLoader`)

- **`benchmark`** (`Literal["benchmark_1", "benchmark_2"]`):  
  Benchmark identifier.

- **`split`** (`"train" | "validation" | "test" | List[str] | None`, default=`None`):  
  Which data splits to load. If `None`, all splits are loaded.

- **`embedding`** (`str`, default=`"bert_cls"`):  
  Embedding method:
  - `"bert_cls"`: CLS token from BERT.
  - `"bert_full"`: Mean pooling over BERT tokens.
  - `"bert_sequential"`: Full token sequence from BERT.
  - `"sbert"`: Sentence-BERT (`all-MiniLM-L6-v2`).
  - `"tfidf"`: TF-IDF (max 512 features).
  - `"glove"`: Averaged GloVE embeddings.

- **`max_len`** (`int`, default=`128`):  
  Maximum sequence length for BERT-based tokenization.

- **`include_axes`** (`bool`, default=`False`):  
  Whether to include the 8 temporal axes with the `parent_text`. Final input = parent embedding + flattened axes.

- **`shuffle_axes`** (`bool`, default=`False`):  
  Shuffle the 8 axes per sample (only if `include_axes=True`).

- **`normalization`** (`"zscore" | "minmax" | "none" | None`, default=`"zscore"`):  
  Target value normalization:
  - `"zscore"`: Zero mean, unit variance.
  - `"minmax"`: Scale to [0, 1].
  - `"none"` or `None`: No normalization.

- **`log_scale`** (`float`, default=`1.1`):  
  Base for logarithmic transformation of target values. If set, `log(y) / log(log_scale)` is applied.

### Usage Example
```python
from v1.utils import DataLoader

# Initialize loader
dl = DataLoader(
    benchmark="benchmark_1",
    split=None,  # Loads all splits
    embedding="bert_cls",
    max_len=128,
    include_axes=True,
    normalization="zscore"
)

# Preprocess data
data = dl.preprocess()  # Returns dict with all splits if split=None
X_train, y_train = data["train"]
X_valid, y_valid = data["validation"]
X_test, y_test = data["test"]
```

## Performance Statistics

### Benchmark I (1254 samples)
| Baseline     | MSE          | MAE        | R²         | NLL        |
| ------------ | ------------ | ---------- | ---------- | ---------- |
| DEBERTA-V3   | 695.7505     | 14.8318    | -1.3404    | 9.9973     |
| Z DISTILBERT | 737.3157     | 15.3509    | -1.5092    | 11.3089    |
| MT-DNN       | **108.3768** | **5.9425** | **0.0314** | 4.5117     |
| ROBERTA      | 909.5181     | 17.2330    | -1.8687    | 15.2214    |
| SBERT-BILSTM | 171.7044     | 8.5698     | -2.3193    | 4.4084     |
| SBERT-FFNN   | 155.8747     | 7.9780     | -2.0203    | **4.3769** |


### Benchmark II (524 samples)
| Baseline     | MSE         | MAE        | R²          | NLL        |
| ------------ | ----------- | ---------- | ----------- | ---------- |
| DEBERTA-V3   | 544.5860    | 13.7358    | -2.7730     | 11.6107    |
| Z DISTILBERT | 640.2664    | 14.7484    | -3.0546     | 13.6647    |
| MT-DNN       | **64.5708** | **4.5253** | **-0.0183** | 4.1865     |
| ROBERTA      | 748.8589    | 16.0097    | -3.5666     | 15.2253    |
| SBERT-BILSTM | 107.4771    | 6.0472     | -1.1621     | 4.1373     |
| SBERT-FFNN   | 153.2178    | 8.5744     | -6.3268     | **3.9842** |

Note: All metrics are computed on Z-score normalized targets on the test set.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation
If you use Chronocept in your work, please cite the following:
```bibtex
@misc{goel2025chronocept,
    title={Chronocept: Instilling a Sense of Time in Machines}, 
    author={Krish Goel and Sanskar Pandey and KS Mahadevan and Harsh Kumar and Vishesh Khadaria},
    year={2025},
    eprint={2505.07637},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2505.07637}, 
}
```

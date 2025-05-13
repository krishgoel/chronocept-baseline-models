# Chronocept Baseline Models
> **Publication**: [Chronocept: Instilling a Sense of Time in Machines](https://arxiv.org/abs/2505.07637)  
**Authors:** Krish Goel, Sanskar Pandey, KS Mahadevan, Harsh Kumar, and Vishesh Khadaria  

> **Dataset**: [huggingface/krishgoel/chronocept](https://huggingface.co/datasets/krishgoel/chronocept)  

This repository contains baseline implementations for Chronocept, the world's first benchmark for modeling validity of textual information as continuous probability distributions over time. The models predict three parameters (location ξ, scale ω, skewness α) that characterize the temporal relevance of textual information using a skew-normal distribution, over a logarithmically transformed time axis.

## `DataLoader` Implementation
The `DataLoader` class [utils/dataloader.py](utils/dataloader.py) is used to load and preprocess the data. It supports multiple embedding methods and data splits.

### Parameters

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
from utils import DataLoader

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
| Baseline | MSE    | MAE    | R²     | NLL    | Spearman |
|----------|--------|--------|--------|--------|----------|
| LR       | 1.3610 | 0.9179 | -0.3610| 1.5730 | 0.2338   |
| XGB      | 0.8884 | 0.7424 | 0.1116 | 1.3598 | 0.2940   |
| SVR      | 0.9067 | 0.7529 | 0.0933 | 1.3700 | 0.3281   |
| FFNN     | **0.8763** | **0.7284** | **0.1237** | **1.3529** | **0.3543**   |
| Bi-LSTM  | 0.9203 | 0.7571 | 0.0797 | 1.3774 | 0.2367   |
| BERT     | 145.8611 | 6.7570 | -0.0090| 3.9103 | -0.0485  |


### Benchmark II (524 samples)
| Baseline | MSE    | MAE    | R²     | NLL    | Spearman |
|----------|--------|--------|--------|--------|----------|
| LR       | 1.1009 | 0.8361 | -0.1009| 1.4670 | 0.3279   |
| XGB      | 0.9580 | 0.8011 | 0.0420 | 1.3975 | 0.2331   |
| SVR      | 0.8889 | 0.7740 | 0.1111 | 1.3601 | 0.3293   |
| FFNN     | **0.8715** | **0.7583** | 0.1285 | 1.3502 | 0.3437   |
| Bi-LSTM  | **0.8702** | 0.7646 | **0.1298** | **1.3494** | **0.3535**   |
| BERT     | 68.1507 | 4.6741 | -0.1122| 3.5299 | -0.2407  |

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

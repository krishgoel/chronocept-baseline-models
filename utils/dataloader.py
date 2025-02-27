import datasets
import numpy as np
import torch
import random
import logging
from transformers import BertTokenizer, BertModel
from typing import Literal, Tuple, Optional, Union, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataLoader:
    def __init__(
        self, 
        benchmark: Literal["benchmark_1", "benchmark_2"],
        split: Optional[Union[Literal["train", "validation", "test"], List[Literal["train", "validation", "test"]]]] = None, 
        embedding: Literal["bert_cls", "bert_full", "bert_sequential", "sbert", "tfidf", "glove"] = "bert_cls", 
        max_len: int = 128,
        include_axes: bool = True,
        shuffle_axes: bool = False,
        normalization: Literal["zscore", "minmax", "none", None] = "zscore"
    ):
        """
        Args:
            benchmark (Literal["benchmark_1", "benchmark_2"]): Benchmark identifier.
            split (Optional[Union["train", "validation", "test", List["train", "validation", "test"]]]): Data split(s) to load.
                If None, all splits (train, validation, test) are loaded.
            embedding (Literal[...]): Embedding method.
                - "bert_cls": BERT embedding using the CLS token.
                - "bert_full": BERT embedding using mean pooling over all tokens.
                - "bert_sequential": BERT embedding returning the full token sequence.
                - "sbert": Sentence-BERT embedding.
                - "tfidf": TF-IDF vectorization.
                - "glove": GloVE-based averaging.
            max_len (int): Maximum sequence length for tokenization.
            include_axes (bool): Whether to include the 8 temporal axes in addition to parent_text.
                When True, the final input vector is the concatenation of the parent's embedding and the flattened axes embeddings.
            shuffle_axes (bool): If True, shuffles the axes order per sample.
            normalization (Literal["zscore", "minmax", "none", None]): How to scale target values column-wise.
                - "zscore" subtracts the mean and divides by the standard deviation (default),
                - "minmax" scales values to the [0, 1] range,
                - "none" or None applies no normalization.
        """
        logger.info("Initializing DataLoader...")
        if benchmark not in {"benchmark_1", "benchmark_2"}:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
        if normalization not in {"zscore", "minmax", "none", None}:
            raise ValueError(f"Unsupported normalization method: {normalization}")
            
        self.benchmark = benchmark
        self.split = split
        self.embedding = embedding.lower()
        self.max_len = max_len
        self.include_axes = include_axes
        self.shuffle_axes = shuffle_axes
        self.normalization_method = normalization

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading dataset for benchmark {self.benchmark} with split {self.split}")
        self.dataset = self._load_dataset()
        logger.info("Dataset loaded successfully.")

        self.vectorizer_fitted = False  # For tfidf, to track if fitted
        self.normalization_params = None  # To store normalization parameters

        logger.info(f"Initializing embedding for method {self.embedding}...")
        self.model, self.tokenizer = self._init_embedding()
        logger.info("Embedding resources initialized.")

    def _load_dataset(self):
        """Loads the dataset from Hugging Face."""
        ds_all = datasets.load_dataset("krishgoel/chronocept", self.benchmark)
        if self.split is None:
            logger.info("No split specified; loading all splits (train, validation, test).")
            return ds_all
        elif isinstance(self.split, list):
            logger.info(f"Loading the following splits: {self.split}")
            return {s: ds_all[s] for s in self.split if s in ds_all}
        elif isinstance(self.split, str):
            logger.info(f"Loading split: {self.split}")
            return ds_all[self.split]
        else:
            raise ValueError("Invalid type for split parameter.")

    def _init_embedding(self) -> Tuple[Optional[torch.nn.Module], Optional[BertTokenizer]]:
        """
        Initializes embedding resources based on the embedding method.
        For BERT and SBERT variants, loads pre-trained models.
        For tfidf and glove, initializes vectorizers.
        """
        if self.embedding in ["bert_cls", "bert_full", "bert_sequential"]:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased")
            model.to(self.device)
            logger.info("Initialized BERT model and tokenizer.")
            return model, tokenizer
        elif self.embedding == "sbert":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized Sentence-BERT model.")
            return model, None
        elif self.embedding == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=512)
            logger.info("Initialized TF-IDF vectorizer.")
            return None, None
        elif self.embedding == "glove":
            import gensim.downloader as api
            self.glove_model = api.load("glove-wiki-gigaword-100")
            logger.info("Initialized GloVE model.")
            return None, None
        else:
            raise ValueError(f"Unsupported embedding method: {self.embedding}")

    def get_embeddings(self, texts: list) -> torch.Tensor:
        """
        Generate embeddings for a list of texts based on the selected embedding method.
        """
        if self.embedding == "bert_cls":
            encoded_input = self.tokenizer(
                texts, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            with torch.no_grad():
                outputs = self.model(**encoded_input)
            return outputs.last_hidden_state[:, 0, :]
        elif self.embedding == "bert_full":
            encoded_input = self.tokenizer(
                texts, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            with torch.no_grad():
                outputs = self.model(**encoded_input)
            return outputs.last_hidden_state.mean(dim=1)
        elif self.embedding == "bert_sequential":
            encoded_input = self.tokenizer(
                texts, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            with torch.no_grad():
                outputs = self.model(**encoded_input)
            return outputs.last_hidden_state
        elif self.embedding == "sbert":
            embeddings = self.model.encode(texts)
            return torch.tensor(embeddings, dtype=torch.float32)
        elif self.embedding == "tfidf":
            if not self.vectorizer_fitted:
                logger.info("Fitting TF-IDF vectorizer on provided texts...")
                embeddings = self.vectorizer.fit_transform(texts).toarray()
                self.vectorizer_fitted = True
            else:
                embeddings = self.vectorizer.transform(texts).toarray()
            return torch.tensor(embeddings, dtype=torch.float32)
        elif self.embedding == "glove":
            embeddings_list = [self._glove_embedding(text) for text in texts]
            return torch.stack(embeddings_list)
        else:
            raise ValueError(f"Unsupported embedding method: {self.embedding}")

    def _glove_embedding(self, text: str) -> torch.Tensor:
        """
        Computes the GloVE embedding for a given text by averaging the token embeddings.
        Tokens not found in the vocabulary are skipped.
        """
        logger.debug(f"Computing GloVE embedding for text: {text[:30]}...")
        tokens = text.split()
        glove_vectors = []
        for token in tokens:
            if token in self.glove_model:
                glove_vectors.append(self.glove_model[token])
        if glove_vectors:
            avg_vector = np.mean(glove_vectors, axis=0)
            return torch.tensor(avg_vector, dtype=torch.float32)
        else:
            return torch.zeros(self.glove_model.vector_size, dtype=torch.float32)

    def _normalize_targets(self, targets: np.ndarray) -> np.ndarray:
        """
        Normalizes target values column-wise using the chosen normalization method.
        Stores computed parameters in self.normalization_params.
        """
        logger.info("Normalizing target values...")
        if self.normalization_method == "zscore":
            means = targets.mean(axis=0)
            stds = targets.std(axis=0)
            stds[stds == 0] = 1  # Avoid division by zero
            self.normalization_params = {"mean": means, "std": stds}
            logger.info(f"Z-score parameters: mean={means}, std={stds}")
            return (targets - means) / stds
        elif self.normalization_method == "minmax":
            mins = targets.min(axis=0)
            maxs = targets.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1  # Avoid division by zero
            self.normalization_params = {"min": mins, "max": maxs}
            logger.info(f"Min-max parameters: min={mins}, max={maxs}")
            return (targets - mins) / ranges
        elif self.normalization_method in {"none", None}:
            logger.info("No normalization applied to targets.")
            return targets
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization_method}")

    def _process_single_dataset(self, ds) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """
        Processes a single Hugging Face dataset split:
          - Computes parent's embeddings.
          - If include_axes is True, computes axes embeddings.
            For sequential embeddings (embedding=="bert_sequential"), the parent's sequence [T, D]
            and each axis’s sequence [T, D] are concatenated along the time dimension
            (resulting in [9*T, D]). For non-sequential embeddings, axes embeddings are flattened.
          - Normalizes targets if available.
        
        Returns:
            A tuple (features, targets), where features is a tensor and targets is a NumPy array (or None).
        """
        logger.info("Processing a single dataset split...")
        parent_texts = ds["parent_text"]
        
        parent_emb = self.get_embeddings(parent_texts)
        if self.embedding == "bert_sequential" and self.include_axes:
            axes_order = [
                "main_outcome_axis", "static_axis", "generic_axis", 
                "hypothetical_axis", "negation_axis", "intention_axis", 
                "opinion_axis", "recurrent_axis"
            ]
            axes_emb_list = []
            for sample in ds:
                axes_texts = [sample["axes"].get(key, "") for key in axes_order]
                if self.shuffle_axes:
                    random.shuffle(axes_texts)
                emb = self.get_embeddings(axes_texts)
                axes_emb_list.append(emb)
            axes_emb = torch.stack(axes_emb_list)
            combined = torch.cat([parent_emb.unsqueeze(1), axes_emb], dim=1)
            combined_emb = combined.view(combined.size(0), -1, combined.size(-1))
            logger.info(f"Combined sequential feature vector shape: {combined_emb.shape}")
        elif self.include_axes:
            axes_order = [
                "main_outcome_axis", "static_axis", "generic_axis", 
                "hypothetical_axis", "negation_axis", "intention_axis", 
                "opinion_axis", "recurrent_axis"
            ]
            axes_emb_list = []
            for sample in ds:
                axes_texts = [sample["axes"].get(key, "") for key in axes_order]
                if self.shuffle_axes:
                    random.shuffle(axes_texts)
                emb = self.get_embeddings(axes_texts)
                emb_flat = emb.flatten()
                axes_emb_list.append(emb_flat)
            axes_emb = torch.stack(axes_emb_list)
            parent_emb = parent_emb.flatten(1)
            combined_emb = torch.cat([parent_emb, axes_emb], dim=1)
            logger.info(f"Combined feature vector shape: {combined_emb.shape}")
        else:
            combined_emb = parent_emb
            logger.info(f"Feature vector shape without axes: {combined_emb.shape}")

        targets = None
        if "target_values" in ds.column_names:
            logger.info("Extracting and normalizing target values...")
            targets = [list(sample["target_values"].values()) for sample in ds]
            targets = np.array(targets)
            targets = self._normalize_targets(targets)
        else:
            logger.info("No target values found in dataset.")
        return combined_emb, targets

    def preprocess(self) -> Union[Tuple[torch.Tensor, Optional[np.ndarray]], dict]:
        """
        Processes the raw dataset(s) and returns feature vectors and targets.
        
        Behavior:
          - If self.split is a single split (string), returns a tuple (features, targets).
          - If self.split is None or a list, returns a dictionary mapping each split name to (features, targets).
          
        The features for each sample consist of the parent's embedding concatenated with (if include_axes is True)
        the flattened axes embeddings.
        """
        logger.info("Starting preprocessing of dataset(s)...")
        if isinstance(self.dataset, dict):
            output = {}
            for split_name, ds in self.dataset.items():
                logger.info(f"Processing split: {split_name} with {len(ds)} samples...")
                output[split_name] = self._process_single_dataset(ds)
            logger.info("Preprocessing of all splits completed.")
            return output
        else:
            logger.info("Processing single split dataset...")
            result = self._process_single_dataset(self.dataset)
            logger.info("Preprocessing of single split completed.")
            return result
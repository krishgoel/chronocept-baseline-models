"""
Encoder modules for different transformer models.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer,
    RobertaModel, RobertaTokenizer,
    DebertaV2Model, DebertaV2Tokenizer,
    DistilBertModel, DistilBertTokenizer
)
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class BaseEncoder(nn.Module):
    """Base encoder class."""
    
    def __init__(self, model_name: str, max_length: int = 512):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
    
    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode a list of texts."""
        raise NotImplementedError
    
    def encode_single_sequence(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode a single text sequence."""
        return self.encode_texts([text])


class SBERTEncoder(BaseEncoder):
    """Sentence-BERT encoder using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_length: int = 512):
        super().__init__(model_name, max_length)
        try:
            from sentence_transformers import SentenceTransformer
            self.sbert_model = SentenceTransformer(model_name)
            self.hidden_size = self.sbert_model.get_sentence_embedding_dimension()
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
    
    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode texts using Sentence-BERT."""
        embeddings = self.sbert_model.encode(texts, convert_to_tensor=True)
        return {
            'embeddings': embeddings,
            'attention_mask': torch.ones(embeddings.shape[0], 1)  # Dummy attention mask
        }


class RoBERTaEncoder(BaseEncoder):
    """RoBERTa encoder."""
    
    def __init__(self, model_name: str = "roberta-base", max_length: int = 512):
        super().__init__(model_name, max_length)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
    
    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode texts using RoBERTa."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move encoded tensors to model device
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
        
        # RobertaModel does not provide pooler_output; synthesize CLS pooling for compatibility
        last_hidden = outputs.last_hidden_state
        cls_pooled = last_hidden[:, 0, :]
        return {
            'last_hidden_states': last_hidden,
            'pooler_output': cls_pooled,
            'attention_mask': encoded['attention_mask']
        }


class DeBERTaEncoder(BaseEncoder):
    """DeBERTa-v3 encoder."""
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", max_length: int = 512):
        super().__init__(model_name, max_length)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        self.model = DebertaV2Model.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
    
    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode texts using DeBERTa-v3."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move encoded tensors to model device
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
        
        # DebertaV2Model returns BaseModelOutput without pooler_output; synthesize CLS pooling
        last_hidden = outputs.last_hidden_state
        cls_pooled = last_hidden[:, 0, :]
        return {
            'last_hidden_states': last_hidden,
            'pooler_output': cls_pooled,
            'attention_mask': encoded['attention_mask']
        }


class DistilBERTEncoder(BaseEncoder):
    """DistilBERT encoder."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 512):
        super().__init__(model_name, max_length)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
    
    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode texts using DistilBERT."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move encoded tensors to model device
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
        
        # DistilBertModel has no pooler_output; synthesize CLS pooling
        last_hidden = outputs.last_hidden_state
        cls_pooled = last_hidden[:, 0, :]
        return {
            'last_hidden_states': last_hidden,
            'pooler_output': cls_pooled,
            'attention_mask': encoded['attention_mask']
        }


class BERTEncoder(BaseEncoder):
    """BERT encoder (for legacy comparisons)."""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        super().__init__(model_name, max_length)
        from transformers import BertModel, BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
    
    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode texts using BERT."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move encoded tensors to model device
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
        
        return {
            'last_hidden_states': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output,
            'attention_mask': encoded['attention_mask']
        }


class AxisEncoder:
    """Utility class for encoding axes with different strategies."""
    
    @staticmethod
    def no_axes(parent_texts: List[str]) -> List[str]:
        """Return only parent texts."""
        return parent_texts
    
    @staticmethod
    def single_sequence_markers(parent_texts: List[str], axes_data: List[Dict[str, str]]) -> List[str]:
        """Create single sequences with axis markers."""
        axes_order = [
            "main_outcome_axis", "static_axis", "generic_axis", 
            "hypothetical_axis", "negation_axis", "intention_axis", 
            "opinion_axis", "recurrent_axis"
        ]
        
        combined_texts = []
        for parent, axes in zip(parent_texts, axes_data):
            combined = parent
            for i, axis_key in enumerate(axes_order):
                if axis_key in axes and axes[axis_key]:
                    combined += f" [AXIS_{i}] {axes[axis_key]}"
            combined_texts.append(combined)
        
        return combined_texts
    
    @staticmethod
    def sbert_concat(parent_texts: List[str], axes_data: List[Dict[str, str]], 
                    encoder: SBERTEncoder) -> torch.Tensor:
        """Concatenate SBERT embeddings per axis."""
        axes_order = [
            "main_outcome_axis", "static_axis", "generic_axis", 
            "hypothetical_axis", "negation_axis", "intention_axis", 
            "opinion_axis", "recurrent_axis"
        ]
        
        all_embeddings = []
        for parent, axes in zip(parent_texts, axes_data):
            # Parent embedding
            parent_emb = encoder.encode_texts([parent])['embeddings']
            
            # Axes embeddings
            axes_texts = [axes.get(key, "") for key in axes_order]
            axes_emb = encoder.encode_texts(axes_texts)['embeddings']
            
            # Concatenate
            combined = torch.cat([parent_emb, axes_emb.flatten()], dim=1)
            all_embeddings.append(combined)
        
        return torch.stack(all_embeddings)

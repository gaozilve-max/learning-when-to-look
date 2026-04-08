# Copyright 2025 Zilve Gao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, List, Optional
import numpy as np
from transformers import PreTrainedTokenizer


def compute_entropy_from_single_token(logprob_dict: Dict[int, Any]) -> float:
    """
    Compute entropy for a single token position given its logprob distribution.
    
    Args:
        logprob_dict: Dictionary mapping token_id to LogProb object
                      Each LogProb object has .logprob attribute
    
    Returns:
        float: Entropy value for this token position
    """
    if not logprob_dict:
        return 0.0
    
    log_probs = np.array([logprob.logprob for logprob in logprob_dict.values()], dtype=np.float32)
    
    # Convert log probabilities to probabilities
    raw_probs = np.exp(log_probs)
    sum_raw = float(np.sum(raw_probs))
    
    # Handle tail mass (probability not captured in top-k)
    tail = max(0.0, 1.0 - sum_raw)
    
    # Calculate entropy: H = -sum(p * log(p))
    entropy = -float(np.sum(raw_probs * log_probs))
    
    # Add tail contribution to entropy
    if tail > 0.0:
        entropy -= float(tail * np.log(max(tail, 1e-12)))
    
    return entropy


def identify_sentence_boundaries(
    token_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    sentence_end_tokens: Optional[List[str]] = None
) -> List[int]:
    """
    Identify the positions (indices) where sentences end.
    
    Args:
        token_ids: List of token IDs in the response
        tokenizer: Tokenizer used to decode tokens
        sentence_end_tokens: List of sentence-ending tokens to look for
                            If None, defaults to common sentence endings
    
    Returns:
        List[int]: List of indices where sentences end (indices of the last token in each sentence)
    """
    if sentence_end_tokens is None:
        # Default sentence endings for Chinese and English
        sentence_end_tokens = ['。', '！', '？', '?', '!', '.', '\n', '<eos>']
    
    sentence_boundaries = []
    
    for idx, token_id in enumerate(token_ids):
        try:
            token_str = tokenizer.decode([token_id])
            # Normalize token string for comparison
            token_normalized = token_str.strip()
            
            if token_normalized in sentence_end_tokens:
                sentence_boundaries.append(idx)
        except Exception:
            # Skip if token cannot be decoded
            continue
    
    # Always add the last token as a sentence boundary if it's not already
    if token_ids and (not sentence_boundaries or sentence_boundaries[-1] != len(token_ids) - 1):
        sentence_boundaries.append(len(token_ids) - 1)
    
    return sentence_boundaries


def compute_sentence_level_entropies(
    token_ids: List[int],
    logprobs_per_token: List[Dict[int, Any]],
    tokenizer: PreTrainedTokenizer,
    min_sentence_length: int = 1
) -> Dict[str, Any]:
    """
    Compute entropy for each sentence in the response.
    Uses mean aggregation to ensure length-invariance: shorter and longer sentences
    are directly comparable as entropy is normalized by sentence length.
    
    Args:
        token_ids: List of token IDs in the response
        logprobs_per_token: List of logprob dicts, one per token position
                           Each dict maps token_id -> LogProb object
        tokenizer: Tokenizer for decoding
        min_sentence_length: Minimum number of tokens to form a valid sentence
    
    Returns:
        Dict containing:
            - "sentence_entropies": List[float] - mean entropy of each sentence
            - "sentence_ranges": List[Tuple[int, int]] - (start_idx, end_idx) for each sentence
            - "sentence_texts": List[str] - decoded text of each sentence
            - "overall_entropy": float - mean entropy across all tokens
    """
    if not token_ids or not logprobs_per_token:
        return {
            "sentence_entropies": [],
            "sentence_ranges": [],
            "sentence_texts": [],
            "overall_entropy": 0.0,
        }
    
    # Ensure lengths match
    if len(token_ids) != len(logprobs_per_token):
        raise ValueError(f"token_ids length ({len(token_ids)}) != logprobs_per_token length ({len(logprobs_per_token)})")
    
    # Compute entropy for each token position
    token_entropies = [compute_entropy_from_single_token(logprob_dict) for logprob_dict in logprobs_per_token]
    
    # Find sentence boundaries
    sentence_boundaries = identify_sentence_boundaries(token_ids, tokenizer)
    
    # If no sentence boundaries found, treat the whole response as one sentence
    if not sentence_boundaries:
        sentence_boundaries = [len(token_ids) - 1]
    
    sentence_entropies = []
    sentence_ranges = []
    sentence_texts = []
    
    start_idx = 0
    for end_idx in sentence_boundaries:
        sentence_length = end_idx - start_idx + 1
        
        # Skip very short sentences
        if sentence_length < min_sentence_length:
            start_idx = end_idx + 1
            continue
        
        # Extract entropies for this sentence
        sentence_token_entropies = token_entropies[start_idx : end_idx + 1]
        
        # Aggregate using mean (length-invariant)
        sentence_entropy = float(np.mean(sentence_token_entropies)) if sentence_token_entropies else 0.0
        
        sentence_entropies.append(sentence_entropy)
        sentence_ranges.append((start_idx, end_idx))
        
        # Decode sentence text
        try:
            sentence_text = tokenizer.decode(token_ids[start_idx : end_idx + 1])
        except Exception:
            sentence_text = ""
        
        sentence_texts.append(sentence_text)
        start_idx = end_idx + 1
    
    # Compute overall entropy (mean across all tokens)
    overall_entropy = float(np.mean(token_entropies)) if token_entropies else 0.0
    
    return {
        "sentence_entropies": sentence_entropies,
        "sentence_ranges": sentence_ranges,
        "sentence_texts": sentence_texts,
        "overall_entropy": overall_entropy,
        "token_entropies": token_entropies,
    }


def compute_batch_sentence_entropies(
    batch_token_ids: List[List[int]],
    batch_logprobs: List[List[Dict[int, Any]]],
    tokenizer: PreTrainedTokenizer,
    min_sentence_length: int = 1
) -> List[Dict[str, Any]]:
    """
    Compute sentence-level entropies for a batch of responses.
    Uses mean aggregation to ensure sentences are compared fairly regardless of length.
    
    Args:
        batch_token_ids: List of token ID sequences, one per sample
        batch_logprobs: List of logprob sequences, one per sample
        tokenizer: Tokenizer for decoding
        min_sentence_length: Minimum tokens for a valid sentence
    
    Returns:
        List of entropy information dicts, one per sample
    """
    results = []
    for token_ids, logprobs_per_token in zip(batch_token_ids, batch_logprobs):
        result = compute_sentence_level_entropies(
            token_ids=token_ids,
            logprobs_per_token=logprobs_per_token,
            tokenizer=tokenizer,
            min_sentence_length=min_sentence_length,
        )
        results.append(result)
    
    return results

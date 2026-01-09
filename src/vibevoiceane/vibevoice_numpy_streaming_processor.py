import os
import json
import logging
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
)
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class VibeVoiceNumpyStreamingProcessor:
    """
    Numpy-only VibeVoice Streaming Processor.
    This class mimics VibeVoiceStreamingProcessor but removes all PyTorch dependencies and
    unused audio processing logic for the inference demo.
    """

    def __init__(self, tokenizer=None, **kwargs):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Instantiate a VibeVoiceNumpyStreamingProcessor from a pretrained model.
        """
        from transformers.utils import cached_file

        # Try to load from local path first, then from HF hub
        config_path = os.path.join(
            pretrained_model_name_or_path, "preprocessor_config.json"
        )
        config = None

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            try:
                config_file = cached_file(
                    pretrained_model_name_or_path, "preprocessor_config.json", **kwargs
                )
                with open(config_file, "r") as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load preprocessor_config.json: {e}")
                logger.warning("Using default configuration")
                config = {}

        # Load tokenizer
        language_model_pretrained_name = config.get(
            "language_model_pretrained_name", None
        ) or kwargs.pop("language_model_pretrained_name", "Qwen/Qwen2.5-1.5B")
        logger.info(f"Loading tokenizer from {language_model_pretrained_name}")

        if "qwen" in language_model_pretrained_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                language_model_pretrained_name, **kwargs
            )
        else:
            # Fallback or allow user to pass tokenizer
            logger.warning(
                f"Tokenizer type {language_model_pretrained_name} might not be supported blindly. Attempting load."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                language_model_pretrained_name, **kwargs
            )

        return cls(tokenizer=tokenizer)

    def process_input_with_cached_prompt(
        self,
        text: Optional[str] = None,
        cached_prompt: Optional[Dict[str, Any]] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[
            str
        ] = None,  # kept for sig compatibility, but we basically ignore or expect None/'np'
        return_attention_mask: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to process one text script based on cached prompt.
        Only supports single examples and text-only processing (no speech inputs).
        """
        # Only support single example
        texts = [text]
        cached_prompts = [cached_prompt]

        # Process each input
        all_encodings = []
        for text_input, cached_prompt_input in zip(texts, cached_prompts):
            script_tokens = self.tokenizer.encode(
                text_input.strip() + "\n", add_special_tokens=False
            )

            # Determine input lengths from cached prompts (numpy dicts)
            def get_length(model_output, key_name):
                # We expect numpy arrays or scalars in the cached prompt dict
                # cached_prompt_input['lm'] -> {'past_key_values': ...} or similar structure

                # Check directly if 'lm' / 'tts_lm' are dicts with 'seen_tokens' or similar
                # The usage in demo suggests: cached_prompt_input['lm'] is the state.

                # Based on previous analysis of the demo script, the structure is:
                # result['lm']['past_key_values']...
                # But here we pass `cached_prompt_input['lm']`

                # The logic in original processor was:
                # Try last_hidden_state (for shape)
                # Try past_key_values (for seen_tokens)

                state = model_output

                # Try to find seen_tokens in past_key_values if present
                if isinstance(state, dict):
                    if "past_key_values" in state:
                        pkv = state["past_key_values"]
                        if isinstance(pkv, dict) and "seen_tokens" in pkv:
                            val = pkv["seen_tokens"]
                            return int(val.item()) if hasattr(val, "item") else int(val)

                    # Direct check if state itself has seen_tokens
                    if "seen_tokens" in state:
                        val = state["seen_tokens"]
                        return int(val.item()) if hasattr(val, "item") else int(val)

                    # Fallback: check headers/shapes if possible?
                    # For now, let's look at the original code again.
                    # It checked `last_hidden_state` shape OR `past_key_values` seen_tokens.
                    if "last_hidden_state" in state:
                        lhs = state["last_hidden_state"]
                        if hasattr(lhs, "shape"):
                            return lhs.shape[1]

                # If we are here, we might fail. But let's assume valid cache for now or 0
                return 0

            input_id_length = get_length(cached_prompt_input["lm"], "lm")
            tts_lm_input_id_length = get_length(cached_prompt_input["tts_lm"], "tts_lm")

            # pseudo input ids and masks
            input_ids = [self.tokenizer.pad_token_id] * input_id_length
            tts_lm_input_ids = [self.tokenizer.pad_token_id] * tts_lm_input_id_length
            speech_input_mask = [False] * tts_lm_input_id_length

            encoding = {
                "input_ids": input_ids,
                "tts_lm_input_ids": tts_lm_input_ids,
                "tts_text_ids": script_tokens,
                "speech_inputs": None,
                "speech_input_mask": speech_input_mask,
            }
            all_encodings.append(encoding)

        # Combine batch
        batch_encoding = self._batch_encode(
            all_encodings,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
        )

        return batch_encoding

    def _batch_encode(
        self,
        encodings: List[Dict[str, Any]],
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
    ) -> BatchEncoding:
        """Combine multiple encodings into a batch with padding."""

        # In this simplified version, we assume single batch item usually,
        # but we iterate to be safe.
        input_ids_list = [enc["input_ids"] for enc in encodings]
        tts_lm_input_ids_list = [enc["tts_lm_input_ids"] for enc in encodings]
        tts_text_ids_list = [enc["tts_text_ids"] for enc in encodings]
        speech_input_masks_list = [enc["speech_input_mask"] for enc in encodings]

        attention_masks = (
            [[1] * len(ids) for ids in input_ids_list]
            if return_attention_mask
            else None
        )
        tts_lm_attention_masks = (
            [[1] * len(ids) for ids in tts_lm_input_ids_list]
            if return_attention_mask
            else None
        )

        batch_encoding = BatchEncoding()

        # Return simple lists or numpy arrays. The demo script converts them to numpy anyway.
        # So we can just return lists and let the caller handle it, OR return numpy if requested.
        # But BatchEncoding usually holds lists if return_tensors is None.

        batch_encoding["input_ids"] = input_ids_list
        batch_encoding["tts_lm_input_ids"] = tts_lm_input_ids_list
        batch_encoding["tts_text_ids"] = tts_text_ids_list

        if return_attention_mask:
            batch_encoding["attention_mask"] = attention_masks
            batch_encoding["tts_lm_attention_mask"] = tts_lm_attention_masks

        batch_encoding["speech_input_mask"] = speech_input_masks_list
        batch_encoding["speech_tensors"] = None
        batch_encoding["speech_masks"] = None

        return batch_encoding

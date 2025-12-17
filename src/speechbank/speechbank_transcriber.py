"""
Speechbank Transcriber Script
Handles transcription of audio files using various models (CrisperWhisper, OpenAI Whisper, etc.).
Updated with anti-hallucination measures including condition_on_previous_text=False and sequential processing for large files.
Updated to use new filename structure with _transcript- prefix.
FIXED: Robust timestamp handling to prevent NoneType errors and repetitive hallucinations.
"""
import argparse
import concurrent.futures
import gc
import json
import librosa
import logging
import numpy as np
import os
import pandas as pd
import re
import soundfile as sf
import tempfile
import threading
import time
import torch
import torchaudio
import traceback
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from num2words import num2words
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any

# Model imports
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import whisper

from speechbank_utils import (
    CONFIG, setup_logging, ensure_directory_exists,
    get_base_filename, get_device, select_best_gpu, 
    clean_gpu_memory, setup_cuda_memory, log_gpu_memory_state
)
from speechbank_assembler import run_assembler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.whisper.generation_whisper")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines.automatic_speech_recognition")
warnings.filterwarnings("ignore", message=".*inputs.*deprecated.*input_features.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*will be ignored.*")
warnings.filterwarnings("ignore", message=".*Whisper did not predict an ending timestamp.*")
warnings.filterwarnings("ignore", message=".*You have passed task.*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*tokens have been added.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")
warnings.filterwarnings("ignore", message=".*Whisper did not predict an ending timestamp, which can happen if audio is cut off in the middle of a word.*")

# Additional suppressions for CrisperWhisper-specific messages
warnings.filterwarnings("ignore", message=".*Special tokens have been added in the vocabulary.*")
warnings.filterwarnings("ignore", message=".*The attention mask is not set and cannot be inferred from input because pad token is same as eos token.*")
warnings.filterwarnings("ignore", message=".*Whisper did not predict an ending timestamp.*WhisperTimeStampLogitsProcessor.*")

# Alternative approach - suppress all UserWarning and FutureWarning from transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Suppress tokenizer warnings specifically
warnings.filterwarnings("ignore", message=".*tokenizer.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.tokenization_utils_base")

# Set transformers logging level to ERROR to suppress INFO/WARNING messages
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.pipelines").setLevel(logging.ERROR)

def safe_clean_gpu_memory():
    """Safely clean GPU memory with error handling and more thorough cleanup."""
    try:
        if torch.cuda.is_available():
            # Get current device
            current_device = torch.cuda.current_device()
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Force synchronization to ensure operations complete
            torch.cuda.synchronize()
            
            # Try to free up any unused cached memory
            try:
                torch.cuda.reset_peak_memory_stats()
            except:
                pass  # This function may not be available in all PyTorch versions
            
            # Additional cleanup attempt
            torch.cuda.empty_cache()
            
        # Force Python garbage collection
        gc.collect()
        
        # Small delay to allow cleanup to complete
        time.sleep(0.1)
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Error during GPU cleanup: {str(e)}")


def force_cleanup_device_transcriber(self, device: str):
    """Force cleanup and removal of a specific device's transcriber."""
    if device in self.transcriber_cache:
        try:
            self.logger.info(f"Force cleaning up transcriber for device {device}")
            self.transcriber_cache[device].cleanup()
        except Exception as e:
            self.logger.warning(f"Error during force cleanup for {device}: {str(e)}")
        finally:
            del self.transcriber_cache[device]
            
        # Additional GPU cleanup if CUDA device
        if "cuda" in device:
            try:
                torch.cuda.set_device(device)
                safe_clean_gpu_memory()
                # Force synchronization
                torch.cuda.synchronize()
            except Exception as e:
                self.logger.warning(f"Error during GPU cleanup for {device}: {str(e)}")


class BaseTranscriber(ABC):
    """Abstract base class for all transcriber implementations."""
    
    def __init__(self, device=None, logger=None):
        """Initialize base transcriber.
        
        Args:
            device: Device to run inference on
            logger: Logger instance to use
        """
        self.device = get_device(device)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with transcription results
        """
        pass
    
    def clean_transcript_text(self, text: str) -> str:
        """Clean the transcript text by removing special characters and fixing spacing.
        
        Args:
            text: The original transcript text
            
        Returns:
            Cleaned transcript text
        """
        if not text:
            return text
            
        # Temporarily protect dysfluencies in square brackets like [UH] and [UM]
        dysfluency_patterns = re.finditer(r'\[(UH|UM|AH|ER|HM|MM|EH)\]', text, re.IGNORECASE)
        placeholders = {}
        
        for i, match in enumerate(dysfluency_patterns):
            placeholder = f"__DYSFLUENCY_{i}__"
            placeholders[placeholder] = match.group(0)
            text = text.replace(match.group(0), placeholder)
        
        # Convert numbers to words using num2words
        text = self._convert_numbers_to_words(text)
        
        # Keep only alphanumeric characters, spaces, periods, commas, apostrophes, and hyphens
        # text = re.sub(r'[^\w\s.,\'-]', ' ', text)
        
        # Remove extra whitespace (including double spaces)
        text = re.sub(r'\s+', ' ', text)
        
        # Restore dysfluencies
        for placeholder, original in placeholders.items():
            text = text.replace(placeholder, original)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _convert_numbers_to_words(self, text: str) -> str:
        """Convert numbers in text to their word equivalents.
        
        Args:
            text: Input text with potential numbers
            
        Returns:
            Text with numbers converted to words
        """        
        def replace_number(match):
            try:
                number = int(match.group())
                # Convert to words and capitalize first letter
                word_form = num2words(number).title()
                return word_form
            except (ValueError, OverflowError):
                # If conversion fails, return original number
                return match.group()
        
        # Match standalone numbers (not part of words)
        # This pattern matches numbers that are surrounded by word boundaries
        number_pattern = r'\b\d+\b'
        result = re.sub(number_pattern, replace_number, text)
        
        return result
    
    def _remove_duplicate_chunks(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate chunks with identical text and timestamps.
        
        Args:
            result: Transcription result with chunks
            
        Returns:
            Result with duplicate chunks removed
        """
        if "chunks" not in result or not result["chunks"]:
            return result
        
        # Track seen combinations of text and timestamp
        seen_combinations = set()
        unique_chunks = []
        
        for chunk in result["chunks"]:
            # Create a hashable key from text and timestamp
            text = chunk.get("text", "")
            timestamp = chunk.get("timestamp", [])
            
            # Convert timestamp to tuple for hashing
            if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                timestamp_key = (timestamp[0], timestamp[1])
            else:
                timestamp_key = tuple(timestamp) if timestamp else ()
            
            combination_key = (text, timestamp_key)
            
            # Only add if we haven't seen this exact combination before
            if combination_key not in seen_combinations:
                seen_combinations.add(combination_key)
                unique_chunks.append(chunk)
            else:
                self.logger.debug(f"Removing duplicate chunk: text='{text}', timestamp={timestamp_key}")
        
        # Update result with deduplicated chunks
        result["chunks"] = unique_chunks
        
        if len(unique_chunks) != len(result.get("chunks", [])):
            removed_count = len(result.get("chunks", [])) - len(unique_chunks)
            self.logger.info(f"Removed {removed_count} duplicate chunks")
        
        return result
    
    def _detect_repetitive_patterns(self, text: str) -> bool:
        """Detect if text contains repetitive patterns that indicate hallucination.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if repetitive patterns detected
        """
        if not text or len(text.strip()) == 0:
            return False
        
        # Check for repeated tokens
        tokens = text.split()
        if len(tokens) < 10:  # Too short to reliably detect patterns
            return False
        
        # Check for excessive repetition of any single token
        from collections import Counter
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        for token, count in token_counts.items():
            # If any token appears more than 30% of the time and more than 10 times
            if count > 10 and (count / total_tokens) > 0.3:
                self.logger.warning(f"Detected excessive repetition of token '{token}': {count}/{total_tokens} ({count/total_tokens:.1%})")
                return True
        
        # Check for repeating sequences
        text_clean = text.strip()
        
        # Look for patterns like "[UM] [UM] [UM]..." or "the the the..."
        pattern_checks = [
            r'(\[UH\]\s*){5,}',  # [UH] repeated 5+ times
            r'(\[UM\]\s*){5,}',  # [UM] repeated 5+ times
            r'(\[AH\]\s*){5,}',  # [AH] repeated 5+ times
            r'(\b\w+\s+)\1{4,}', # Any word repeated 5+ times
        ]
        
        for pattern in pattern_checks:
            if re.search(pattern, text_clean, re.IGNORECASE):
                self.logger.warning(f"Detected repetitive pattern matching: {pattern}")
                return True
        
        return False
    
    def _is_transcript_empty(self, result: Dict[str, Any]) -> bool:
        """Check if transcript result is empty or effectively blank.
        
        Args:
            result: Transcription result dictionary
            
        Returns:
            True if transcript is empty/blank
        """
        text = result.get('text', '').strip()
        
        # Check if completely empty
        if not text or len(text) == 0:
            self.logger.warning("Transcript is completely empty")
            return True
        
        # Log the actual text for debugging
        self.logger.debug(f"Transcript text (first 200 chars): {text[:200]}")
        
        # Check if it's just repetitive hallucination
        if self._detect_repetitive_patterns(text):
            self.logger.warning("Transcript detected as repetitive hallucination, treating as empty")
            self.logger.debug(f"Repetitive text sample: {text[:100]}")
            return True
        
        # Log successful transcript detection
        self.logger.debug(f"Transcript appears valid, length: {len(text)} characters")
        return False
    
    def save_transcript(self, result: Dict[str, Any], output_path_base: str) -> Tuple[str, bool]:
        """Save the transcription result to both text and JSON files.
        
        Args:
            result: The transcription result from the model
            output_path_base: Base path without extension where the transcripts should be saved
            
        Returns:
            tuple: (txt_path, success)
        """
        # Check if transcript is empty before processing
        if self._is_transcript_empty(result):
            self.logger.warning("Transcript is empty or repetitive, not saving files")
            return None, False
        
        # Remove duplicate chunks
        result = self._remove_duplicate_chunks(result)
        
        # Define file paths
        txt_path = f"{output_path_base}.txt"
        json_path = f"{output_path_base}.json"
        
        try:
            # Clean the transcript text
            cleaned_text = self.clean_transcript_text(result.get('text', ''))
            
            # Check again after cleaning
            if not cleaned_text.strip():
                self.logger.warning("Transcript is empty after cleaning, not saving files")
                return None, False
            
            # Final check for repetitive patterns after cleaning
            if self._detect_repetitive_patterns(cleaned_text):
                self.logger.warning("Cleaned transcript still contains repetitive patterns, not saving files")
                return None, False
            
            # Create a copy of the result for JSON
            result_copy = result.copy()
            result_copy['text'] = cleaned_text
            
            # Apply cleaning to each chunk's text if applicable
            if 'chunks' in result_copy:
                for chunk in result_copy['chunks']:
                    if 'text' in chunk:
                        chunk['text'] = self.clean_transcript_text(chunk['text'])
            
            # Ensure directory exists
            ensure_directory_exists(txt_path)
            
            # Save text file with just the cleaned transcript text
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
                
            # Save JSON file with the full result including timestamps
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_copy, f, indent=2)
                    
            self.logger.info(f"Transcript saved to {os.path.basename(txt_path)}")
            self.logger.info(f"Metadata saved to {os.path.basename(json_path)}")
            return txt_path, True
            
        except Exception as e:
            self.logger.error(f"Failed to save transcripts: {str(e)}")
            return None, False
    
    def cleanup(self):
        """Clean up model resources."""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'pipe'):
                del self.pipe
            if hasattr(self, 'processor'):
                del self.processor
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {str(e)}")
        finally:
            safe_clean_gpu_memory()


class CrisperWhisperTranscriber(BaseTranscriber):
    """Handles transcription using the CrisperWhisper model."""
    
    def __init__(self, model_name=None, device=None, chunk_length_s=None, batch_size=None):
        """Initialize the CrisperWhisper transcriber.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to run inference on
            chunk_length_s: Length of audio chunks in seconds
            batch_size: Batch size for processing
        """
        super().__init__(device)
        
        # Load CrisperWhisper-specific settings from config
        crisper_config = CONFIG.get('transcribers', {}).get('crisper_whisper', {})
        common_config = CONFIG.get('transcribers', {}).get('common', {})
        
        self.model_name = model_name or crisper_config.get('model_name', 'nyrahealth/CrisperWhisper')
        self.chunk_length_s = chunk_length_s or crisper_config.get('chunk_length_s', 30)
        self.batch_size = batch_size or crisper_config.get('batch_size', 1)
        self.pause_split_threshold = crisper_config.get('pause_split_threshold', 0.12)
        self.language = common_config.get('language', 'en')
        self.task = common_config.get('task', 'transcribe')
        
        # Model loading parameters
        self.attn_implementation = crisper_config.get('attn_implementation', 'eager')
        self.low_cpu_mem_usage = crisper_config.get('low_cpu_mem_usage', True)
        self.use_safetensors = crisper_config.get('use_safetensors', True)
        
        # Anti-hallucination parameters
        self.condition_on_previous_text = False  # Force to False for anti-hallucination
        
        self.logger.info(f"Initializing CrisperWhisper with model {self.model_name} on {self.device}")
        self.logger.info(f"Using chunk_length_s={self.chunk_length_s}, batch_size={self.batch_size}")
        self.logger.info(f"Anti-hallucination: condition_on_previous_text={self.condition_on_previous_text}")
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the CrisperWhisper model and processor with simplified approach."""
        try:
            # Clean memory first
            safe_clean_gpu_memory()
            
            # Use consistent dtypes
            torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
            
            self.logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            self.logger.info("Loading model...")
            # Load model with minimal settings to avoid conflicts
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                use_safetensors=self.use_safetensors,
                attn_implementation=self.attn_implementation,
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Create pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                chunk_length_s=self.chunk_length_s,
                batch_size=self.batch_size,
                return_timestamps="word",
                torch_dtype=torch_dtype,
                device=self.device
            )
            
            self.logger.info("CrisperWhisper model loaded successfully")
            
            # Log memory usage after loading
            if "cuda" in self.device:
                log_gpu_memory_state(self.logger)
                
        except Exception as e:
            self.logger.error(f"Failed to initialize CrisperWhisper model: {str(e)}")
            safe_clean_gpu_memory()
            raise
    
    def _should_use_sequential_processing(self, audio_path: str) -> bool:
        """Determine if sequential processing should be used instead of chunking."""
        try:
            # Check file size and duration
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            duration = librosa.get_duration(path=audio_path)
            
            # Use sequential for large files (>10 minutes OR >200MB)
            large_file_threshold_mb = CONFIG.get('transcribers', {}).get('common', {}).get('large_file_threshold_mb', 200)
            duration_threshold = 600  # 10 minutes
            
            return duration > duration_threshold or file_size_mb > large_file_threshold_mb
            
        except Exception as e:
            self.logger.warning(f"Could not determine if file needs sequential processing: {str(e)}")
            return False
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe the audio file using CrisperWhisper with anti-hallucination measures.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with transcription results
        """
        self.logger.info(f"Transcribing with CrisperWhisper: {os.path.basename(audio_path)}")
        
        # Clean memory before transcription
        if "cuda" in self.device:
            safe_clean_gpu_memory()
        
        try:
            # Get audio duration for logging
            duration = librosa.get_duration(path=audio_path)
            self.logger.info(f"Audio duration: {duration:.2f} seconds")
            
            # Check if we should use sequential processing for large files
            if self._should_use_sequential_processing(audio_path):
                return self._transcribe_sequential(audio_path)
            else:
                # For files under 10 minutes, try different approaches based on duration
                return self._transcribe_smart(audio_path, duration)
                
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise
    
    def _transcribe_smart(self, audio_path: str, duration: float) -> Dict[str, Any]:
        """Smart transcription that chooses the best approach based on duration."""
        
        # For very short files (under 1 minute), use no chunking
        if duration <= 60:
            self.logger.info("Using single-pass transcription for short file")
            return self._transcribe_single_pass(audio_path)
        
        # For files that are only slightly longer than chunk size, try single pass first
        elif duration <= self.chunk_length_s + 15:  # Allow 15s buffer
            self.logger.info("Attempting single-pass for medium-short file")
            try:
                result = self._transcribe_single_pass(audio_path)
                # Verify coverage and check for hallucinations
                if self._verify_coverage(result, duration) and not self._detect_repetitive_patterns(result.get('text', '')):
                    return result
                else:
                    self.logger.warning("Single-pass had poor coverage or repetitive patterns, falling back to manual chunking")
                    return self._transcribe_manual_chunks(audio_path, duration)
            except Exception as e:
                self.logger.warning(f"Single-pass failed: {str(e)}, falling back to manual chunking")
                return self._transcribe_manual_chunks(audio_path, duration)
        
        # For longer files, use manual chunking approach
        else:
            self.logger.info("Using manual chunking for longer file")
            return self._transcribe_manual_chunks(audio_path, duration)
    
    def _transcribe_single_pass(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe entire file in a single pass without chunking."""
        
        result = self.pipe(
            audio_path,
            return_timestamps="word",
            chunk_length_s=None,  # Disable chunking entirely
            generate_kwargs={
                "task": self.task,
                "language": self.language,
                "max_new_tokens": 440,
                "num_beams": 1,
                "do_sample": False,
                "forced_decoder_ids": None,
                "repetition_penalty": 1.2,  # Add repetition penalty
                "no_repeat_ngram_size": 3,   # Prevent 3-gram repetitions
            }
        )
        
        # Apply pause adjustment with error handling
        result = self._adjust_pauses(result)
        
        self.logger.info("Single-pass transcription completed successfully")
        return result
    
    def _safe_timestamp_conversion(self, timestamp_value: Any, fallback_time: float = 0.0) -> float:
        """Safely convert timestamp value to float, handling None and invalid values.
        
        Args:
            timestamp_value: The timestamp value to convert
            fallback_time: Fallback time if conversion fails
            
        Returns:
            Float timestamp value
        """
        if timestamp_value is None:
            return fallback_time
        
        try:
            return float(timestamp_value)
        except (ValueError, TypeError):
            self.logger.debug(f"Invalid timestamp value: {timestamp_value}, using fallback: {fallback_time}")
            return fallback_time
    
    def _transcribe_manual_chunks(self, audio_path: str, duration: float) -> Dict[str, Any]:
        """Manual chunking approach that ensures complete coverage with robust timestamp handling."""
        self.logger.info("Using manual chunking approach with robust timestamp handling")
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Use smaller chunks for better accuracy, no overlap to avoid confusion
            chunk_duration = min(self.chunk_length_s - 5, 25)  # 5s buffer, max 25s
            
            # Calculate chunks with no overlap
            chunks_info = []
            start_time = 0
            
            while start_time < duration:
                end_time = min(start_time + chunk_duration, duration)
                chunks_info.append((start_time, end_time))
                start_time = end_time  # No overlap
            
            self.logger.info(f"Processing {len(chunks_info)} non-overlapping chunks of ~{chunk_duration}s each")
            
            all_chunks = []
            all_text_parts = []
            failed_chunks = 0
            
            for i, (start_time, end_time) in enumerate(chunks_info):
                self.logger.debug(f"Processing chunk {i+1}/{len(chunks_info)}: {start_time:.2f}s-{end_time:.2f}s")
                
                # Extract chunk
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                chunk_audio = y[start_sample:end_sample]
                
                # Skip very short chunks (less than 0.5 seconds)
                if len(chunk_audio) < sr * 0.5:
                    self.logger.debug(f"Skipping very short chunk {i+1}")
                    continue
                
                # Save chunk to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    sf.write(temp_path, chunk_audio, sr)
                
                try:
                    # Transcribe chunk without chunking (treat as single piece)
                    chunk_result = self.pipe(
                        temp_path,
                        return_timestamps="word",
                        chunk_length_s=None,  # No internal chunking
                        generate_kwargs={
                            "task": self.task,
                            "language": self.language,
                            "max_new_tokens": 440,
                            "num_beams": 1,
                            "do_sample": False,
                            "forced_decoder_ids": None,
                            "repetition_penalty": 1.2,  # Add repetition penalty
                            "no_repeat_ngram_size": 3,   # Prevent 3-gram repetitions
                            "temperature": 0.0,  # Deterministic output
                        }
                    )
                    
                    # Process the chunk result
                    chunk_text = chunk_result.get('text', '').strip()
                    
                    # Check for repetitive patterns in this chunk
                    if self._detect_repetitive_patterns(chunk_text):
                        self.logger.warning(f"Chunk {i+1} contains repetitive patterns, skipping")
                        failed_chunks += 1
                        continue
                    
                    if chunk_text:
                        # Ensure proper spacing for text joining
                        # Strip whitespace and ensure we have clean text for joining
                        clean_chunk_text = chunk_text.strip()
                        if clean_chunk_text:
                            all_text_parts.append(clean_chunk_text)
                    
                    # Adjust timestamps to global time and add to collection with robust handling
                    if 'chunks' in chunk_result and chunk_result['chunks']:
                        for word_chunk in chunk_result['chunks']:
                            if 'timestamp' in word_chunk and isinstance(word_chunk['timestamp'], (list, tuple)) and len(word_chunk['timestamp']) >= 2:
                                # Use robust timestamp conversion
                                chunk_start = self._safe_timestamp_conversion(word_chunk['timestamp'][0], 0.0)
                                chunk_end = self._safe_timestamp_conversion(word_chunk['timestamp'][1], chunk_duration)
                                
                                # Validate relative timestamps within chunk
                                if chunk_start < 0:
                                    chunk_start = 0.0
                                if chunk_end <= chunk_start:
                                    chunk_end = chunk_start + 0.1  # Minimum 0.1s duration
                                if chunk_end > chunk_duration:
                                    chunk_end = chunk_duration
                                
                                # Adjust to global time
                                global_start = chunk_start + start_time
                                global_end = chunk_end + start_time
                                
                                # Final validation of global timestamps
                                if global_start < 0:
                                    global_start = start_time
                                if global_end > duration:
                                    global_end = duration
                                if global_end <= global_start:
                                    global_end = min(global_start + 0.1, duration)
                                
                                # Get the word text and ensure proper spacing
                                word_text = word_chunk.get('text', '').strip()
                                
                                # Create new chunk with adjusted timestamp
                                adjusted_chunk = {
                                    'text': word_text,
                                    'timestamp': [global_start, global_end]
                                }
                                all_chunks.append(adjusted_chunk)
                            else:
                                # Handle chunks without proper timestamp data
                                self.logger.debug(f"Chunk {i+1}: word_chunk missing valid timestamp, creating fallback")
                                chunk_text_part = word_chunk.get('text', '').strip()
                                if chunk_text_part:
                                    # Create a fallback timestamp within the chunk's time range
                                    fallback_chunk = {
                                        'text': chunk_text_part,
                                        'timestamp': [start_time, end_time]
                                    }
                                    all_chunks.append(fallback_chunk)
                    else:
                        # If no word-level chunks, create one chunk for the entire segment
                        if chunk_text:
                            fallback_chunk = {
                                'text': chunk_text.strip(),
                                'timestamp': [start_time, end_time]
                            }
                            all_chunks.append(fallback_chunk)
                    
                except Exception as e:
                    self.logger.error(f"Error processing chunk {i+1}: {str(e)}")
                    failed_chunks += 1
                    # Continue with other chunks instead of failing completely
                    continue
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            # Check if too many chunks failed
            total_chunks = len(chunks_info)
            if failed_chunks > total_chunks * 0.5:  # More than 50% failed
                self.logger.error(f"Too many chunks failed ({failed_chunks}/{total_chunks}), transcript likely unreliable")
                return {"text": "", "chunks": []}
            
            # Combine results with proper spacing
            # Join text parts with spaces, ensuring no double spaces or missing spaces
            if all_text_parts:
                # Join with single spaces and clean up any spacing issues
                combined_text = ' '.join(all_text_parts)
                # Remove any double spaces that might have been introduced
                combined_text = re.sub(r'\s+', ' ', combined_text).strip()
            else:
                combined_text = ""
            
            # Final check for repetitive patterns in combined text
            if self._detect_repetitive_patterns(combined_text):
                self.logger.error("Combined text contains repetitive patterns, transcript may be hallucinated")
                # Return empty result to trigger retry or failure
                return {"text": "", "chunks": []}
            
            # Sort chunks by start time to ensure proper order
            all_chunks.sort(key=lambda x: x['timestamp'][0] if x.get('timestamp') and len(x['timestamp']) > 0 else 0)
            
            # Post-process chunks to ensure proper spacing in word-level text
            processed_chunks = []
            for chunk in all_chunks:
                chunk_text = chunk.get('text', '')
                if chunk_text:
                    # Ensure the chunk text is properly spaced
                    # Remove leading/trailing whitespace but preserve internal structure
                    clean_text = chunk_text.strip()
                    if clean_text:
                        processed_chunk = chunk.copy()
                        processed_chunk['text'] = clean_text
                        processed_chunks.append(processed_chunk)
            
            merged_result = {
                "text": combined_text,
                "chunks": processed_chunks
            }
            
            # Verify coverage
            if self._verify_coverage(merged_result, duration):
                self.logger.info(f"Manual chunking completed successfully with good coverage")
            else:
                self.logger.warning(f"Manual chunking completed but coverage may be incomplete")
            
            # Log chunk failure rate
            if failed_chunks > 0:
                self.logger.info(f"Manual chunking: {failed_chunks}/{total_chunks} chunks failed")
            
            return merged_result
            
        except Exception as e:
            self.logger.error(f"Manual chunking failed: {str(e)}")
            raise
    
    def _verify_coverage(self, result: Dict[str, Any], expected_duration: float) -> bool:
        """Verify that transcription covers the expected duration."""
        if 'chunks' not in result or not result['chunks']:
            return False
        
        # Find the last timestamp
        last_timestamp = 0
        for chunk in result['chunks']:
            if 'timestamp' in chunk and isinstance(chunk['timestamp'], (list, tuple)) and len(chunk['timestamp']) >= 2:
                end_time = self._safe_timestamp_conversion(chunk['timestamp'][1], 0)
                if end_time > last_timestamp:
                    last_timestamp = end_time
        
        coverage_ratio = last_timestamp / expected_duration if expected_duration > 0 else 0
        self.logger.info(f"Coverage: {last_timestamp:.2f}s / {expected_duration:.2f}s ({coverage_ratio:.1%})")
        
        # Consider coverage good if we get at least 95% of the audio
        return coverage_ratio >= 0.95
    
    def _transcribe_sequential(self, audio_path: str) -> Dict[str, Any]:
        """Process large files sequentially without chunking to avoid hallucinations."""
        self.logger.info("Using sequential processing for large file to prevent hallucinations")
        
        try:
            # For very large files, we need to process them in larger segments
            duration = librosa.get_duration(path=audio_path)
            
            if duration > 1800:  # > 30 minutes, process in 5-minute segments
                return self._process_in_large_segments(audio_path)
            else:
                # For files under 30 minutes, try to process as one piece with no chunking
                self.logger.info("Processing as single piece without chunking")
                
                result = self._transcribe_single_pass(audio_path)
                
                # Check for hallucinations in the result
                if self._detect_repetitive_patterns(result.get('text', '')):
                    self.logger.warning("Sequential processing produced repetitive patterns, trying segmented approach")
                    return self._process_in_large_segments(audio_path)
                
                self.logger.info("Sequential transcription completed successfully")
                return result
                    
        except Exception as e:
            self.logger.error(f"Sequential transcription failed: {str(e)}")
            raise
    
    def _process_in_large_segments(self, audio_path: str) -> Dict[str, Any]:
        """Process very large files in 5-minute segments with anti-hallucination measures."""
        self.logger.info("Processing very large file in 5-minute segments")
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Use 5-minute segments with no overlap for simplicity and accuracy
            segment_duration = 300  # 5 minutes
            
            num_segments = int(np.ceil(duration / segment_duration))
            
            self.logger.info(f"Segmenting {duration:.2f}s audio into {num_segments} segments of {segment_duration}s each")
            
            all_text_parts = []
            all_chunks = []
            failed_segments = 0
            
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min(start_time + segment_duration, duration)
                
                self.logger.info(f"Processing segment {i+1}/{num_segments}: {start_time:.2f}s-{end_time:.2f}s")
                
                # Extract segment
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment = y[start_sample:end_sample]
                
                # Save segment to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    sf.write(temp_path, segment, sr)
                
                try:
                    # Transcribe segment with anti-hallucination settings and no internal chunking
                    segment_result = self.pipe(
                        temp_path,
                        return_timestamps="word",
                        chunk_length_s=None,  # No chunking for segments
                        generate_kwargs={
                            "task": self.task,
                            "language": self.language,
                            "max_new_tokens": 440,
                            "num_beams": 1,
                            "do_sample": False,
                            "forced_decoder_ids": None,
                            "repetition_penalty": 1.2,
                            "no_repeat_ngram_size": 3,
                            "temperature": 0.0,
                        }
                    )
                    
                    # Process text
                    segment_text = segment_result.get('text', '').strip()
                    
                    # Check for repetitive patterns in this segment
                    if self._detect_repetitive_patterns(segment_text):
                        self.logger.warning(f"Segment {i+1} contains repetitive patterns, skipping")
                        failed_segments += 1
                        continue
                    
                    if segment_text:
                        all_text_parts.append(segment_text)
                    
                    # Adjust timestamps to global time with robust handling
                    if 'chunks' in segment_result:
                        for chunk in segment_result['chunks']:
                            if 'timestamp' in chunk and isinstance(chunk['timestamp'], (list, tuple)) and len(chunk['timestamp']) >= 2:
                                chunk_start = self._safe_timestamp_conversion(chunk['timestamp'][0], 0.0)
                                chunk_end = self._safe_timestamp_conversion(chunk['timestamp'][1], segment_duration)
                                
                                # Validate and adjust timestamps
                                if chunk_start < 0:
                                    chunk_start = 0.0
                                if chunk_end <= chunk_start:
                                    chunk_end = chunk_start + 0.1
                                if chunk_end > segment_duration:
                                    chunk_end = segment_duration
                                
                                adjusted_chunk = {
                                    'text': chunk.get('text', ''),
                                    'timestamp': [chunk_start + start_time, chunk_end + start_time]
                                }
                                all_chunks.append(adjusted_chunk)
                    
                except Exception as e:
                    self.logger.error(f"Error processing segment {i+1}: {str(e)}")
                    failed_segments += 1
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            # Check if too many segments failed
            if failed_segments > num_segments * 0.5:
                self.logger.error(f"Too many segments failed ({failed_segments}/{num_segments}), transcript likely unreliable")
                return {"text": "", "chunks": []}
            
            # Combine results
            combined_text = ' '.join(all_text_parts)
            
            # Final check for repetitive patterns
            if self._detect_repetitive_patterns(combined_text):
                self.logger.error("Combined segment text contains repetitive patterns")
                return {"text": "", "chunks": []}
            
            # Sort chunks by timestamp
            all_chunks.sort(key=lambda x: x['timestamp'][0] if x.get('timestamp') and len(x['timestamp']) > 0 else 0)
            
            return {
                "text": combined_text,
                "chunks": all_chunks
            }
            
        except Exception as e:
            self.logger.error(f"Large segment processing failed: {str(e)}")
            raise
    
    def _adjust_pauses(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust pause timings as recommended by CrisperWhisper documentation with robust error handling."""
        if "chunks" not in pipeline_output or not pipeline_output["chunks"]:
            self.logger.debug("No chunks found in pipeline output, skipping pause adjustment")
            return pipeline_output
            
        adjusted_chunks = pipeline_output["chunks"].copy()
        
        for i in range(len(adjusted_chunks) - 1):
            current_chunk = adjusted_chunks[i]
            next_chunk = adjusted_chunks[i + 1]
            
            # Check if both chunks have valid timestamp data
            if "timestamp" not in current_chunk or "timestamp" not in next_chunk:
                self.logger.debug(f"Missing timestamp in chunk {i} or {i+1}, skipping pause adjustment")
                continue
            
            current_timestamp = current_chunk["timestamp"]
            next_timestamp = next_chunk["timestamp"]
            
            # Validate timestamp format and values
            if (not isinstance(current_timestamp, (list, tuple)) or len(current_timestamp) < 2 or
                not isinstance(next_timestamp, (list, tuple)) or len(next_timestamp) < 2):
                self.logger.debug(f"Invalid timestamp format in chunk {i} or {i+1}, skipping pause adjustment")
                continue
            
            # Use robust timestamp conversion
            current_start = self._safe_timestamp_conversion(current_timestamp[0], 0.0)
            current_end = self._safe_timestamp_conversion(current_timestamp[1], 1.0)
            next_start = self._safe_timestamp_conversion(next_timestamp[0], current_end + 0.1)
            next_end = self._safe_timestamp_conversion(next_timestamp[1], next_start + 1.0)
            
            # Check for logical timestamp order
            if current_end > next_start:
                self.logger.debug(f"Overlapping timestamps in chunks {i} and {i+1}: "
                                f"current_end={current_end}, next_start={next_start}")
                continue
            
            # Calculate pause duration safely
            try:
                pause_duration = next_start - current_end
                
                if pause_duration > 0:
                    if pause_duration > self.pause_split_threshold:
                        distribute = self.pause_split_threshold / 2
                    else:
                        distribute = pause_duration / 2
                    
                    # Apply adjustments with validation
                    new_current_end = current_end + distribute
                    new_next_start = next_start - distribute
                    
                    # Ensure we don't create invalid timestamps
                    if new_current_end <= new_next_start and new_current_end >= current_start:
                        adjusted_chunks[i]["timestamp"] = [current_start, new_current_end]
                        adjusted_chunks[i + 1]["timestamp"] = [new_next_start, next_end]
                        self.logger.debug(f"Adjusted pause between chunks {i} and {i+1}: "
                                        f"distributed {distribute:.3f}s")
                    else:
                        self.logger.debug(f"Pause adjustment would create invalid timestamps for chunks {i} and {i+1}")
                        
            except Exception as e:
                self.logger.warning(f"Error adjusting pause between chunks {i} and {i+1}: {str(e)}")
                continue
        
        pipeline_output["chunks"] = adjusted_chunks
        return pipeline_output

   
class WhisperTranscriber(BaseTranscriber):
    """Handles transcription using OpenAI's Whisper model with anti-hallucination measures."""
    
    def __init__(self, model_size=None, device=None, download_root=None, in_memory=None):
        """Initialize the Whisper transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to run inference on
            download_root: Root directory to download models to
            in_memory: Whether to load model in memory
        """
        super().__init__(device)
        
        # Load Whisper-specific settings from config
        whisper_config = CONFIG.get('transcribers', {}).get('whisper', {})
        common_config = CONFIG.get('transcribers', {}).get('common', {})
        
        self.model_size = model_size or whisper_config.get('model_size', 'large-v3')
        self.download_root = download_root or whisper_config.get('download_root')
        self.in_memory = in_memory if in_memory is not None else whisper_config.get('in_memory', False)
        
        # Transcription parameters
        self.language = common_config.get('language', 'en')
        self.task = common_config.get('task', 'transcribe')
        self.beam_size = whisper_config.get('beam_size', 5)
        self.best_of = whisper_config.get('best_of', 5)
        self.patience = whisper_config.get('patience')
        self.length_penalty = whisper_config.get('length_penalty')
        self.temperature = whisper_config.get('temperature', 0)
        self.compression_ratio_threshold = whisper_config.get('compression_ratio_threshold', 2.4)
        self.logprob_threshold = whisper_config.get('logprob_threshold', -1.0)
        self.no_speech_threshold = whisper_config.get('no_speech_threshold', 0.6)
        self.condition_on_previous_text = False  # Force to False for anti-hallucination
        self.initial_prompt = whisper_config.get('initial_prompt')
        self.word_timestamps = whisper_config.get('word_timestamps', True)
        
        self.logger.info(f"Initializing Whisper {self.model_size} model on {self.device}")
        self.logger.info(f"Anti-hallucination: condition_on_previous_text={self.condition_on_previous_text}")
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            self.model = whisper.load_model(
                self.model_size, 
                device=self.device,
                download_root=self.download_root,
                in_memory=self.in_memory
            )
            self.logger.info(f"Whisper {self.model_size} model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {str(e)}")
            raise
    
    def _should_use_sequential_processing(self, audio_path: str) -> bool:
        """Determine if sequential processing should be used instead of chunking."""
        try:
            # Check file size and duration
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            duration = librosa.get_duration(path=audio_path)
            
            # Use sequential for large files (>10 minutes OR >200MB)
            large_file_threshold_mb = CONFIG.get('transcribers', {}).get('common', {}).get('large_file_threshold_mb', 200)
            duration_threshold = 600  # 10 minutes
            
            return duration > duration_threshold or file_size_mb > large_file_threshold_mb
            
        except Exception as e:
            self.logger.warning(f"Could not determine if file needs sequential processing: {str(e)}")
            return False
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe the audio file using Whisper with anti-hallucination measures.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with transcription results
        """
        self.logger.info(f"Transcribing with Whisper: {os.path.basename(audio_path)}")
        
        try:
            # Check if we should use sequential processing for large files
            if self._should_use_sequential_processing(audio_path):
                return self._transcribe_sequential(audio_path)
            else:
                return self._transcribe_standard(audio_path)
                
        except Exception as e:
            error_msg = str(e).lower()
            if ("triton" in error_msg or 
                "pointer argument" in error_msg or 
                "nonetype" in error_msg or
                "not subscriptable" in error_msg):
                
                self.logger.warning("Triton/CUDA compatibility issue detected, retrying without word timestamps")
                try:
                    # Retry without word timestamps
                    result = self._transcribe_without_word_timestamps(audio_path)
                    formatted_result = self._format_whisper_result(result)
                    self.logger.info("Transcription completed successfully (without word timestamps)")
                    return formatted_result
                    
                except Exception as retry_e:
                    self.logger.error(f"Retry without word timestamps failed: {str(retry_e)}")
                    raise
            else:
                self.logger.error(f"Transcription failed: {str(e)}")
                raise
    
    def _transcribe_sequential(self, audio_path: str) -> Dict[str, Any]:
        """Process large files sequentially without chunking to avoid hallucinations."""
        self.logger.info("Using sequential processing for large file to prevent hallucinations")
        
        try:
            duration = librosa.get_duration(path=audio_path)
            
            if duration > 1800:  # > 30 minutes, process in segments
                return self._process_in_large_segments(audio_path)
            else:
                # For files under 30 minutes, process as one piece
                result = self.model.transcribe(
                    audio_path,
                    language=self.language,
                    task=self.task,
                    beam_size=self.beam_size,
                    best_of=self.best_of,
                    patience=self.patience,
                    length_penalty=self.length_penalty,
                    temperature=self.temperature,
                    compression_ratio_threshold=self.compression_ratio_threshold,
                    logprob_threshold=self.logprob_threshold,
                    no_speech_threshold=self.no_speech_threshold,
                    condition_on_previous_text=self.condition_on_previous_text,
                    initial_prompt=None,  # Clear prompt for each file
                    word_timestamps=self.word_timestamps,
                    verbose=False
                )
                
                # Convert Whisper format to match expected format
                formatted_result = self._format_whisper_result(result)
                
                # Check for repetitive patterns
                if self._detect_repetitive_patterns(formatted_result.get('text', '')):
                    self.logger.warning("Sequential processing produced repetitive patterns, trying segmented approach")
                    return self._process_in_large_segments(audio_path)
                
                self.logger.info("Sequential transcription completed successfully")
                return formatted_result
                
        except Exception as e:
            self.logger.error(f"Sequential transcription failed: {str(e)}")
            raise
    
    def _process_in_large_segments(self, audio_path: str) -> Dict[str, Any]:
        """Process very large files in 5-minute segments with anti-hallucination measures."""
        self.logger.info("Processing very large file in 5-minute segments")
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Use 5-minute segments with minimal overlap
            segment_duration = 300  # 5 minutes
            overlap_duration = 10   # 10 second overlap
            
            num_segments = int(np.ceil((duration - overlap_duration) / (segment_duration - overlap_duration)))
            
            self.logger.info(f"Segmenting {duration:.2f}s audio into {num_segments} segments of {segment_duration}s with {overlap_duration}s overlap")
            
            all_results = []
            failed_segments = 0
            
            for i in range(num_segments):
                start_time = i * (segment_duration - overlap_duration)
                end_time = min(start_time + segment_duration, duration)
                
                self.logger.info(f"Processing segment {i+1}/{num_segments}: {start_time:.2f}s-{end_time:.2f}s")
                
                # Extract segment
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment = y[start_sample:end_sample]
                
                # Save segment to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    sf.write(temp_path, segment, sr)
                
                try:
                    # Transcribe segment with anti-hallucination settings
                    segment_result = self.model.transcribe(
                        temp_path,
                        language=self.language,
                        task=self.task,
                        beam_size=1,  # Conservative for segments
                        best_of=1,
                        temperature=0.0,  # Deterministic
                        compression_ratio_threshold=self.compression_ratio_threshold,
                        logprob_threshold=self.logprob_threshold,
                        no_speech_threshold=self.no_speech_threshold,
                        condition_on_previous_text=False,  # Never use previous context
                        initial_prompt=None,  # Clean slate for each segment
                        word_timestamps=self.word_timestamps,
                        verbose=False
                    )
                    
                    # Check for repetitive patterns in segment
                    segment_text = segment_result.get('text', '')
                    if self._detect_repetitive_patterns(segment_text):
                        self.logger.warning(f"Segment {i+1} contains repetitive patterns, skipping")
                        failed_segments += 1
                        continue
                    
                    # Adjust timestamps to global time (skip overlap for non-first segments)
                    actual_start_time = start_time
                    if i > 0:
                        actual_start_time += overlap_duration  # Skip overlap region
                    
                    if 'segments' in segment_result:
                        for segment_data in segment_result['segments']:
                            if 'start' in segment_data and 'end' in segment_data:
                                segment_data['start'] += actual_start_time
                                segment_data['end'] += actual_start_time
                            
                            if 'words' in segment_data:
                                for word in segment_data['words']:
                                    if 'start' in word and 'end' in word:
                                        word['start'] += actual_start_time
                                        word['end'] += actual_start_time
                    
                    all_results.append(segment_result)
                    
                except Exception as e:
                    self.logger.error(f"Error processing segment {i+1}: {str(e)}")
                    failed_segments += 1
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            # Check if too many segments failed
            if failed_segments > num_segments * 0.5:
                self.logger.error(f"Too many segments failed ({failed_segments}/{num_segments}), transcript likely unreliable")
                return {"text": "", "chunks": []}
            
            # Merge results with overlap handling
            merged_result = self._merge_whisper_segments_with_overlap_handling(all_results, overlap_duration)
            
            # Final check for repetitive patterns in merged result
            if self._detect_repetitive_patterns(merged_result.get('text', '')):
                self.logger.error("Merged result contains repetitive patterns")
                return {"text": "", "chunks": []}
            
            # Convert to expected format
            formatted_result = self._format_whisper_result(merged_result)
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"Large segment processing failed: {str(e)}")
            raise
    
    def _transcribe_standard(self, audio_path: str) -> Dict[str, Any]:
        """Standard transcription for smaller files."""
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            task=self.task,
            beam_size=self.beam_size,
            best_of=self.best_of,
            patience=self.patience,
            length_penalty=self.length_penalty,
            temperature=self.temperature,
            compression_ratio_threshold=self.compression_ratio_threshold,
            logprob_threshold=self.logprob_threshold,
            no_speech_threshold=self.no_speech_threshold,
            condition_on_previous_text=self.condition_on_previous_text,
            initial_prompt=self.initial_prompt,
            word_timestamps=self.word_timestamps,
            verbose=False
        )
        
        # Convert Whisper format to match expected format
        formatted_result = self._format_whisper_result(result)
        
        # Check for repetitive patterns
        if self._detect_repetitive_patterns(formatted_result.get('text', '')):
            self.logger.warning("Standard transcription produced repetitive patterns, trying sequential approach")
            return self._transcribe_sequential(audio_path)
        
        self.logger.info("Standard transcription completed successfully")
        return formatted_result
    
    def _transcribe_without_word_timestamps(self, audio_path: str) -> Dict[str, Any]:
        """Fallback transcription without word timestamps for compatibility issues."""
        return self.model.transcribe(
            audio_path,
            language=self.language,
            task=self.task,
            beam_size=self.beam_size,
            best_of=self.best_of,
            patience=self.patience,
            length_penalty=self.length_penalty,
            temperature=self.temperature,
            compression_ratio_threshold=self.compression_ratio_threshold,
            logprob_threshold=self.logprob_threshold,
            no_speech_threshold=self.no_speech_threshold,
            condition_on_previous_text=self.condition_on_previous_text,
            initial_prompt=self.initial_prompt,
            word_timestamps=False,
            verbose=False
        )
    
    def _merge_whisper_segments_with_overlap_handling(self, segment_results: List, overlap_duration: float) -> Dict[str, Any]:
        """Merge Whisper segments while handling overlaps."""
        if not segment_results:
            return {"text": "", "segments": []}
        
        merged_text = ""
        all_segments = []
        
        for i, result in enumerate(segment_results):
            segment_text = result.get("text", "").strip()
            
            if i == 0:
                # First segment - take everything
                merged_text = segment_text
            else:
                # Subsequent segments - handle overlap
                if merged_text and segment_text:
                    merged_text += " " + segment_text
                elif segment_text:
                    merged_text = segment_text
            
            # Add segments with timestamp adjustment already applied
            if "segments" in result:
                all_segments.extend(result["segments"])
        
        return {
            "text": merged_text,
            "segments": all_segments
        }
    
    def _format_whisper_result(self, whisper_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Whisper output format to match expected format.
        
        Args:
            whisper_result: Raw result from Whisper
            
        Returns:
            Formatted result dictionary
        """
        formatted = {
            "text": whisper_result.get("text", ""),
            "chunks": []
        }
        
        # Convert segments to chunks if available
        if "segments" in whisper_result:
            for segment in whisper_result["segments"]:
                # Add word-level chunks if available
                if "words" in segment and segment["words"]:
                    for word in segment["words"]:
                        chunk = {
                            "text": word.get("word", ""),
                            "timestamp": (word.get("start", 0), word.get("end", 0))
                        }
                        formatted["chunks"].append(chunk)
                else:
                    # Add segment as a single chunk
                    chunk = {
                        "text": segment.get("text", ""),
                        "timestamp": (segment.get("start", 0), segment.get("end", 0))
                    }
                    formatted["chunks"].append(chunk)
        
        return formatted


class SpeechbankTranscriptionProcessor:
    """Main processor for speechbank transcription tasks."""
    
    def __init__(self, df=None, transcriber_type=None, max_parallel_workers=1, device=None, **transcriber_kwargs):
        """Initialize the processor.
        
        Args:
            df: DataFrame with speechbank files information
            transcriber_type: Type of transcriber to use (None uses default from config)
            max_parallel_workers: Maximum number of parallel workers (default 1 for stability)
            device: Specific device to use (overrides auto-detection)
            **transcriber_kwargs: Additional arguments for the transcriber
        """
        self.df = df
        self.logger = setup_logging('speechbank_transcription')
        self.max_parallel_workers = max_parallel_workers
        self.specified_device = device  # Store the specified device
        
        # Load transcriber settings from config
        transcribers_config = CONFIG.get('transcribers', {})
        self.transcriber_type = transcriber_type or transcribers_config.get('default_transcriber', 'crisper_whisper')
        self.transcriber_kwargs = transcriber_kwargs
        
        # Load common transcription settings
        self.common_config = transcribers_config.get('common', {})
        
        # Setup CUDA memory conservatively
        setup_cuda_memory(memory_fraction=0.7)
        
        # Get available devices
        self.available_devices = self._get_available_devices()
        self.logger.info(f"Available devices: {self.available_devices}")
        
        # Device locks for thread safety
        self.device_locks = {device: threading.Lock() for device in self.available_devices}
        
        # Cache transcribers to avoid reloading models
        self.transcriber_cache = {}
    
    def _get_transcriber(self, device: str) -> BaseTranscriber:
        """Get or create a cached transcriber for the given device with better error handling."""
        if device not in self.transcriber_cache:
            self.logger.info(f"Creating new transcriber for device {device}")
            try:
                self.transcriber_cache[device] = self._create_transcriber(device)
            except Exception as e:
                self.logger.error(f"Failed to create transcriber for {device}: {str(e)}")
                # Try CPU fallback if GPU fails
                if "cuda" in device:
                    self.logger.warning("Falling back to CPU transcriber")
                    self.transcriber_cache["cpu"] = self._create_transcriber("cpu")
                    return self.transcriber_cache["cpu"]
                else:
                    raise
        return self.transcriber_cache[device]
    
    def cleanup_transcribers(self):
        """Clean up all cached transcribers with better error handling."""
        for device, transcriber in list(self.transcriber_cache.items()):
            try:
                self.logger.info(f"Cleaning up transcriber for device {device}")
                transcriber.cleanup()
            except Exception as e:
                self.logger.warning(f"Error cleaning up transcriber for {device}: {str(e)}")
            finally:
                # Always try to remove from cache
                try:
                    del self.transcriber_cache[device]
                except:
                    pass
        
        # Final cleanup attempt
        self.transcriber_cache.clear()
        safe_clean_gpu_memory()
    
    def _get_available_devices(self) -> List[str]:
        """Get list of available devices for processing."""
        devices = []
        
        # If a specific device was specified, use only that device
        if self.specified_device:
            self.logger.info(f"Using specified device: {self.specified_device}")
            
            # Validate the specified device
            if self.specified_device == "cpu":
                devices.append("cpu")
            elif self.specified_device.startswith("cuda"):
                if torch.cuda.is_available():
                    try:
                        # Extract device number if specified (e.g., "cuda:1" -> 1)
                        if ":" in self.specified_device:
                            device_num = int(self.specified_device.split(":")[1])
                        else:
                            device_num = 0
                        
                        # Check if the specified GPU exists and is available
                        if device_num < torch.cuda.device_count():
                            torch.cuda.set_device(device_num)
                            # Test if GPU is working
                            test_tensor = torch.randn(10, device=self.specified_device)
                            del test_tensor
                            devices.append(self.specified_device)
                            self.logger.info(f"Successfully validated specified device: {self.specified_device}")
                        else:
                            self.logger.error(f"Specified GPU {self.specified_device} does not exist. Available GPUs: 0-{torch.cuda.device_count()-1}")
                            devices.append("cpu")
                    except Exception as e:
                        self.logger.error(f"Specified GPU {self.specified_device} is not available: {str(e)}")
                        devices.append("cpu")
                else:
                    self.logger.error(f"CUDA not available, falling back to CPU despite specified device {self.specified_device}")
                    devices.append("cpu")
            else:
                self.logger.error(f"Invalid device specification: {self.specified_device}, falling back to CPU")
                devices.append("cpu")
        
        else:
            # Auto-detect devices (original behavior)
            if torch.cuda.is_available():
                # For stability, prefer fewer parallel workers on fewer devices
                num_gpus = min(torch.cuda.device_count(), self.max_parallel_workers)
                for i in range(num_gpus):
                    try:
                        torch.cuda.set_device(i)
                        # Test if GPU is working
                        test_tensor = torch.randn(10, device=f'cuda:{i}')
                        del test_tensor
                        devices.append(f"cuda:{i}")
                    except Exception as e:
                        self.logger.warning(f"GPU {i} unavailable: {str(e)}")
                
                if not devices:
                    self.logger.warning("No usable CUDA devices found, this will be very slow")
                    devices.append("cpu")
            else:
                devices.append("cpu")
        
        return devices
    
    def _create_transcriber(self, device: str, **kwargs) -> BaseTranscriber:
        """Create a transcriber instance based on type."""
        all_kwargs = {**self.transcriber_kwargs, **kwargs, 'device': device}
        
        if self.transcriber_type == "crisper_whisper":
            # Filter out kwargs that CrisperWhisperTranscriber doesn't accept
            crisper_kwargs = {k: v for k, v in all_kwargs.items() 
                            if k in ['device', 'model_name', 'chunk_length_s', 'batch_size']}
            return CrisperWhisperTranscriber(**crisper_kwargs)
        elif self.transcriber_type == "whisper":
            # Filter out kwargs that WhisperTranscriber doesn't accept
            whisper_kwargs = {k: v for k, v in all_kwargs.items() 
                            if k in ['device', 'model_size', 'download_root', 'in_memory']}
            return WhisperTranscriber(**whisper_kwargs)
        else:
            raise ValueError(f"Unknown transcriber type: {self.transcriber_type}")
    
    def get_files_to_process(self) -> pd.DataFrame:
        """Find files that need transcription, filtered by task transcription settings."""
        if self.df is None:
            raise ValueError("DataFrame not loaded")
        
        # Import ALLOWED_TASKS from speechbank_utils
        from speechbank_utils import ALLOWED_TASKS
        
        # Handle 'all' transcriber type
        if self.transcriber_type == "all":
            # Find files that need ANY transcription
            mask_crisper = (self.df['speechbank_audio_edit_filepath'].notna() & 
                        (self.df['speechbank_audio_edit_filepath'] != '')) & \
                        (self.df['crisper_whisper_raw_filepath'].isna() | 
                        (self.df['crisper_whisper_raw_filepath'] == ''))
            
            mask_whisper = (self.df['speechbank_audio_edit_filepath'].notna() & 
                        (self.df['speechbank_audio_edit_filepath'] != '')) & \
                        (self.df['whisper_raw_filepath'].isna() | 
                        (self.df['whisper_raw_filepath'] == ''))
            
            # Files that need either crisper_whisper OR whisper transcription
            mask = mask_crisper | mask_whisper
            
        else:
            # Determine which column to check based on transcriber type
            if self.transcriber_type == "crisper_whisper":
                transcript_column = 'crisper_whisper_raw_filepath'
            else:
                transcript_column = 'whisper_raw_filepath'
            
            # Select rows that have audio_edit files but no transcript
            mask = (self.df['speechbank_audio_edit_filepath'].notna() & 
                    (self.df['speechbank_audio_edit_filepath'] != '')) & \
                (self.df[transcript_column].isna() | 
                    (self.df[transcript_column] == ''))
        
        # Apply task-based filtering for transcription eligibility
        def should_transcribe_task(taskname):
            """Check if a task should be transcribed based on config."""
            if pd.isna(taskname) or taskname == '':
                return False
            
            task_config = ALLOWED_TASKS.get(taskname, {})
            return task_config.get('transcribe', False)
        
        # Create task filter mask
        task_mask = self.df['speechbank_taskname'].apply(should_transcribe_task)
        
        # Combine file availability mask with task filter
        combined_mask = mask & task_mask
        
        to_process = self.df[combined_mask].copy()
        
        # Filter out rows with non-existent audio files
        valid_rows = []
        
        for idx, row in to_process.iterrows():
            audio_path = row['speechbank_audio_edit_filepath']
            
            if os.path.exists(audio_path):
                valid_rows.append(idx)
            else:
                self.logger.warning(f"Audio file does not exist: {os.path.basename(audio_path)}")
        
        # Log which tasks were filtered out
        all_excluded_by_task = self.df[mask & ~task_mask]
        if len(all_excluded_by_task) > 0:
            excluded_tasks = all_excluded_by_task['speechbank_taskname'].value_counts()
            self.logger.info(f"Excluded {len(all_excluded_by_task)} files based on task transcription settings:")
            for task, count in excluded_tasks.items():
                task_config = ALLOWED_TASKS.get(task, {})
                transcribe_setting = task_config.get('transcribe', False)
                self.logger.info(f"  {task}: {count} files (transcribe: {transcribe_setting})")
        
        to_process = to_process.loc[valid_rows] if valid_rows else to_process.iloc[0:0]
        
        self.logger.info(f"Found {len(to_process)} files to process with {self.transcriber_type}")
        
        # Log breakdown by task for transparency
        if len(to_process) > 0:
            task_breakdown = to_process['speechbank_taskname'].value_counts()
            self.logger.info("Files to process by task:")
            for task, count in task_breakdown.items():
                self.logger.info(f"  {task}: {count} files")
        
        return to_process

    def generate_transcript_path(self, row: pd.Series, audio_path: str) -> str:
        """Generate the base path for transcript files with proper version numbering and _transcript- prefix."""
        audio_dir = os.path.dirname(audio_path)
        base_name = row['speechbank_filename']
        
        # Get version based on transcriber type
        if self.transcriber_type == "crisper_whisper":
            # Hardcoded version for CrisperWhisper
            version = "largev3-v20240930"
            transcript_base = f"{base_name}_transcript-crisper-whisper-raw-ver-{version}"
        else:  # whisper
            # For OpenAI Whisper, get actual model version
            version = self._get_whisper_version()
            transcript_base = f"{base_name}_transcript-whisper-raw-ver-{version}"
        
        return os.path.join(audio_dir, transcript_base)

    def _get_whisper_version(self) -> str:
        """Get version identifier for Whisper model."""
        # Get model size from kwargs or config
        if hasattr(self, 'transcriber_kwargs') and 'model_size' in self.transcriber_kwargs:
            model_size = self.transcriber_kwargs['model_size']
        else:
            model_size = CONFIG.get('transcribers', {}).get('whisper', {}).get('model_size', 'large-v3')
        
        try:
            # Try to get actual Whisper version
            import whisper
            
            # Try to get Whisper package version and format it properly
            whisper_version = getattr(whisper, '__version__', None)
            if whisper_version:
                self.logger.info(f"Whisper package version: {whisper_version}")
                # Format: model-size-vYYYYMMDD
                clean_model_size = model_size.replace('-', '')  # largev3
                clean_whisper_version = whisper_version.replace('.', '')  # e.g., 20240930
                version = f"{clean_model_size}-v{clean_whisper_version}"
                return version
            
            # Try to get version from whisper module file
            import os
            import time
            whisper_file = whisper.__file__
            if os.path.exists(whisper_file):
                # Get modification time of whisper module as a version indicator
                mtime = os.path.getmtime(whisper_file)
                date_str = time.strftime('%Y%m%d', time.localtime(mtime))
                clean_model_size = model_size.replace('-', '')
                version = f"{clean_model_size}-v{date_str}"
                self.logger.info(f"Using Whisper file date as version: {version}")
                return version
        
        except Exception as e:
            self.logger.warning(f"Error getting Whisper version: {str(e)}")
        
        # Use fallback version with today's date
        from datetime import datetime
        fallback_date = datetime.now().strftime('%Y%m%d')
        clean_model_size = model_size.replace('-', '')
        fallback_version = f"{clean_model_size}-v{fallback_date}"
        self.logger.warning(f"Using fallback version: {fallback_version}")
        return fallback_version
    
    def process_single_file(self, idx: int, row: pd.Series, device: str, retry_count: int = 0) -> Tuple[bool, Optional[str]]:
        """Process a single file with QC checks and retry logic.
        
        Args:
            idx: Index of the row in the DataFrame
            row: DataFrame row with file information
            device: Device to use for processing
            retry_count: Current retry attempt
            
        Returns:
            tuple: (success, txt_path)
        """
        audio_path = row['speechbank_audio_edit_filepath']
        
        # Skip if audio path is missing
        if not audio_path or pd.isna(audio_path):
            self.logger.warning(f"Skipping row with empty audio path: {row.get('HELPERID', 'unknown')}")
            return False, None
        
        # Check if file exists
        if not os.path.exists(audio_path):
            self.logger.warning(f"Audio file does not exist: {os.path.basename(audio_path)}")
            return False, None
        
        # Generate transcript base path (includes _transcript- prefix)
        transcript_base_path = self.generate_transcript_path(row, audio_path)
        
        self.logger.info(f"Processing: {os.path.basename(audio_path)} on {device} (attempt {retry_count + 1})")
        self.logger.info(f"Output: {os.path.basename(transcript_base_path)}.txt/.json")
        
        # For retries, clean up transcriber BEFORE acquiring lock
        if retry_count > 0:
            self.logger.info(f"Retry attempt {retry_count + 1}: Cleaning up transcriber for clean state")
            
            # Remove the cached transcriber without holding the lock
            if device in self.transcriber_cache:
                try:
                    # Create a temporary reference to the transcriber
                    old_transcriber = self.transcriber_cache.get(device)
                    # Remove from cache first
                    del self.transcriber_cache[device]
                    # Then cleanup outside of any locks
                    if old_transcriber:
                        try:
                            old_transcriber.cleanup()
                        except Exception as e:
                            self.logger.warning(f"Error cleaning up old transcriber: {str(e)}")
                        finally:
                            del old_transcriber
                except Exception as e:
                    self.logger.warning(f"Error removing cached transcriber: {str(e)}")
            
            # Additional memory cleanup for retries
            safe_clean_gpu_memory()
            time.sleep(2.0)  # Longer pause for retries
        
        # Now acquire device lock for the actual processing
        lock_acquired = False
        try:
            # Use a timeout on lock acquisition to prevent indefinite blocking
            lock_acquired = self.device_locks[device].acquire(timeout=60)
            
            if not lock_acquired:
                self.logger.error(f"Failed to acquire device lock for {device} within timeout")
                return False, None
            
            # Clean GPU memory before processing
            if "cuda" in device:
                torch.cuda.set_device(device)
                safe_clean_gpu_memory()
            
            try:
                # Get cached transcriber (or create new one if removed)
                transcriber = self._get_transcriber(device)
                
                # Transcribe with timeout mechanism
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Transcription timed out")
                
                # Set a timeout for transcription (5 minutes per file should be plenty)
                if hasattr(signal, 'SIGALRM'):  # Unix/Linux only
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(300)  # 5 minute timeout
                    
                try:
                    result = transcriber.transcribe_audio(audio_path)
                finally:
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)  # Cancel the alarm
                        signal.signal(signal.SIGALRM, old_handler)
                
                # Check if transcript is empty or repetitive
                if transcriber._is_transcript_empty(result):
                    self.logger.warning(f"Empty or repetitive transcript detected for {os.path.basename(audio_path)}")
                    
                    # Release lock before retry
                    if lock_acquired:
                        self.device_locks[device].release()
                        lock_acquired = False
                    
                    if retry_count < 2:  # Allow up to 2 retries for hallucinations
                        self.logger.info(f"Retrying transcription for {os.path.basename(audio_path)} (attempt {retry_count + 2})")
                        return self.process_single_file(idx, row, device, retry_count + 1)
                    else:
                        self.logger.error(f"Transcript still empty/repetitive after {retry_count + 1} attempts for {os.path.basename(audio_path)}. Skipping file.")
                        return False, None
                
                # Save transcript (this will check for empty again and handle duplicates)
                txt_path, success = transcriber.save_transcript(result, transcript_base_path)
                
                if not success:
                    # Release lock before retry
                    if lock_acquired:
                        self.device_locks[device].release()
                        lock_acquired = False
                        
                    if retry_count < 1:  # Allow one retry for save failures
                        self.logger.warning(f"Failed to save transcript for {os.path.basename(audio_path)}, retrying...")
                        return self.process_single_file(idx, row, device, retry_count + 1)
                    else:
                        self.logger.error(f"Failed to save transcript for {os.path.basename(audio_path)} after retry. Skipping file.")
                        return False, None
                
                return success, txt_path
                
            except TimeoutError:
                self.logger.error(f"Transcription timed out for {os.path.basename(audio_path)}")
                return False, None
                
            except Exception as e:
                self.logger.error(f"Transcription failed on {device}: {str(e)}")
                
                # For certain types of errors, don't retry to avoid infinite loops
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['out of memory', 'cuda error', 'device-side assert']):
                    self.logger.error(f"Critical GPU error detected, not retrying: {str(e)}")
                    return False, None
                
                # Release lock before retry
                if lock_acquired:
                    self.device_locks[device].release()
                    lock_acquired = False
                
                if retry_count < 2:  # Allow up to 2 retries for general failures
                    self.logger.info(f"Retrying due to error for {os.path.basename(audio_path)} (attempt {retry_count + 2})")
                    
                    # Force transcriber recreation for error retries
                    if device in self.transcriber_cache:
                        try:
                            old_transcriber = self.transcriber_cache.get(device)
                            del self.transcriber_cache[device]
                            if old_transcriber:
                                old_transcriber.cleanup()
                        except Exception as cleanup_e:
                            self.logger.warning(f"Error cleaning up transcriber after error: {str(cleanup_e)}")
                    
                    # Clean memory before retry
                    safe_clean_gpu_memory()
                    time.sleep(2.0)
                    return self.process_single_file(idx, row, device, retry_count + 1)
                else:
                    self.logger.error(f"Transcription failed for {os.path.basename(audio_path)} after {retry_count + 1} attempts. Skipping file.")
                    return False, None
                    
        finally:
            # Always release the lock if we acquired it
            if lock_acquired:
                self.device_locks[device].release()
    
    def process_files(self) -> pd.DataFrame:
        """Process all files that need transcription."""
        to_process = self.get_files_to_process()
        
        if len(to_process) == 0:
            self.logger.info("No files to process")
            return self.df
        
        # Handle 'all' transcriber type by running both transcribers
        if self.transcriber_type == "all":
            self.logger.info("Running all available transcribers (crisper_whisper and whisper)")
            
            # Store original transcriber kwargs and type
            original_transcriber_type = self.transcriber_type
            original_kwargs = self.transcriber_kwargs.copy()
            
            # Process with crisper_whisper first
            print(f"\n\n")
            self.logger.info("Running CrisperWhisper Transcriber")
            self.transcriber_type = "crisper_whisper"
            # Filter kwargs for crisper_whisper (remove model_size if present)
            self.transcriber_kwargs = {k: v for k, v in original_kwargs.items() 
                                    if k not in ['model_size']}
            self.cleanup_transcribers()  # Clear cache
            self._process_files_with_transcriber(to_process)
            
            # Process with whisper second
            print(f"\n\n")
            self.logger.info("Running Whisper Transcriber")
            self.transcriber_type = "whisper"
            # Use original kwargs for whisper (includes model_size)
            self.transcriber_kwargs = original_kwargs.copy()
            self.cleanup_transcribers()  # Clear cache
            self._process_files_with_transcriber(to_process)
            
            # Restore original settings
            self.transcriber_type = original_transcriber_type
            self.transcriber_kwargs = original_kwargs
            
        else:
            # Process with single transcriber
            self._process_files_with_transcriber(to_process)
        
        # Clean up transcribers at the end
        self.cleanup_transcribers()
        
        return self.df

    def _process_files_with_transcriber(self, to_process: pd.DataFrame) -> None:
        """Process files with the current transcriber type."""
        self.logger.info(f"Starting transcription for {len(to_process)} files with {self.transcriber_type}")
        
        # Sort by file size (smaller first) for better load balancing
        to_process['file_size_mb'] = to_process['speechbank_audio_edit_filepath'].apply(
            lambda x: os.path.getsize(x) / (1024 * 1024) if os.path.exists(x) else 0
        )
        to_process = to_process.sort_values('file_size_mb')
        
        # Clean memory before starting
        safe_clean_gpu_memory()
        
        # Clear transcriber cache when switching transcriber types
        # This ensures we don't have stale transcribers with wrong parameters
        self.transcriber_cache.clear()
        
        # Prepare file assignments
        file_assignments = []
        device_idx = 0
        
        for i, (idx, row) in enumerate(to_process.iterrows()):
            device = self.available_devices[device_idx % len(self.available_devices)]
            file_assignments.append((idx, row, device))
            device_idx += 1
        
        successes = 0
        
        # Process files
        if len(self.available_devices) > 1 and self.max_parallel_workers > 1:
            self.logger.info(f"Processing files in parallel across {len(self.available_devices)} devices")
            
            # Use ThreadPoolExecutor for parallel processing
            max_workers = min(len(self.available_devices), self.max_parallel_workers)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_assignment = {
                    executor.submit(self._process_file_worker, assignment): assignment 
                    for assignment in file_assignments
                }
                
                # Process completed tasks
                for future in tqdm(concurrent.futures.as_completed(future_to_assignment), 
                                total=len(file_assignments), desc=f"Transcribing ({self.transcriber_type})"):
                    assignment = future_to_assignment[future]
                    idx, _, _ = assignment
                    
                    try:
                        result_idx, success, txt_path = future.result()
                        
                        if success and txt_path:
                            # Update dataframe with transcript path
                            if self.transcriber_type == "crisper_whisper":
                                self.df.loc[result_idx, 'crisper_whisper_raw_filepath'] = txt_path
                                self.df.loc[result_idx, 'CRISPER_WHISPER_RAW'] = 1
                            else:
                                self.df.loc[result_idx, 'whisper_raw_filepath'] = txt_path
                                self.df.loc[result_idx, 'WHISPER_RAW'] = 1
                            successes += 1
                        
                    except Exception as e:
                        row = to_process.loc[idx]
                        self.logger.error(f"Failed to process {row.get('HELPERID', 'unknown')}: {str(e)}")
        
        else:
            # Sequential processing
            self.logger.info("Processing files sequentially")
            
            for idx, row, device in tqdm(file_assignments, desc=f"Transcribing ({self.transcriber_type})"):
                try:
                    success, txt_path = self.process_single_file(idx, row, device)
                    
                    if success and txt_path:
                        # Update dataframe with transcript path
                        if self.transcriber_type == "crisper_whisper":
                            self.df.loc[idx, 'crisper_whisper_raw_filepath'] = txt_path
                            self.df.loc[idx, 'CRISPER_WHISPER_RAW'] = 1
                        else:
                            self.df.loc[idx, 'whisper_raw_filepath'] = txt_path
                            self.df.loc[idx, 'WHISPER_RAW'] = 1
                        successes += 1
                    
                    # Clean memory between files
                    if "cuda" in device:
                        safe_clean_gpu_memory()
                        time.sleep(0.5)  # Brief pause for GPU recovery
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {row.get('HELPERID', 'unknown')}: {str(e)}")
        
        self.logger.info(f"Processed {successes}/{len(to_process)} files successfully with {self.transcriber_type}")
    
    def _process_file_worker(self, file_info: Tuple[int, pd.Series, str]) -> Tuple[int, bool, Optional[str]]:
        """Worker function for parallel processing."""
        idx, row, device = file_info
        success, txt_path = self.process_single_file(idx, row, device)
        return idx, success, txt_path


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Speechbank Transcriber - Transcribe audio files with anti-hallucination measures')
    parser.add_argument('--transcriber', default='all',
                        choices=['all', 'crisper_whisper', 'whisper'], 
                        help='Transcriber model to use (use "all" to run both)')
    parser.add_argument('--whisper-model', default='large-v3', 
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Whisper model size (if using whisper transcriber)')
    parser.add_argument('--device', help='Device to use (cuda/cpu/cuda:0/etc)')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='Maximum number of parallel workers (default: 1 for stability)')
    parser.add_argument('--sequential', action='store_true',
                       help='Force sequential processing (no parallelization)')
    
    args = parser.parse_args()
    
    try:        
        # Run the assembler first to get the DataFrame
        logger = setup_logging('speechbank_transcriber')
        logger.info("Running speechbank assembler first...")
        
        # The run_assembler function now returns (df, invalid_files)
        df, invalid_files = run_assembler(save_output=False)
        
        # Log assembler results
        logger.info(f"Assembler found {len(df)} valid file families")
        if invalid_files:
            logger.info(f"Assembler found {len(invalid_files)} invalid files")

        # Prepare transcriber kwargs
        transcriber_kwargs = {}
        if args.whisper_model and (args.transcriber == 'whisper' or args.transcriber == 'all'):
            transcriber_kwargs['model_size'] = args.whisper_model
        
        # Set max workers (force to 1 if sequential)
        max_workers = 1 if args.sequential else args.max_workers
        
        # Initialize transcription processor
        logger.info(f"Initializing transcription processor with {args.transcriber}...")
        if args.device:
            logger.info(f"Using specified device: {args.device}")
        logger.info("Enhanced anti-hallucination measures enabled:")
        logger.info("  - condition_on_previous_text=False")
        logger.info("  - Sequential processing for large files")
        logger.info("  - Repetitive pattern detection and filtering")
        logger.info("  - Robust timestamp handling to prevent NoneType errors")
        logger.info("  - Retry logic for failed chunks")
        logger.info("  - Repetition penalty and n-gram blocking")
        
        processor = SpeechbankTranscriptionProcessor(
            df=df, 
            transcriber_type=args.transcriber,
            max_parallel_workers=max_workers,
            device=args.device,  # Pass the device parameter here
            **transcriber_kwargs
        )
        
        # Process files
        logger.info("Starting transcription process...")
        updated_df = processor.process_files()
        
        logger.info("Transcription process completed.")
        
    except Exception as e:
        logger.error(f"Error during transcription process: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
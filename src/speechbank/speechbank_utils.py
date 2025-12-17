"""
Speechbank Utilities Module
Contains shared functions and utilities used across all speechbank scripts.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime

# Load configuration files
def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """Load main configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_file_types(file_types_path: str = 'speechbank_file_types.json') -> Dict[str, List[str]]:
    """Load allowed file types configuration."""
    with open(file_types_path, 'r') as f:
        return json.load(f)


def load_languages(languages_path: str = 'speechbank_languages.json') -> Dict[str, str]:
    """Load allowed languages configuration."""
    with open(languages_path, 'r') as f:
        return json.load(f)


def load_tasks(tasks_path: str = 'speechbank_tasks.json') -> Dict[str, Dict[str, Any]]:
    """Load allowed tasks configuration."""
    with open(tasks_path, 'r') as f:
        return json.load(f)


# Load all configs at module level for easy access
CONFIG = load_config()
ALLOWED_FILE_TYPES = load_file_types()
ALLOWED_LANGUAGES = load_languages()
ALLOWED_TASKS = load_tasks()


class FilenameValidation:
    """Class to handle filename validation and component extraction."""
    
    def __init__(self):
        self.base_prefixes = set(CONFIG['filename_structure']['base_prefixes'])
        self.suffix_prefixes = set(CONFIG['filename_structure']['suffix_prefixes'])
        self.special_suffixes = set(CONFIG['filename_structure']['special_suffixes'])
        self.transcript_types = CONFIG['filename_structure']['transcript_types']
        self.audio_edit_flags = CONFIG['filename_structure']['audio_edit_flags']
    
    def parse_filename_components(self, filename: str) -> Tuple[Dict[str, str], List[str]]:
        """Extract components from filename using new structure.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Tuple of (components dict, list of validation errors)
        """
        base = Path(filename).stem
        components = {
            'pidn': '',
            'speechbank_ses_date': '',
            'speechbank_taskname': '',
            'speechbank_langcode': '',
            'speechbank_extrafiletag': '',
            'transcript_type': '',
            'edit_components': [],
            'is_final': False
        }
        errors = []
        
        # Check for _final suffix first
        if base.endswith('_final'):
            components['is_final'] = True
            base = base[:-6]  # Remove '_final'
        
        # Split by underscores
        parts = base.split('_')
        base_parts = []
        suffix_parts = []
        found_suffix = False
        
        for part in parts:
            if not found_suffix:
                # Check if this part starts with a base prefix
                is_base_part = any(part.startswith(prefix) for prefix in self.base_prefixes)
                if is_base_part:
                    base_parts.append(part)
                else:
                    # Check if this part starts with a suffix prefix
                    is_suffix_part = any(part.startswith(prefix) for prefix in self.suffix_prefixes)
                    if is_suffix_part:
                        found_suffix = True
                        suffix_parts.append(part)
                    else:
                        errors.append(f"Invalid filename component: '{part}' - does not match expected prefixes")
            else:
                suffix_parts.append(part)
        
        # Parse base components
        for part in base_parts:
            if part.startswith('sub-'):
                components['pidn'] = part[4:]
            elif part.startswith('ses-'):
                components['speechbank_ses_date'] = part[4:]
            elif part.startswith('task-'):
                components['speechbank_taskname'] = part[5:]
            elif part.startswith('lang-'):
                components['speechbank_langcode'] = part[5:]
            elif part.startswith('tag-'):
                components['speechbank_extrafiletag'] = part[4:]
        
        # Parse suffix components
        for part in suffix_parts:
            if part.startswith('transcript-'):
                transcript_content = part[11:]  # Remove 'transcript-' prefix
                components['transcript_type'] = transcript_content
            elif part.startswith('edit-'):
                edit_content = part[5:]  # Remove 'edit-' prefix
                components['edit_components'] = edit_content.split('-')
        
        return components, errors
    
    def get_base_filename(self, filename: str) -> str:
        """Extract base filename from components that should be grouped together.
        
        Args:
            filename: Filename to process
            
        Returns:
            Base filename for grouping
        """
        components, _ = self.parse_filename_components(filename)
        
        # Build base filename from core components
        base_parts = []
        if components['pidn']:
            base_parts.append(f"sub-{components['pidn']}")
        if components['speechbank_ses_date']:
            base_parts.append(f"ses-{components['speechbank_ses_date']}")
        if components['speechbank_taskname']:
            base_parts.append(f"task-{components['speechbank_taskname']}")
        if components['speechbank_langcode']:
            base_parts.append(f"lang-{components['speechbank_langcode']}")
        if components['speechbank_extrafiletag']:
            base_parts.append(f"tag-{components['speechbank_extrafiletag']}")
        
        return '_'.join(base_parts)
    
    def validate_filename(self, filename: str) -> Tuple[bool, List[str]]:
        """Validate filename structure.
        
        Args:
            filename: Filename to validate
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        components, errors = self.parse_filename_components(filename)
        
        # Check required components
        if not components['pidn']:
            errors.append("Missing required component: sub-")
        if not components['speechbank_ses_date']:
            errors.append("Missing required component: ses-")
        if not components['speechbank_taskname']:
            errors.append("Missing required component: task-")
        
        # Validate date format
        if components['speechbank_ses_date'] and len(components['speechbank_ses_date']) != 8:
            errors.append(f"Invalid date format: {components['speechbank_ses_date']} (expected YYYYMMDD)")
        
        # Validate task name
        if components['speechbank_taskname'] and components['speechbank_taskname'] not in ALLOWED_TASKS:
            errors.append(f"Invalid task name: {components['speechbank_taskname']}")
        
        # Validate language code if present
        if components['speechbank_langcode'] and components['speechbank_langcode'] not in ALLOWED_LANGUAGES.values():
            errors.append(f"Invalid language code: {components['speechbank_langcode']}")
        
        return len(errors) == 0, errors
    
    def get_transcript_type_info(self, transcript_type: str) -> Optional[Dict[str, Any]]:
        """Get transcript type configuration.
        
        Args:
            transcript_type: Transcript type string from filename
            
        Returns:
            Configuration dict for transcript type or None
        """
        for type_name, config in self.transcript_types.items():
            for pattern in config['patterns']:
                if transcript_type.startswith(pattern):
                    return {
                        'type_name': type_name,
                        'config': config
                    }
        return None


# Create global instance for easy access
filename_validator = FilenameValidation()


# Updated utility functions using new validation system
def parse_filename_components(filename: str) -> Dict[str, str]:
    """Extract components from filename using new structure.
    
    Args:
        filename: Filename to parse
        
    Returns:
        Dictionary with extracted components
    """
    components, _ = filename_validator.parse_filename_components(filename)
    return components


def get_base_filename(filename: str) -> str:
    """Extract base filename for grouping files together.
    
    Args:
        filename: Filename to process
        
    Returns:
        Base filename for grouping
    """
    return filename_validator.get_base_filename(filename)


def validate_filename(filename: str) -> Tuple[bool, List[str]]:
    """Validate filename structure.
    
    Args:
        filename: Filename to validate
        
    Returns:
        Tuple of (is_valid, list of errors)
    """
    return filename_validator.validate_filename(filename)


def should_ignore_file(filename: str) -> bool:
    """Check if a file should be ignored based on filename patterns.
    
    Args:
        filename: Filename to check
        
    Returns:
        True if file should be ignored
    """
    ignore_patterns = [re.compile(pattern) for pattern in CONFIG.get('ignore_patterns', [])]
    for pattern in ignore_patterns:
        if pattern.search(filename):
            return True
    return False


def format_date(date_str: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD format.
    
    Args:
        date_str: Date string in YYYYMMDD format
        
    Returns:
        Formatted date string
    """
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str


def get_filetype_info(filepath: str) -> Dict[str, Any]:
    """Determine filetype category and metadata based on filename and extension.
    
    Args:
        filepath: Path to check
        
    Returns:
        Dictionary with filetype information
    """
    filename = Path(filepath).name
    ext = Path(filepath).suffix[1:]  # Get extension without dot
    components, errors = filename_validator.parse_filename_components(filename)
    
    # Create extension sets for faster lookup
    audio_extensions = set(ALLOWED_FILE_TYPES['audio'])
    video_extensions = set(ALLOWED_FILE_TYPES['video'])
    transcript_extensions = set(ALLOWED_FILE_TYPES['transcript'])
    
    result = {
        'category': 'other',
        'is_valid': len(errors) == 0,
        'errors': errors,
        'components': components
    }
    
    # Determine file category
    if ext in audio_extensions:
        if components['is_final']:
            result['category'] = 'audio_final'
        elif components['edit_components']:
            result['category'] = 'audio_edit'
        else:
            result['category'] = 'audio_raw'
    elif ext in video_extensions:
        result['category'] = 'video_raw'
    elif ext == 'json':
        result['category'] = 'json'
        # Check if it's a transcript JSON by looking at the base filename
        base_name = Path(filepath).stem
        if '_transcript-' in base_name:
            result['category'] = 'transcript_json'
    elif ext in transcript_extensions:
        if components['transcript_type']:
            transcript_info = filename_validator.get_transcript_type_info(components['transcript_type'])
            if transcript_info:
                result['category'] = 'transcript'
                result['transcript_info'] = transcript_info
            else:
                result['category'] = 'transcript_unknown'
        else:
            result['category'] = 'transcript_generic'
    
    return result


# Logging setup
def setup_logging(name: str = 'speechbank', level: str = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        name: Logger name
        level: Logging level (defaults to config setting)
        
    Returns:
        Configured logger
    """
    if level is None:
        level = CONFIG.get('logging_level', 'INFO')
    
    # Create logs directory if it doesn't exist
    log_dir = Path(CONFIG.get('log_files_dir', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(name)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


# Directory path helpers
def get_speechbank_dir() -> Path:
    """Get the main speechbank directory path."""
    return Path(CONFIG['speechbank_dir'])


def get_archive_dir() -> Path:
    """Get the archive directory path."""
    return Path(CONFIG['archive_dir'])


def get_quarantine_dir() -> Path:
    """Get the quarantine directory path."""
    return Path(CONFIG['quarantine_dir'])


def get_metrics_dir() -> Path:
    """Get the metrics directory path."""
    return Path(CONFIG['metrics_dir'])


def get_backups_dir() -> Path:
    """Get the backups directory path."""
    return Path(CONFIG['backups_dir'])


# Utility functions for file operations
def ensure_directory_exists(filepath: str) -> None:
    """Ensure the directory for the given filepath exists.
    
    Args:
        filepath: File path to check
    """
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)


def generate_helperid(pidn: str, date: str) -> str:
    """Generate HELPERID from PIDN and date.
    
    Args:
        pidn:
        date: Date in YYYY-MM-DD format
        
    Returns:
        Generated HELPERID
    """
    pidn_value = int(pidn) if pidn.isdigit() else pidn
    return f"{pidn_value}_{date}"


def get_default_columns() -> Dict[str, Any]:
    """Get default column values from config.
    
    Returns:
        Dictionary with all default column values
    """
    columns = {}
    default_columns_config = CONFIG.get('default_columns', {})
    
    # Add main columns first
    columns.update(default_columns_config.get('main_columns', {}))
    
    # Add filepath columns
    columns.update(default_columns_config.get('filepath_columns', {}))
    
    # Add status columns
    columns.update(default_columns_config.get('status_columns', {}))
    
    # Add other columns
    columns.update(default_columns_config.get('other_columns', {}))
    
    return columns


# GPU/CPU utility functions
def get_device(device: Optional[str] = None) -> str:
    """Get the device to use for computation.
    
    Args:
        device: Specific device to use ("cuda", "cuda:0", "cpu", or None for auto-select)
        
    Returns:
        Device string to use
    """
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def select_best_gpu(min_free_memory_mb: float = 8000) -> Optional[int]:
    """Select the GPU with most available memory.
    
    Args:
        min_free_memory_mb: Minimum free memory required in MB
        
    Returns:
        GPU device ID with most free memory, or None if no suitable GPU
    """
    import torch

    if not torch.cuda.is_available():
        return None
    
    # Clean memory on all GPUs first
    clean_gpu_memory()
    
    best_device = None
    max_free_memory = 0
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.synchronize()
        
        # Get actual available memory
        memory_reserved = torch.cuda.memory_reserved(i)
        memory_allocated = torch.cuda.memory_allocated(i)
        memory_total = torch.cuda.get_device_properties(i).total_memory
        memory_free = memory_total - memory_reserved
        
        memory_free_mb = memory_free / 1024**2
        
        if memory_free_mb >= min_free_memory_mb and memory_free_mb > max_free_memory:
            max_free_memory = memory_free_mb
            best_device = i
    
    return best_device


def clean_gpu_memory():
    """Clean GPU memory by emptying cache and running garbage collection."""
    import gc
    import torch
    
    if torch.cuda.is_available():
        # Clear cache on all devices
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
            except:
                pass  # Skip problematic devices
        
        # Synchronize all devices
        try:
            torch.cuda.synchronize()
        except:
            pass
            
    # Force Python garbage collection
    gc.collect()


def setup_cuda_memory(memory_fraction: float = 0.7):
    """Setup CUDA memory settings for optimal performance.
    
    Args:
        memory_fraction: Fraction of GPU memory to use (0.0-1.0)
    """
    import torch
    
    if torch.cuda.is_available():
        # Set environment variables for better memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,expandable_segments:True'
        
        # Set memory fraction for each GPU
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                torch.cuda.set_per_process_memory_fraction(memory_fraction, device=i)
            except:
                pass  # Skip problematic devices
        
        # Force immediate allocation to detect issues early
        try:
            torch.cuda.empty_cache()
        except:
            pass


def log_gpu_memory_state(logger: Optional[logging.Logger] = None):
    """Log current GPU memory state for all available GPUs.
    
    Args:
        logger: Logger to use (creates one if not provided)
    """
    import torch
    
    if not torch.cuda.is_available():
        return
        
    if logger is None:
        logger = logging.getLogger(__name__)
        
    for i in range(torch.cuda.device_count()):
        try:
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = torch.cuda.get_device_properties(i).total_memory
            free = total - allocated
            
            logger.info(f"GPU {i} memory: {free / 1024**2:.2f}MB free / {total / 1024**2:.2f}MB total "
                       f"({allocated / 1024**2:.2f}MB allocated, {reserved / 1024**2:.2f}MB reserved)")
        except Exception as e:
            logger.warning(f"Could not get memory info for GPU {i}: {str(e)}")


# Import error handling
try:
    import torch
    
except ImportError:
    # If torch is not installed, GPU functions will return CPU defaults
    def get_device(device: Optional[str] = None) -> str:
        return device or "cpu"
    
    def select_best_gpu() -> Optional[int]:
        return None
    
    def clean_gpu_memory():
        pass
    
    def setup_cuda_memory(memory_fraction: float = 0.7):
        pass
    
    def log_gpu_memory_state(logger: Optional[logging.Logger] = None):
        pass
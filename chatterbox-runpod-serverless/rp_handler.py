import runpod
import torch
import torchaudio
import nltk
import os
import time
import base64
import tempfile
import logging
from typing import Optional, Dict, Any, List
import traceback
import subprocess

# Import ChatterboxTTS
try:
    from chatterbox.tts import ChatterboxTTS
except ImportError as e:
    print(f"Error: Could not import ChatterboxTTS: {e}")
    ChatterboxTTS = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model caching
tts_model = None
device = None

# Model cache configuration
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', '/runpod-volume/models')
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def diagnose_cuda():
    """Comprehensive CUDA diagnostics"""
    logger.info("=== CUDA Diagnostics ===")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}")
            logger.info(f"  Compute capability: {props.major}.{props.minor}")
            logger.info(f"  Memory: {props.total_memory / 1e9:.2f} GB")
            logger.info(f"  Multi-processors: {props.multi_processor_count}")
    else:
        # Check system level NVIDIA availability
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.warning("nvidia-smi found but PyTorch cannot access CUDA!")
                logger.info("nvidia-smi output (first 500 chars):")
                logger.info(result.stdout[:500])
            else:
                logger.error("nvidia-smi not found - no NVIDIA driver installed")
        except FileNotFoundError:
            logger.error("nvidia-smi command not found")

def ensure_nltk_data():
    """Ensure NLTK punkt tokenizer is available."""
    nltk_data_dir = os.path.join(MODEL_CACHE_DIR, 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        logger.info("NLTK punkt tokenizer found")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
        logger.info("NLTK punkt tokenizer downloaded")

def initialize_model():
    """Initialize the ChatterboxTTS model with optimizations."""
    global tts_model, device
    
    if tts_model is not None:
        logger.info("Model already initialized")
        return tts_model
    
    logger.info("Initializing ChatterboxTTS model...")
    
    # Run CUDA diagnostics
    diagnose_cuda()
    
    if ChatterboxTTS is None:
        raise RuntimeError("ChatterboxTTS module not available")
    
    # Set environment variables for model caching
    os.environ['HF_HOME'] = MODEL_CACHE_DIR
    os.environ['TORCH_HOME'] = MODEL_CACHE_DIR
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(MODEL_CACHE_DIR, 'transformers')
    
    # Determine device with explicit GPU selection
    if torch.cuda.is_available():
        device = "cuda:0"
        torch.cuda.set_device(0)
        # Enable TF32 for A100/H100 GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        logger.error("CUDA not available - falling back to CPU")
        logger.error("This will be VERY slow. Please ensure GPU is properly configured.")
    
    try:
        start_load = time.time()
        
        # Load model with memory optimization
        with torch.cuda.amp.autocast(enabled=(device != "cpu")):
            tts_model = ChatterboxTTS.from_pretrained(
                device=device,
                use_safetensors=True  # Faster loading if supported
            )
        
        load_time = time.time() - start_load
        logger.info(f"Model loaded in {load_time:.2f}s on {device}")
        logger.info(f"Model sample rate: {tts_model.sr} Hz")
        
        # Warm up the model with a tiny generation
        if device != "cpu":
            logger.info("Warming up model...")
            with torch.no_grad():
                _ = tts_model.generate("test", temperature=0.1)
            logger.info("Model warmup complete")
        
        # Ensure NLTK data
        ensure_nltk_data()
        
        return tts_model
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        logger.error(traceback.format_exc())
        raise

def decode_audio_prompt_base64(audio_base64: str) -> Optional[str]:
    """Decode base64 audio data and save to temporary file."""
    try:
        audio_data = base64.b64decode(audio_base64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            return tmp_file.name
            
    except Exception as e:
        logger.error(f"Failed to decode audio prompt: {e}")
        return None

def split_text_into_chunks(text: str, max_chars: int = 200) -> List[str]:
    """Split text into chunks, optimized for speed."""
    try:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
            else:
                current_chunk += sentence + " "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    except:
        # Fallback to simple splitting
        return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def generate_audio_chunk(
    text_chunk: str,
    model: ChatterboxTTS,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.7,
    cfg_weight: float = 0.4,
    exaggeration: float = 0.4
) -> Optional[torch.Tensor]:
    """Generate audio for a single chunk with GPU optimization."""
    try:
        start_gen = time.time()
        
        # Use automatic mixed precision for faster generation
        with torch.cuda.amp.autocast(enabled=(device != "cpu")):
            wav_tensor = model.generate(
                text_chunk,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration
            )
        
        gen_time = time.time() - start_gen
        logger.info(f"Generated {len(text_chunk)} chars in {gen_time:.2f}s")
        
        # Ensure proper shape [1, samples]
        wav_tensor = wav_tensor.cpu().float()
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        
        return wav_tensor
        
    except Exception as e:
        logger.error(f"Error generating audio chunk: {e}")
        return None

def text_to_speech_pipeline(
    text: str,
    model: ChatterboxTTS,
    max_chars_per_chunk: int = 200,
    inter_chunk_silence_ms: int = 250,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.7,
    cfg_weight: float = 0.4,
    exaggeration: float = 0.4
) -> Optional[torch.Tensor]:
    """Optimized TTS pipeline."""
    try:
        chunks = split_text_into_chunks(text, max_chars_per_chunk)
        audio_tensors = []
        sample_rate = model.sr
        
        logger.info(f"Processing {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            chunk_tensor = generate_audio_chunk(
                chunk, model, audio_prompt_path,
                temperature, cfg_weight, exaggeration
            )
            
            if chunk_tensor is not None:
                audio_tensors.append(chunk_tensor)
                
                # Add silence between chunks
                if i < len(chunks) - 1 and inter_chunk_silence_ms > 0:
                    silence_samples = int(sample_rate * inter_chunk_silence_ms / 1000)
                    silence = torch.zeros((1, silence_samples), dtype=chunk_tensor.dtype)
                    audio_tensors.append(silence)
        
        if not audio_tensors:
            return None
        
        return torch.cat(audio_tensors, dim=1)
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return None

def audio_tensor_to_base64(audio_tensor: torch.Tensor, sample_rate: int) -> str:
    """Convert audio tensor to base64 WAV."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        torchaudio.save(tmp.name, audio_tensor, sample_rate)
        
        with open(tmp.name, 'rb') as f:
            audio_data = f.read()
        
        os.unlink(tmp.name)
        return base64.b64encode(audio_data).decode('utf-8')

def validate_input(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Validate input parameters."""
    # Accept both 'text' and 'prompt'
    text = job_input.get('text') or job_input.get('prompt')
    if not text:
        raise ValueError("Missing required parameter: 'text' or 'prompt'")
    
    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty")
    
    return {
        'text': text,
        'audio_prompt_base64': job_input.get('audio_prompt_base64'),
        'max_chars_per_chunk': min(job_input.get('max_chars_per_chunk', 200), 300),
        'inter_chunk_silence_ms': job_input.get('inter_chunk_silence_ms', 250),
        'temperature': max(0.1, min(job_input.get('temperature', 0.7), 1.5)),
        'cfg_weight': max(0.0, min(job_input.get('cfg_weight', 0.4), 1.0)),
        'exaggeration': max(0.0, min(job_input.get('exaggeration', 0.4), 1.0))
    }

def handler(job):
    """Optimized RunPod handler."""
    start_time = time.time()
    
    try:
        logger.info("=== Handler Started ===")
        
        # Quick GPU check
        if not torch.cuda.is_available():
            logger.error("GPU NOT AVAILABLE - Performance will be poor!")
        
        # Initialize model
        model = initialize_model()
        
        # Process input
        job_input = job.get('input', {})
        validated_input = validate_input(job_input)
        
        # Handle audio prompt
        audio_prompt_path = None
        if validated_input['audio_prompt_base64']:
            audio_prompt_path = decode_audio_prompt_base64(
                validated_input['audio_prompt_base64']
            )
        
        # Generate audio
        audio_tensor = text_to_speech_pipeline(
            text=validated_input['text'],
            model=model,
            max_chars_per_chunk=validated_input['max_chars_per_chunk'],
            inter_chunk_silence_ms=validated_input['inter_chunk_silence_ms'],
            audio_prompt_path=audio_prompt_path,
            temperature=validated_input['temperature'],
            cfg_weight=validated_input['cfg_weight'],
            exaggeration=validated_input['exaggeration']
        )
        
        if audio_tensor is None:
            raise RuntimeError("Failed to generate audio")
        
        # Convert to base64
        audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)
        
        # Clean up
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            os.unlink(audio_prompt_path)
        
        # Response
        processing_time = time.time() - start_time
        duration = audio_tensor.shape[1] / model.sr
        
        logger.info(f"Success: {duration:.1f}s audio in {processing_time:.2f}s")
        
        return {
            "status": "success",
            "audio_base64": audio_base64,
            "metadata": {
                "duration_seconds": round(duration, 2),
                "sample_rate": model.sr,
                "processing_time_seconds": round(processing_time, 2),
                "device": device,
                "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
            }
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Handler failed: {e}")
        logger.error(traceback.format_exc())
        
        return {
            "status": "error",
            "error": str(e),
            "processing_time_seconds": round(processing_time, 2)
        }

# Pre-initialize when worker starts
if __name__ == '__main__':
    logger.info("Starting ChatterboxTTS Worker...")
    
    try:
        # Pre-initialize for faster first request
        logger.info("Pre-initializing model...")
        initialize_model()
        logger.info("Ready to serve requests!")
        
        # Start RunPod serverless
        runpod.serverless.start({
            'handler': handler,
            'return_aggregate_stream': False
        })
        
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        raise
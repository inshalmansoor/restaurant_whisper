import sys
from pathlib import Path
from typing import Optional, Tuple, Any
from huggingface_hub import snapshot_download
import whisper
from gtts import gTTS  # only used if you want to optionally warm TTS
import torch

MODEL_MARKERS = [
    "pytorch_model.bin",
    "model.safetensors",
    "tf_model.h5",
    "flax_model.msgpack",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def ensure_model_dir(
    repo_id: str,
    target_dir: Path,
    force: bool = False,
    *,
    whisper_name: Optional[str] = None,
    is_gtts: bool = False,
) -> Tuple[Path, Optional[Any]]:
    """
    Ensure an artifact is available under `target_dir`.

    - For HuggingFace repos (default): snapshot_download(repo_id, cache_dir=target_dir),
      then search the snapshot tree for a model marker and return (model_dir, None).

    - For Whisper: provide whisper_name (e.g. "large"). The function will try to load
      the local model from target_dir/whisper_name if present, otherwise call whisper.load_model
      which downloads into download_root=target_dir. Returns (model_dir, whisper_model).

    - For gTTS: set is_gtts=True (or use repo_id == "gtts"). This just ensures a cache dir and returns (target_dir, None).

    Returns:
      (Path_to_model_dir_or_cache, extra)
      where extra is None for HF/gTTS, or the loaded Whisper model for whisper_name case.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # --- Whisper special-case ---
    if whisper_name:
        whisper_root = target_dir
        whisper_dir = whisper_root / whisper_name
        whisper_root.mkdir(parents=True, exist_ok=True)

        if whisper_dir.exists() and any(whisper_dir.iterdir()):
            print(f"[i] Loading Whisper model from disk: {whisper_dir}")
            # try to load from local dir (many whisper wrappers accept a path)
            model = whisper.load_model(str(whisper_dir))
            return whisper_dir, model
        else:
            print(f"[i] Whisper model '{whisper_name}' not found locally. Downloading into {whisper_root} ...")
            # whisper will download into download_root
            model = whisper.load_model(whisper_name, download_root=str(whisper_root))
            # try to detect local folder if snapshot created it
            if whisper_dir.exists() and any(whisper_dir.iterdir()):
                print(f"[i] Whisper downloaded to: {whisper_dir}")
                return whisper_dir, model
            # fallback: return the root and the model object
            return whisper_root, model

    # --- gTTS special-case (no offline model, just ensure cache dir) ---
    if is_gtts or repo_id.lower() == "gtts":
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[i] Ensured gTTS cache dir: {target_dir}")
        return target_dir, None

    # --- Default: HuggingFace snapshot logic ---
    print(f"[i] Ensuring HF repo {repo_id} is downloaded into {target_dir} (force={force})...")
    try:
        snapshot_path = snapshot_download(repo_id=repo_id, cache_dir=str(target_dir), force_download=force)
    except Exception as e:
        print(f"[!] snapshot_download failed for {repo_id}: {e}", file=sys.stderr)
        raise

    snapshot_path = Path(snapshot_path)
    print(f"[i] snapshot_download returned: {snapshot_path}")

    def has_marker(p: Path) -> bool:
        if not p.is_dir():
            return False
        for m in MODEL_MARKERS:
            if (p / m).exists():
                return True
        # check children one level down
        for child in p.iterdir():
            if child.is_dir():
                for m in MODEL_MARKERS:
                    if (child / m).exists():
                        return True
        return False

    # direct hit
    if has_marker(snapshot_path):
        return snapshot_path, None

    # children
    for child in snapshot_path.iterdir():
        if has_marker(child):
            return child, None

    # deeper search (rglob)
    for p in snapshot_path.rglob("*"):
        if p.is_dir():
            for m in MODEL_MARKERS:
                if (p / m).exists():
                    return p, None

    # nothing found
    raise OSError(
        f"No model files found under snapshot path {snapshot_path}. "
        f"Expected one of: {MODEL_MARKERS}. Inspect the folder to see where the model files live."
    )

# assumes ensure_model_dir is available in the same module or imported
# ensure_model_dir(repo_id: str, target_dir: Path, force: bool=False, *, whisper_name: Optional[str]=None, is_gtts: bool=False)

def prepare_and_load_whisper(model_name: str = "large-v3",
                             target_root: Path | str = Path("models/whisper"),
                             force: bool = False) -> Any:
    """
    Ensure whisper model files exist under target_root and return a loaded whisper model.
    Uses ensure_model_dir(..., whisper_name=model_name) to do the heavy lifting if available.
    Returns the loaded whisper model object.
    """
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    # Call your ensure_model_dir helper. It should return (path, extra)
    # For the whisper case ensure_model_dir(..., whisper_name=model_name) returns (path, model_obj)
    try:
        model_path, maybe_model = ensure_model_dir("whisper", target_root, force, whisper_name=model_name)
    except TypeError:
        # If your ensure_model_dir signature differs, try calling with whisper_name only
        model_path, maybe_model = ensure_model_dir(target_root, whisper_name=model_name)

    # If ensure_model_dir returned a loaded model, return it directly
    if maybe_model is not None:
        return maybe_model

    # Otherwise attempt to load from the discovered path (best-effort)
    try:
        # prefer to load by path if it looks like a model folder
        return whisper.load_model(str(model_path))
    except Exception:
        # fallback: ask whisper to download/load into the root
        return whisper.load_model(model_name, download_root=str(target_root))

def prepare_and_load_whisper_with_gpu(
    model_name: str = "large",
    target_root: Path | str = Path("models/whisper"),
    force: bool = False,
    fallback_smaller: str | None = "base"
):
    """
    Ensure whisper files exist (calls ensure_model_dir) and load the whisper model.
    Use GPU if torch.cuda.is_available(); otherwise CPU. On OOM or load failure, fall back to CPU
    or to `fallback_smaller` model (if provided).
    Returns the loaded whisper model.
    """
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    # Ensure files/dirs exist (your ensure_model_dir should handle whisper_name special-case)
    # If your ensure_model_dir returns (path, model) as in earlier helper, use that.
    try:
        model_path, maybe_model = ensure_model_dir("whisper", target_root, force, whisper_name=model_name)
    except Exception as e:
        print(f"[!] ensure_model_dir failed for whisper {model_name}: {e}")
        # continue — we'll still try to load via whisper.load_model which will download if needed

    # choose device
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"[i] Torch CUDA available: {use_cuda}. Attempting to load whisper on device='{device}'")

    # If ensure_model_dir already returned a loaded model, try to move it to device
    if 'maybe_model' in locals() and maybe_model is not None:
        model = maybe_model
        try:
            model.to(device)
            print(f"[i] Moved existing Whisper model to {device}")
            return model
        except Exception as e:
            print(f"[!] Could not move existing model to {device}: {e} — continuing to (re)load")

    # Try to load on chosen device (prefer whisper API device argument)
    try:
        # whisper.load_model accepts device= on most versions
        try:
            model = whisper.load_model(model_name, device=device, download_root=str(target_root))
        except TypeError:
            # older whisper versions may not accept device param -> load then .to(device)
            model = whisper.load_model(model_name, download_root=str(target_root))
            try:
                model.to(device)
            except Exception:
                pass
        print(f"[✓] Loaded Whisper '{model_name}' on {device}")
        return model

    except RuntimeError as e:
        # common case: CUDA OOM or driver problem
        print(f"[!] RuntimeError loading Whisper on {device}: {e}")

        # if we attempted GPU and failed, try CPU fallback
        if use_cuda:
            print("[i] Falling back to CPU load of the same model...")
            try:
                model = whisper.load_model(model_name, device="cpu", download_root=str(target_root))
                print(f"[✓] Loaded Whisper '{model_name}' on cpu")
                return model
            except Exception as e2:
                print(f"[!] CPU load also failed: {e2}")

        # if provided, try a smaller model as last resort
        if fallback_smaller and fallback_smaller != model_name:
            try:
                print(f"[i] Attempting to load a smaller model '{fallback_smaller}' on CPU")
                model = whisper.load_model(fallback_smaller, device="cpu", download_root=str(target_root))
                print(f"[✓] Loaded smaller Whisper model '{fallback_smaller}' on cpu")
                return model
            except Exception as e3:
                print(f"[!] Failed to load fallback smaller model '{fallback_smaller}': {e3}")

        # final attempt: raise to let caller decide (or load without device param)
        raise

    except Exception as e:
        # catch-all: try CPU
        print(f"[!] Unexpected error loading Whisper on {device}: {e}")
        try:
            model = whisper.load_model(model_name, device="cpu", download_root=str(target_root))
            print(f"[✓] Loaded Whisper '{model_name}' on cpu (after unexpected error)")
            return model
        except Exception as e2:
            print(f"[✗] Final fallback load failed: {e2}")
            raise
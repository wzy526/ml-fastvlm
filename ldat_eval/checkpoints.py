"""Head tensors keep their original format; metadata is a JSON sidecar."""
import json
from pathlib import Path

import torch


def save_head(path, state, *, metadata, **extra):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"head": state, **extra} if any(extra.values()) else state
    torch.save(payload, path)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_head(path, backbone, *, kind, where_layer=None, head_prefix=None, inference=False):
    path = Path(path).expanduser()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    sidecar = path.with_suffix(path.suffix + ".json")
    meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    expected = dict(kind=kind, model_family=backbone.family,
                    hidden_size=backbone.hidden_size, factor=backbone.factor)
    if where_layer is not None:
        expected["where_layer"] = where_layer
    if head_prefix is not None:
        expected["head_prefix"] = head_prefix
    if inference and kind == "ldat":
        expected.update(ldat_in="native", ldat_win="dat", pe="global")
        if payload.get("lora") or payload.get("win_head"):
            raise ValueError("This inference route requires a density-only checkpoint; "
                             "LoRA/window-head weights must not be silently ignored.")
        if payload.get("train_cfg", "ldat") != "ldat":
            raise ValueError("Expected a checkpoint trained with --train_cfg ldat")
    for key, value in expected.items():
        if key in meta and meta[key] != value:
            raise ValueError(f"{path}: {key}={meta[key]!r}, expected {value!r}; "
                             "train a head for this backbone/configuration")
    state = payload["head"] if "head" in payload else payload
    if kind == "where":
        state = {k.removeprefix("where_head."): v for k, v in state.items()}
    width_key = "img_in.0.weight" if kind == "where" else "inp.0.weight"
    if width_key not in state or state[width_key].numel() != backbone.hidden_size:
        raise ValueError(f"{path}: {kind} head input width does not match {backbone.hidden_size}; "
                         "Qwen3.5 head weights cannot be reused on Qwen2.5-VL-7B")
    return state, meta

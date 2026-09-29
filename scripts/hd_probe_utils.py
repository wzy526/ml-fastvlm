"""Image identity and deterministic wrong-image controls shared by HD probes."""

import hashlib


def image_fingerprint(image):
    """Hash the full decoded RGB image, including its geometry.

    Paths and leading pixels are not reliable identities: datasets can repeat
    one image under several questions/files, and documents often share a blank
    header while differing farther down the page.
    """
    rgb = image.convert("RGB")
    digest = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def different_image_indices(fingerprints):
    """Choose a different-content donor for every row, or fail before inference.

    The search is deterministic for an ordered input list and covers every
    candidate. Donors may repeat: a bijection is not always possible when many
    questions share an image. This is a wrong-image control, not a permutation
    of image-question rows with guaranteed one-to-one usage.
    """
    fingerprints = list(fingerprints)
    n = len(fingerprints)
    if len(set(fingerprints)) < 2:
        raise ValueError(
            "shuffle requires at least two different image contents; "
            f"found {len(set(fingerprints))} among {n} samples"
        )
    start = max(1, n // 2)
    return [
        next((i + step) % n for step in range(start, start + n)
             if fingerprints[(i + step) % n] != fingerprint)
        for i, fingerprint in enumerate(fingerprints)
    ]

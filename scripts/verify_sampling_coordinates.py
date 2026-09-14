#!/usr/bin/env python3
"""Verify DAT's normalized (x, y) sampling convention on a rectangular map."""

import argparse

import torch
import torch.nn.functional as F


def reference_grid(height, width):
    margin_y = 1.0 / max(height - 1, 1)
    margin_x = 1.0 / max(width - 1, 1)
    grid_y = torch.linspace(-1.0 + margin_y, 1.0 - margin_y, height)
    grid_x = torch.linspace(-1.0 + margin_x, 1.0 - margin_x, width)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)


def expected_values(height, width):
    grid = reference_grid(height, width)[0]
    x = (grid[..., 0] + 1.0) * 0.5 * (width - 1)
    y = (grid[..., 1] + 1.0) * 0.5 * (height - 1)
    return 10.0 * y + x


def render(source, sampled, wrong, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    panels = (
        (source, "source: value = 10*y + x"),
        (sampled, "fixed: grid_sample(x, y)"),
        (wrong, "old bug: grid_sample(y, x)"),
    )
    for ax, (values, title) in zip(axes, panels):
        image = ax.imshow(values.numpy(), cmap="viridis", vmin=0, vmax=24,
                          interpolation="nearest", aspect="equal")
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                ax.text(x, y, f"{values[y, x]:.1f}", ha="center", va="center",
                        color="white" if values[y, x] < 16 else "black", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("x / width")
        ax.set_ylabel("y / height")
    fig.colorbar(image, ax=axes, shrink=0.8)
    fig.savefig(output, dpi=160)
    print(f"saved visualization: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None,
                        help="optional comparison PNG showing fixed vs swapped sampling")
    args = parser.parse_args()

    height, width = 3, 5
    source = (
        10.0 * torch.arange(height)[:, None] + torch.arange(width)[None, :]
    ).unsqueeze(0).unsqueeze(0)
    grid = reference_grid(height, width)

    sampled = F.grid_sample(
        source.float(), grid, mode="bilinear", align_corners=True,
    )[0, 0]
    expected = expected_values(height, width)
    wrong = F.grid_sample(
        source.float(), grid[..., (1, 0)], mode="bilinear", align_corners=True,
    )[0, 0]

    torch.testing.assert_close(sampled, expected)
    assert not torch.allclose(wrong, expected)

    print("PASS: direct (x, y) grid matches the analytical rectangular-map result")
    print("source:")
    print(source[0, 0])
    print("sampled:")
    print(sampled)
    print("old swapped result (must differ):")
    print(wrong)

    if args.output:
        render(source[0, 0], sampled, wrong, args.output)


if __name__ == "__main__":
    main()

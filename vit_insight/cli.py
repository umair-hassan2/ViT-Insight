from __future__ import annotations

import argparse

from PIL import Image

from vit_insight import Registry, load_model, run_forward
from vit_insight.viz import per_layer_frames, rollout_overlay, save_gif


def main(argv=None):
    p = argparse.ArgumentParser(prog="vit-insight")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models", help="list supported models")

    e = sub.add_parser("explain", help="save a rollout overlay (and optional per-layer GIF)")
    e.add_argument("--model", required=True)
    e.add_argument("--image", required=True)
    e.add_argument("--labels", help="comma-separated text labels (CLIP/SigLIP)")
    e.add_argument("--out", default="rollout.png")
    e.add_argument("--gif", help="also write per-layer GIF to this path")
    e.add_argument("--head-fusion", default="mean", choices=["mean", "max", "min"])
    e.add_argument("--discard-ratio", type=float, default=0.0)
    e.add_argument("--no-residual", action="store_true")
    e.add_argument("--device")
    args = p.parse_args(argv)

    reg = Registry()
    if args.cmd == "models":
        for s in reg:
            print(f"{s.id:50s} {s.objective:28s} prefix={s.prefix_tokens} patch={s.patch_size}")
        return

    lm = load_model(args.model, args.device)
    labels = [x.strip() for x in args.labels.split(",")] if args.labels else None
    res = run_forward(lm, Image.open(args.image).convert("RGB"), labels)
    rollout_overlay(
        res,
        lm.spec,
        head_fusion=args.head_fusion,
        discard_ratio=args.discard_ratio,
        residual=not args.no_residual,
    ).save(args.out)
    print(f"wrote {args.out}")
    if args.gif:
        save_gif(per_layer_frames(res, lm.spec, head_fusion=args.head_fusion), args.gif)
        print(f"wrote {args.gif}")
    if res.pred is not None:
        label = labels[res.pred] if labels else lm.model.config.id2label.get(res.pred, res.pred)
        print(f"prediction: {label} (p={res.pred_prob:.3f})")


if __name__ == "__main__":
    main()

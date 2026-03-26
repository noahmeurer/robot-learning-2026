"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, VALID_BACKBONES, build_policy, get_policy_checkpoint_config

# TODO: Any imports you want from torch or other libraries we use. Not allowed: libraries we don't use
from torch.utils.data import DataLoader, random_split

# TODO: Choose your own hyperparameters!
EPOCHS = 200
BATCH_SIZE = 512
LR = 1e-4
VAL_SPLIT = 0.1
def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        # TODO: Implement the training step for one batch here.
        # This mostly: Get states and action_chunks onto the correct device, compute the loss, and step the optimizer.
        # Move data to device
        states = states.to(device)
        action_chunks = action_chunks.to(device)

        # Compute loss
        loss = model.compute_loss(states, action_chunks)
        
        # Compute gradients and step optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update total loss and number of batches
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        # TODO: Implement the evaluation step for one batch here.
        # Move data to device
        states = states.to(device)
        action_chunks = action_chunks.to(device)

        # Compute loss
        loss = model.compute_loss(states, action_chunks)

        # Update total loss and number of batches
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    # TODO: You may add any cli arguments that make life easier for you like learning rate etc.
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help=(
            "Path to a JSON task config file. If provided, config values override CLI values "
            "(for supported fields)."
        ),
    )
    parser.add_argument(
        "--zarr", type=Path, default=None, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--extra-zarr",
        type=Path,
        nargs="*",
        default=None,
        help="Optional additional processed .zarr stores to merge in (e.g. for DAgger).",
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=f"Learning rate (default: module LR={LR}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: module EPOCHS={EPOCHS}).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help=(
            f"Model backbone to use. Supported: {tuple(sorted(VALID_BACKBONES))}. "
            "If provided in --task-config, it overrides CLI; defaults to 'mlp'."
        ),
    )
    parser.add_argument(
        "--backbone-kwargs",
        type=str,
        default=None,
        help=(
            "Optional JSON object with per-backbone kwargs, e.g. "
            '\'{"hidden_dim": 768, "depth": 6}\'.'
        ),
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help=(
            "Optional checkpoint name stem (e.g. 'ex1'). "
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional path to an existing checkpoint to resume from. "
            "Loads model weights and continues for additional epochs."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    cli_save_name = args.save_name

    # ── load config (optional) ─────────────────────────────────────────
    if args.task_config is not None:
        cfg_path = Path(f"configs/{args.task_config}.json")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise TypeError(f"Config must be a JSON object (dict), got: {type(cfg)}")

        # Config-first: if a field is present in config, it overrides CLI.
        if "zarr" in cfg and cfg["zarr"] is not None:
            args.zarr = Path(cfg["zarr"])
        if "extra_zarr" in cfg and cfg["extra_zarr"] is not None:
            args.extra_zarr = [Path(p) for p in cfg["extra_zarr"]]
        if "policy" in cfg and cfg["policy"] is not None:
            args.policy = str(cfg["policy"])
        if "chunk_size" in cfg and cfg["chunk_size"] is not None:
            args.chunk_size = int(cfg["chunk_size"])
        if "state_keys" in cfg and cfg["state_keys"] is not None:
            args.state_keys = cfg["state_keys"]
        if "action_keys" in cfg and cfg["action_keys"] is not None:
            args.action_keys = cfg["action_keys"]
        if "lr" in cfg and cfg["lr"] is not None:
            args.lr = float(cfg["lr"])
        if "epochs" in cfg and cfg["epochs"] is not None:
            args.epochs = int(cfg["epochs"])
        if "backbone" in cfg and cfg["backbone"] is not None:
            args.backbone = str(cfg["backbone"])
        if "backbone_kwargs" in cfg and cfg["backbone_kwargs"] is not None:
            args.backbone_kwargs = cfg["backbone_kwargs"]
        if "save_name" in cfg and cfg["save_name"] is not None:
            args.save_name = str(cfg["save_name"])
        if (
            "resume_from_checkpoint" in cfg
            and cfg["resume_from_checkpoint"] is not None
        ):
            args.resume_from_checkpoint = Path(cfg["resume_from_checkpoint"])
        if "seed" in cfg and cfg["seed"] is not None:
            args.seed = int(cfg["seed"])

    # Exception to config-first behavior: an explicitly provided CLI --save-name
    # should take precedence over config save_name.
    if cli_save_name is not None:
        args.save_name = cli_save_name

    # Default backbone if neither CLI nor config specified one.
    if args.backbone is None:
        args.backbone = "mlp"
    if args.backbone not in VALID_BACKBONES:
        raise SystemExit(
            f"Invalid backbone '{args.backbone}'. Supported backbones: {tuple(sorted(VALID_BACKBONES))}"
        )
    if args.backbone_kwargs is None:
        backbone_kwargs: dict[str, object] = {}
    elif isinstance(args.backbone_kwargs, str):
        try:
            parsed = json.loads(args.backbone_kwargs)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON for --backbone-kwargs: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SystemExit("--backbone-kwargs must decode to a JSON object.")
        backbone_kwargs = parsed
    elif isinstance(args.backbone_kwargs, dict):
        backbone_kwargs = args.backbone_kwargs
    else:
        raise SystemExit(
            f"Unsupported backbone_kwargs type: {type(args.backbone_kwargs)}. "
            "Expected dict or JSON string."
        )
    if args.zarr is None:
        raise SystemExit(
            "You must specify --zarr, either directly on the CLI or via --task-config "
            "(JSON field 'zarr')."
        )
    num_epochs = args.epochs if args.epochs is not None else EPOCHS
    if num_epochs <= 0:
        raise SystemExit("--epochs must be a positive integer.")
    if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.exists():
        raise SystemExit(
            f"--resume-from-checkpoint file not found: {args.resume_from_checkpoint}"
        )

    # Require explicit state/action specification via CLI or config.
    # (We intentionally do NOT fall back to the dataset's default single key here.)
    if args.state_keys is None or args.action_keys is None:
        raise SystemExit(
            "You must specify both --state-keys and --action-keys, either directly on the CLI "
            "or via --task-config (with JSON fields 'state_keys' and 'action_keys')."
        )

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    normalizer = Normalizer.from_data(states, actions)

    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=args.chunk_size,
        normalizer=normalizer,
    )
    print(f"Dataset: {len(dataset)} samples, chunk_size={args.chunk_size}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    # ── train / val split ─────────────────────────────────────────────
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        backbone=args.backbone,
        **backbone_kwargs,
        # TODO: build with your desired specifications
    ).to(device)
    resolved_policy_cfg = get_policy_checkpoint_config(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # TODO: implement an optimizer and scheduler
    lr = args.lr if args.lr is not None else LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=lr * 0.01,  # anneal close to (but not all the way to) zero
    )

    # ── training loop ─────────────────────────────────────────────────
    start_epoch = 0
    best_val = float("inf")
    resumed_ckpt: Path | None = None

    if args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in ckpt:
            raise SystemExit(
                "Resume checkpoint is missing required key 'model_state_dict'."
            )

        # Compatibility checks for safe resume.
        ckpt_policy_type = ckpt.get("policy_type")
        if ckpt_policy_type is None:
            print("WARNING: resume checkpoint has no 'policy_type'; skipping this check.")
        elif str(ckpt_policy_type) != args.policy:
            raise SystemExit(
                "Resume checkpoint policy mismatch: "
                f"checkpoint={ckpt_policy_type!r}, current={args.policy!r}."
            )

        ckpt_state_dim = ckpt.get("state_dim")
        if ckpt_state_dim is None:
            print("WARNING: resume checkpoint has no 'state_dim'; skipping this check.")
        elif int(ckpt_state_dim) != int(states.shape[1]):
            raise SystemExit(
                "Resume checkpoint state_dim mismatch: "
                f"checkpoint={int(ckpt_state_dim)}, current={int(states.shape[1])}."
            )

        ckpt_action_dim = ckpt.get("action_dim")
        if ckpt_action_dim is None:
            print("WARNING: resume checkpoint has no 'action_dim'; skipping this check.")
        elif int(ckpt_action_dim) != int(actions.shape[1]):
            raise SystemExit(
                "Resume checkpoint action_dim mismatch: "
                f"checkpoint={int(ckpt_action_dim)}, current={int(actions.shape[1])}."
            )

        ckpt_chunk_size = ckpt.get("chunk_size")
        if ckpt_chunk_size is None:
            print("WARNING: resume checkpoint has no 'chunk_size'; skipping this check.")
        elif int(ckpt_chunk_size) != int(args.chunk_size):
            raise SystemExit(
                "Resume checkpoint chunk_size mismatch: "
                f"checkpoint={int(ckpt_chunk_size)}, current={int(args.chunk_size)}."
            )

        ckpt_state_keys = ckpt.get("state_keys")
        if ckpt_state_keys is None:
            print("WARNING: resume checkpoint has no 'state_keys'; skipping this check.")
        elif list(ckpt_state_keys) != list(args.state_keys):
            raise SystemExit(
                "Resume checkpoint state_keys mismatch between checkpoint and current run."
            )

        ckpt_action_keys = ckpt.get("action_keys")
        if ckpt_action_keys is None:
            print("WARNING: resume checkpoint has no 'action_keys'; skipping this check.")
        elif list(ckpt_action_keys) != list(args.action_keys):
            raise SystemExit(
                "Resume checkpoint action_keys mismatch between checkpoint and current run."
            )

        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        if "epoch" not in ckpt:
            print("WARNING: resume checkpoint has no 'epoch'; starting from epoch 0.")
        best_val = float(ckpt.get("val_loss", float("inf")))
        if "val_loss" not in ckpt:
            print(
                "WARNING: resume checkpoint has no 'val_loss'; best_val initialized to inf."
            )
        resumed_ckpt = ckpt_path

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    if args.save_name is None:
        save_stem = f"{action_space}_{args.policy}"
    else:
        custom = args.save_name.strip()
        if not custom:
            raise SystemExit("--save-name must be a non-empty string when provided.")
        if custom.endswith(".pt"):
            custom = custom[:-3]
        if custom.startswith("best_model_"):
            custom = custom[len("best_model_") :]
        if not custom:
            raise SystemExit(
                "--save-name must include a non-empty name after 'best_model_'."
            )
        save_stem = custom

    save_name = f"best_model_{save_stem}.pt"

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        save_name = f"best_model_{save_stem}_dagger{n_dagger_eps}ep.pt"
    # Default: checkpoints/<task>/
    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")
    save_path = ckpt_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Single consolidated run summary.
    print("\nResolved training configuration:")
    print(f"  device={device}")
    print(f"  zarr={args.zarr}")
    print(
        f"  extra_zarr={[str(p) for p in args.extra_zarr] if args.extra_zarr else []}"
    )
    print(f"  policy={args.policy}")
    print(f"  backbone={args.backbone}")
    print(f"  backbone_kwargs_input={backbone_kwargs}")
    if "backbone" in resolved_policy_cfg and "backbone_kwargs" in resolved_policy_cfg:
        print(f"  backbone_resolved={resolved_policy_cfg['backbone']}")
        print(
            f"  backbone_kwargs_resolved={resolved_policy_cfg['backbone_kwargs']}"
        )
    print(f"  chunk_size={args.chunk_size}")
    print(f"  state_keys={args.state_keys}")
    print(f"  action_keys={args.action_keys}")
    print(f"  state_dim={states.shape[1]}")
    print(f"  action_dim={actions.shape[1]}")
    print(f"  dataset_size={len(dataset)}")
    print(f"  train_size={n_train}")
    print(f"  val_size={n_val}")
    print(f"  batch_size={BATCH_SIZE}")
    print(f"  epochs={num_epochs}")
    print(f"  lr={lr}")
    print(f"  seed={args.seed}")
    print(f"  model_parameters={n_params:,}")
    print(f"  checkpoint_path={save_path}")
    print(f"  resume_from_checkpoint={resumed_ckpt}")
    print(f"  resume_start_epoch={start_epoch}")
    print(f"  initial_best_val={best_val}")

    final_epoch = start_epoch + num_epochs
    for epoch in range(start_epoch + 1, final_epoch + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            policy_ckpt_cfg = get_policy_checkpoint_config(model)
            if "backbone" not in policy_ckpt_cfg or "backbone_kwargs" not in policy_ckpt_cfg:
                policy_ckpt_cfg = {
                    "backbone": args.backbone,
                    "backbone_kwargs": dict(backbone_kwargs),
                }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer": {
                        "state_mean": normalizer.state_mean,
                        "state_std": normalizer.state_std,
                        "action_mean": normalizer.action_mean,
                        "action_std": normalizer.action_std,
                    },
                    "chunk_size": args.chunk_size,
                    "policy_type": args.policy,
                    "state_keys": args.state_keys,
                    "action_keys": args.action_keys,
                    "state_dim": int(states.shape[1]),
                    "action_dim": int(actions.shape[1]),
                    "val_loss": val_loss,
                    **policy_ckpt_cfg,
                },
                save_path,
            )
            tag = " ✓ saved"

        print(
            f"Epoch {epoch:3d}/{final_epoch} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}{tag}"
        )

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()

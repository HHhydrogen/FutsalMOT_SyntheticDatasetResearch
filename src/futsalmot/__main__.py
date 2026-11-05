"""默认inspect；所有准备命令均不加载Torch/模型、训练或读取GT图像。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .data import DEFAULT_REAL, CycleSampler, detection_train_uids, inspect_dataset, load_config
from .training import plan, train
from .evaluation import evaluate

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="futsalmot")
    parser.add_argument("command", choices=("inspect", "dry-run", "train", "evaluate"), nargs="?", default="inspect")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/experiment.yaml")
    parser.add_argument("--dataset", choices=("soccernet", "sportsmot", "all"), default=None, help="inspect默认all；dry-run默认配置来源，显式all不允许混合")
    parser.add_argument("--task", choices=("detect", "pose", "segment"))
    parser.add_argument("--real-images", type=int)
    parser.add_argument("--synthetic-images", type=int)
    parser.add_argument("--real-epochs", type=int, choices=(50, 100))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "inspect":
            names = list(DEFAULT_REAL) if args.dataset in (None, "all") else [args.dataset]
            paths = config.get("real_datasets", DEFAULT_REAL)
            result = {"status": "OK", "datasets": {name: inspect_dataset(paths[name]) for name in names}, "models_loaded": False, "GPU_used": False}
        elif args.command == "dry-run":
            task = args.task or config["task"]
            if args.dataset == "all":
                raise ValueError("dry-run不允许跨库混合；请显式选择soccernet或sportsmot来源内面板。")
            configured_panel = config["candidate_budget"]["real_panel"]
            panel = args.dataset or configured_panel
            if panel not in DEFAULT_REAL:
                raise ValueError("未批准的真实来源面板")
            configured_n_r = config["candidate_budget"].get("N_R") if panel == configured_panel else None
            n_r = args.real_images if args.real_images is not None else configured_n_r
            if n_r is None and task == "detect":
                n_r = inspect_dataset(config["real_datasets"][panel])["counts"]["train"]["detection_images"]
            n_s = args.synthetic_images if args.synthetic_images is not None else config["candidate_budget"].get("N_S")
            hypothetical = args.real_images is not None or args.synthetic_images is not None
            if n_s is None or n_r is None:
                result = {"status": "AWAITING_DATA_COUNTS", "task": task, "N_S": n_s, "N_R": n_r, "reason": "不假造未冻结的合成整图池或缺GT任务真实整图数；可用显式--*-images作假设算例。", "training_executed": False}
            else:
                result = {"status": "CANDIDATE_PREVIEW_ONLY", "task": task, "hypothetical_counts": hypothetical, "real_panel": panel, **plan(n_s, n_r, epochs_s=config["candidate_budget"]["epochs_S"], epochs_r=args.real_epochs or config["candidate_budget"]["epochs_R"], batch=config["numeric"]["effective_batch"])}
                result["UID_binding"] = {"S": "pending_real_synthetic_manifest_no_UIDs_fabricated", "R": "hypothetical_count_only"}
                if task == "detect" and args.real_images is None:
                    uids = detection_train_uids(config["real_datasets"][panel])
                    if len(uids) != n_r:
                        raise ValueError("候选N_R与真实train UID数不一致")
                    sampler = CycleSampler(uids, task=task, domain="R", seed=config["seed"])
                    result["UID_binding"]["R"] = {"unique_train_samples": len(uids), "pool_sha256": sampler.pool_sha, "first_effective_batch": sampler.take(config["numeric"]["effective_batch"])}
        elif args.command == "train":
            result = train(config)
        else:
            result = evaluate(config, split=args.split)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if result["status"] in ("BLOCKED", "AWAITING_DATA_COUNTS") else 0
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "ERROR", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

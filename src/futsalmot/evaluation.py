"""显式评价门禁；真实指标计算尚未实现，不静默输出伪分数。"""
from .training import training_blockers

def evaluate(config: dict, *, split: str = "val") -> dict:
    if split not in ("val", "test"): raise ValueError("评价split只能是val或test")
    blockers = training_blockers(config)
    if split == "test" and config.get("gate", {}).get("test_open_authorized") is not True:
        blockers.append("test未获统一开启授权")
    blockers.append("检测/Pose/Mask/MOT评价接口与ignore规则尚未实现并核验")
    return {"status": "BLOCKED", "command": "evaluate", "split": split, "blockers": blockers, "sample_content_read": False, "model_loaded": False, "GPU_used": False, "metrics": None}

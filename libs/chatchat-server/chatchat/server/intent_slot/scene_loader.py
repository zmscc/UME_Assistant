# scene_loader.py
import json
import os
from typing import Dict, Any


def load_scene_templates() -> Dict[str, Any]:
    """
    从 scene_templates.json 文件加载场景配置。
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, "scene_templates.json")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
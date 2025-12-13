import re
import os
import json
import json5
import logging
import requests
from typing import Dict, List, Optional, Any
import hashlib
from fastapi import Request


# from . import config  # 假设 config.py 存在


def extract_float(s: str) -> float:
    """提取字符串中的第一个浮点数，若无则返回 0.0"""
    match = re.search(r'-?\d+(?:\.\d+)?', s)
    return float(match.group()) if match else 0.0


def extract_continuous_digits(text: str) -> List[str]:
    """提取所有连续数字字符串（如 ['123', '45']）"""
    return re.findall(r'\d+', text)


def extract_first_json(input_string: str) -> Optional[Dict[str, Any]]:
    # 先清理代码块
    cleaned = re.sub(r'^```(?:json)?\s*', '', input_string.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)

    try:
        obj = json5.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def get_raw_slot(parameters: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """根据参数模板初始化槽位字典，值为 None"""
    return {p["name"]: None for p in parameters}


def is_slot_fully_filled(slot: Dict[str, Optional[str]]) -> bool:
    """检查所有槽位是否已填充（非 None 且非空字符串）"""
    for value in slot.values():
        if value is None or str(value).strip() == "":
            return False
    return True


def update_slot(new_values: Dict[str, Any], slot: Dict[str, Optional[str]]) -> None:
    """用 new_values 更新 slot，仅当值非空时更新"""
    for key, value in new_values.items():
        if key in slot and value not in (None, "", "null", "None"):
            slot[key] = str(value).strip()


def format_name_value_for_logging(slot: Dict[str, Optional[str]]) -> str:
    """格式化槽位用于日志打印"""
    lines = []
    for name, value in slot.items():
        lines.append(f"name: {name}, Value: {value}")
    return "\n".join(lines)


def send_message(prompt: str, user_input: Optional[str] = None) -> Optional[str]:
    """
    向大模型发送请求
    - prompt: 实际发给模型的用户消息内容（可能是完整指令）
    - user_input: 仅用于日志打印（可选）

    配置完全写死，适配 DashScope / OpenAI 兼容 API
    """
    # ====================【写死配置】====================
    API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    API_KEY = "sk-ce8d16460fb24dcd93a69e400a7cbb1f"
    MODEL = "qwen-max"
    SYSTEM_PROMPT = "你是一个客服助手。"
    # ================================================

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # 构造 messages：system（可选） + user（= prompt）
    messages = []
    if SYSTEM_PROMPT.strip():
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})  # ← 关键：prompt 作为 user 内容

    data = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512
    }

    # ===== 日志打印=====
    print('--------------------------------------------------------------------')
    if False:  # 假设 DEBUG=False，走 else 分支
        pass
    elif user_input:
        print('用户输入:', user_input)
    else:
        print('Prompt:', prompt[:200] + ('...' if len(prompt) > 200 else ''))
    print('--------------------------------------------------------------------')

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=data,
            timeout=30,
            verify=False
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
        print('LLM输出:', answer)
        print('--------------------------------------------------------------------')
        return answer.strip()

    except Exception as e:
        print(f"❌ LLM 调用失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print("错误详情:", e.response.text)
        return None


def load_scene_templates(file_path: str) -> Dict[str, Any]:
    """加载单个场景模板 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_all_scene_configs() -> Dict[str, Dict[str, Any]]:
    """加载与本模块同目录下的 scene_templates.json 中的所有场景配置"""
    # 获取当前模块所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_file = os.path.join(current_dir, "scene_templates.json")

    all_configs: Dict[str, Dict[str, Any]] = {}

    if not os.path.exists(template_file):
        logging.error(f"未找到配置文件: {template_file}")
        return all_configs

    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            current = json.load(f)

        for key, value in current.items():
            if key in all_configs:
                logging.warning(f"场景键 '{key}' 已存在，跳过重复定义")
            else:
                all_configs[key] = value

    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")

    return all_configs


def filename_to_classname(filename: str) -> str:
    """snake_case 转 CamelCase"""
    return ''.join(part.capitalize() for part in filename.split('_'))


def build_system_prompt_for_scenes(scenes: Dict[str, Dict[str, Any]]) -> str:
    prompt_parts = [
        "你是一个中国南方航空智能客服助手。请严格按以下规则响应：",
        "1. 用户输入可能涉及多个服务场景，请判断最匹配的一个。",
        "2. 仅从以下预定义场景中选择，不要编造新场景。",
        "3. 输出必须是合法 JSON，格式为：{'scene': '场景名', 'parameters': {...}}",
        "4. parameters 中只包含该场景定义的参数，未提及的留空或省略。",
        "5. 不要输出任何解释、Markdown 或额外文本。\n"
    ]

    for scene_name, cfg in scenes.items():
        params_desc = []
        for p in cfg.get("parameters", []):
            req = "（必填）" if p.get("required", False) else "（选填）"
            params_desc.append(f"- {p['name']}: {p['desc']} {req}")

        example = ""
        if cfg.get("example"):
            parts = cfg["example"].split("\n答：")
            if len(parts) > 1:
                example = parts[1].strip()

        prompt_parts.append(f"【场景名称】{scene_name}")
        prompt_parts.append(f"【中文名】{cfg.get('name', scene_name)}")
        prompt_parts.append(f"【描述】{cfg.get('description', '')}")
        prompt_parts.append("【参数】\n" + "\n".join(params_desc))
        if example:
            prompt_parts.append(f"【示例输出】{example}")
        prompt_parts.append("")

    prompt_parts.append("现在请处理用户输入：")
    return "\n".join(prompt_parts)


def _get_client_fingerprint(request: Request) -> str:
    """生成客户端唯一指纹（IP + User-Agent）"""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")
    raw = f"{client_ip}:{user_agent}"
    return hashlib.md5(raw.encode()).hexdigest()


# ==========================
# 以下为可选保留函数（按需启用）
# ==========================

# def extract_floats(s: str) -> List[float]:
#     """提取所有浮点数"""
#     return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', s)] or [0.0]

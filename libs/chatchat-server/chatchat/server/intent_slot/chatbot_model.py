from typing import Dict, List, Optional, Any, Tuple
from .utils import (
    send_message,
    extract_first_json,
    extract_float,
    load_all_scene_configs,
    build_system_prompt_for_scenes
)

RELATED_INTENT_THRESHOLD = 0.6


# ===== 新增：构建 system prompt（需加入 utils.py）=====
def build_system_prompt_for_scenes(scenes: Dict[str, Dict[str, Any]]) -> str:
    prompt_parts = [
        "你是一个中国南方航空智能客服助手。请严格按以下规则响应：",
        "1. 用户输入可能涉及多个服务场景，请判断最匹配的一个。",
        "2. 仅从以下预定义场景中选择，不要编造新场景。",
        '3. 输出必须是合法 JSON，格式为：{"scene": "场景名", "parameters": {...}}',
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


class SceneProcessor:
    def process(self, user_input: str, context: Any) -> str:
        raise NotImplementedError


class CommonProcessor(SceneProcessor):
    def __init__(self, scene_config: Dict):
        self.scene_config = scene_config
        self.scene_name = scene_config["name"]
        self.slot = {p["name"]: None for p in scene_config["parameters"]}

    def process(self, user_input: str, context: Any) -> str:
        if self.scene_name in ["问候处理", "其它查询"]:
            return ""

        params_desc = "\n".join([
            f"{p['name']} ({p['type']}): {p['desc']}" +
            (" [必填]" if p.get("required", False) else " [选填]")
            for p in self.scene_config["parameters"]
        ])
        prompt = (
            f"你是一个信息抽取助手。请从用户输入中提取以下参数，并以纯 JSON 格式返回（只包含提到的字段）：\n"
            f"{params_desc}\n\n"
            f"用户输入：{user_input}\n"
            f"只返回 JSON，不要任何解释、Markdown 或额外文本。"
        )

        raw_resp = send_message(prompt, user_input)
        new_values = {}
        if raw_resp:
            json_obj = extract_first_json(raw_resp)
            if json_obj and isinstance(json_obj, dict):
                new_values = json_obj

        for key, value in new_values.items():
            if key in self.slot and value not in (None, "", "null", "None"):
                self.slot[key] = str(value).strip()

        fully_filled = True
        for p in self.scene_config["parameters"]:
            if p.get("required", False):
                val = self.slot.get(p["name"])
                if val is None or str(val).strip() == "":
                    fully_filled = False
                    break

        if fully_filled:
            return self._build_final_query()
        else:
            return self._ask_missing()

    def _build_final_query(self) -> str:
        parts = []
        for p in self.scene_config["parameters"]:
            name = p["name"]
            desc = p["desc"].split("，")[0]
            value = self.slot.get(name)
            if value is not None:
                parts.append(f"{desc}为 {value}")
        return "，".join(parts) + "，请问相关规定是什么？"
    # def _build_final_query(self) -> str:
    #     # 先构造结构化信息（用于 prompt）
    #     context_parts = []
    #     for p in self.scene_config["parameters"]:
    #         name = p["name"]
    #         desc = p["desc"].split("，")[0]
    #         value = self.slot.get(name)
    #         if value is not None:
    #             context_parts.append(f"{desc}：{value}")
    #
    #     scene_name = self.scene_config.get("name", "当前服务")
    #     context_str = "；".join(context_parts)
    #
    #     # 构造 prompt 让 LLM 生成自然问句
    #     prompt = (
    #         f"你是一个航空客服助手，请根据以下用户意图和参数，生成一句自然、完整、适合搜索知识库的问题。\n"
    #         f"场景：{scene_name}\n"
    #         f"参数：{context_str}\n"
    #         f"要求：\n"
    #         f"- 问题要口语化、流畅\n"
    #         f"- 包含所有关键信息\n"
    #         f"- 以问号结尾\n"
    #         f"- 不要包含“根据以上信息”等冗余表述\n"
    #         f"生成的问题："
    #     )
    #
    #     from .utils import send_message  # 假设你的 LLM 调用函数在这里
    #     try:
    #         natural_query = send_message(prompt, "")  # user_input 为空，因为全在 prompt 里
    #         # 简单清洗：去掉可能的前缀（如“问题：”）
    #         natural_query = natural_query.strip().lstrip("问题：").strip()
    #         if not natural_query.endswith(("?", "？")):
    #             natural_query += "？"
    #         return natural_query
    #     except Exception as e:
    #         # fallback：如果 LLM 调用失败，回退到原始拼接
    #         parts = []
    #         for p in self.scene_config["parameters"]:
    #             name = p["name"]
    #             desc = p["desc"].split("，")[0]
    #             value = self.slot.get(name)
    #             if value is not None:
    #                 parts.append(f"{desc}为 {value}")
    #         return "，".join(parts) + "，请问相关规定是什么？"

    def _ask_missing(self) -> str:
        missing = []
        for p in self.scene_config["parameters"]:
            if p.get("required", False) and self.slot.get(p["name"]) in (None, ""):
                missing.append(p["desc"])
        return "，".join(missing) + "？" if missing else "请补充更多信息。"


class ChatbotModel:
    def __init__(self, scene_templates: Dict[str, Dict]):
        self.scene_templates = scene_templates
        self.current_purpose: str = ""
        self.processors: Dict[str, CommonProcessor] = {}
        self._system_prompt = build_system_prompt_for_scenes(scene_templates)

    def recognize_intent_and_extract_slots(self, user_input: str) -> Tuple[Optional[str], Dict]:
        full_prompt = self._system_prompt + "\n用户输入：" + user_input
        raw_resp = send_message(full_prompt, user_input)
        if not raw_resp:
            return None, {}

        json_obj = extract_first_json(raw_resp)
        if not json_obj or not isinstance(json_obj, dict):
            return None, {}

        scene = json_obj.get("scene")
        params = json_obj.get("parameters", {})
        if scene and scene in self.scene_templates:
            return scene, params
        return None, {}

    def is_related_to_last_intent(self, user_input: str, history: List[Dict[str, str]]) -> bool:
        if not self.current_purpose:
            return False

        history_text = "\n".join(
            f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}"
            for msg in history[-6:]
        )

        prompt = (
            "你是一个对话状态跟踪器，请判断用户当前输入是否仍然属于当前任务场景。\n\n"
            f"当前任务场景: {self.scene_templates[self.current_purpose]['description']}\n\n"
            f"最近对话历史:\n{history_text}\n\n"
            f"用户最新输入: {user_input}\n\n"
            "请仅根据以上信息，判断最新输入是否与当前任务相关。\n"
            "输出一个0.0到1.0之间的浮点数，1.0表示完全相关，0.0表示完全无关。"
        )
        result = send_message(prompt, None)
        score = extract_float(result) if result else 0.0
        return score > RELATED_INTENT_THRESHOLD

    def process_multi_question(self, user_input: str, conversation_id: str) -> str:
        from .session_manager import get_conversation_history
        history = get_conversation_history(conversation_id)

        if self.current_purpose and self.is_related_to_last_intent(user_input, history):
            processor = self.get_processor_for_scene(self.current_purpose)
            return processor.process(user_input, None)
        else:
            # 检测到不相关输入，重置当前意图和槽位状态（不清 session）
            if self.current_purpose:
                # 放弃当前意图，清空状态，为新意图做准备
                self.current_purpose = ""
                self.processors.clear()

            scene, params = self.recognize_intent_and_extract_slots(user_input)
            # ===== 特殊场景短路处理 =====
            if scene == "greeting":
                # 问候语：标记为最终响应
                self.current_purpose = "greeting"  # 👈 关键：设置当前意图
                return "【FINAL】您好！我是南航智能客服，请问有什么可以帮您？"
            elif scene == "other_scenario":
                # 兜底场景：返回空字符串，让上层走 RAG
                self.current_purpose = ""  # 显式清空意图
                return user_input
            # ===== 结束新增 =====
            if scene:
                self.current_purpose = scene
                processor = self.get_processor_for_scene(scene)
                # 注入已提取的参数
                for k, v in params.items():
                    if k in processor.slot and v not in (None, "", "null", "None"):
                        processor.slot[k] = str(v).strip()
                # 检查是否完成
                fully_filled = True
                for p in processor.scene_config["parameters"]:
                    if p.get("required", False):
                        val = processor.slot.get(p["name"])
                        if val is None or str(val).strip() == "":
                            fully_filled = False
                            break
                if fully_filled:
                    return processor._build_final_query()
                else:
                    return processor._ask_missing()
            else:
                self.current_purpose = ""
                return ""

    def get_processor_for_scene(self, scene_name: str) -> CommonProcessor:
        if scene_name not in self.processors:
            config = self.scene_templates[scene_name]
            self.processors[scene_name] = CommonProcessor(config)
        return self.processors[scene_name]

    def to_dict(self) -> dict:
        return {
            "current_purpose": self.current_purpose,
            "processors": {
                name: {"slot": proc.slot}
                for name, proc in self.processors.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict, scene_templates: dict) -> "ChatbotModel":
        instance = cls(scene_templates)
        instance.current_purpose = data["current_purpose"]
        for name, proc_data in data["processors"].items():
            proc = CommonProcessor(scene_templates[name])
            proc.slot = proc_data["slot"]
            instance.processors[name] = proc
        return instance

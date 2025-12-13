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
            return self._generate_natural_query_with_llm()
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

    def _ask_missing(self) -> str:
        missing = []
        for p in self.scene_config["parameters"]:
            if p.get("required", False) and self.slot.get(p["name"]) in (None, ""):
                missing.append(p["desc"])
        return "，".join(missing) + "？" if missing else "请补充更多信息。"

    def _generate_natural_query_with_llm(self) -> str:
        """
        使用 LLM 将槽位信息转换为自然语言问题。
        """
        slot_desc = []
        for p in self.scene_config["parameters"]:
            name = p["name"]
            desc = p["desc"].split("，")[0]
            value = self.slot.get(name)
            if value is not None and str(value).strip():
                slot_desc.append(f"{desc}是{value}")

        if not slot_desc:
            return "请问相关规定是什么？"

        prompt = (
                "你是一个航空客服助手，请将以下结构化信息转换成一个自然、流畅、完整的中文问题，"
                "用于向知识库查询相关政策。不要添加解释或额外内容，只输出问题。\n\n"
                "信息：" + "；".join(slot_desc) + "\n\n"
                                              "问题："
        )
        llm_resp = send_message(prompt, None)
        if llm_resp and llm_resp.strip():
            # 清理可能的多余标点
            question = llm_resp.strip().rstrip("。？?").strip()
            if not question.endswith("？"):
                question += "？"
            return question
        else:
            # fallback 到拼接
            return "，".join(slot_desc) + "，请问相关规定是什么？"


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

    def process_multi_question(self, user_input: str, conversation_id: str) -> Tuple[str, bool]:
        from .session_manager import get_conversation_history
        history = get_conversation_history(conversation_id)

        if self.current_purpose and self.is_related_to_last_intent(user_input, history):
            processor = self.get_processor_for_scene(self.current_purpose)
            # 先更新槽位（processor.process 内部应完成槽位填充）
            response = processor.process(user_input, None)

            # ✅ 关键：动态判断是否已填满
            fully_filled = True
            for p in processor.scene_config.get("parameters", []):
                if p.get("required", False):
                    val = processor.slot.get(p["name"])
                    if val is None or str(val).strip() == "":
                        fully_filled = False
                        break

            return (response, fully_filled)  # ← 根据实际状态返回！

        else:
            # 检测到不相关输入，重置当前意图和槽位状态（不清 session）
            if self.current_purpose:
                self.current_purpose = ""
                self.processors.clear()

            scene, params = self.recognize_intent_and_extract_slots(user_input)

            # ===== 特殊场景短路处理 =====
            if scene == "greeting":
                self.current_purpose = "greeting"
                final_msg = "【FINAL】您好！我是南航智能客服，请问有什么可以帮您？"
                return (final_msg, False)

            elif scene == "other_scenario":
                self.current_purpose = ""
                return (user_input, True)

            # ===== 槽位场景处理 =====
            if scene:
                self.current_purpose = scene
                processor = self.get_processor_for_scene(scene)
                for k, v in params.items():
                    if k in processor.slot and v not in (None, "", "null", "None"):
                        processor.slot[k] = str(v).strip()

                fully_filled = True
                for p in processor.scene_config.get("parameters", []):
                    if p.get("required", False):
                        val = processor.slot.get(p["name"])
                        if val is None or str(val).strip() == "":
                            fully_filled = False
                            break

                if fully_filled:
                    query = processor._generate_natural_query_with_llm()
                    return (query, True)
                else:
                    question = processor._ask_missing()
                    return (question, False)
            else:
                self.current_purpose = ""
                return ("", False)

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

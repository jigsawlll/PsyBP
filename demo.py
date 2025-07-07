# app_cli_original_functions.py
# ──────────────────────────────────────────────────────────────
#  功能：保留原文件所有函数结构，移除 Flask/DB，改造为命令行应用。
#  说明：运行前请确保已安装所需库：pip install transformers torch
# ──────────────────────────────────────────────────────────────
import os
import glob
import threading
import torch
from datetime import datetime
from threading import Thread
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TextIteratorStreamer, BitsAndBytesConfig)

# ──────────────────────────────────────────────────────────────
#  0. 全局常量 (保留)
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_USER_DIR = os.path.join(BASE_DIR, "user_data")  # 每个用户：history / portrait
DEFAULT_USER_ID = "cli_user"  # 为命令行界面设置一个默认用户ID

os.makedirs(BASE_USER_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
#  1. 用户目录工具 (保留)
# ──────────────────────────────────────────────────────────────
def create_user_dirs(user_id: str):
    """为指定 user_id 创建 user_data/{uid}/history portrait 目录"""
    root = os.path.join(BASE_USER_DIR, user_id)
    os.makedirs(os.path.join(root, "history"), exist_ok=True)
    os.makedirs(os.path.join(root, "portrait"), exist_ok=True)


def list_history_files(user_id: str):
    p = os.path.join(BASE_USER_DIR, user_id, "history")
    return sorted(os.path.basename(f) for f in glob.glob(os.path.join(p, "*.txt")))


def list_portrait_files(user_id: str):
    p = os.path.join(BASE_USER_DIR, user_id, "portrait")
    return sorted(os.path.basename(f) for f in glob.glob(os.path.join(p, "*.txt")))


# ──────────────────────────────────────────────────────────────
#  2. 模型加载 (保留)
# ──────────────────────────────────────────────────────────────
MODEL_PATH = r"E:\model\PsyBPLLM"  # ★ 根据实际路径修改
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def _load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_cfg
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    return model, tokenizer, streamer


# ──────────────────────────────────────────────────────────────
#  3. 对话历史容器 (保留)
# ──────────────────────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的心理咨询师和心理专家，并且精通从文字中提取来访者的基本信息并能基于来访者的基本信息与患者进行心理咨询."
)
USER_CHAT_HISTORIES = {}  # user_id -> list[dict]
_history_lock = threading.Lock()


def get_user_chat_history(uid: str):
    """获取或初始化指定用户的对话历史"""
    if uid not in USER_CHAT_HISTORIES:
        with _history_lock:
            if uid not in USER_CHAT_HISTORIES:
                USER_CHAT_HISTORIES[uid] = [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
                ]
    return USER_CHAT_HISTORIES[uid]


# ──────────────────────────────────────────────────────────────
#  4. 保存对话 & 生成 / 注入画像 (保留)
# ──────────────────────────────────────────────────────────────
def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_conversation(uid: str, history: list):
    """将内存中的对话历史保存到文件"""
    if not uid: return "错误：无法在未指定用户的情况下保存！"

    # 将 list[dict] 格式的历史记录转换为字符串
    conv_str = ""
    for msg in history:
        if msg['role'] != 'system':  # 通常不保存 system prompt
            conv_str += f"{msg['role'].capitalize()}: {msg['content']}\n\n"

    if not conv_str.strip():
        return "对话为空，无需保存。"

    hist_dir = os.path.join(BASE_USER_DIR, uid, "history")
    os.makedirs(hist_dir, exist_ok=True)
    fname = f"history_{_timestamp()}.txt"
    try:
        with open(os.path.join(hist_dir, fname), "w", encoding="utf-8") as f:
            f.write(conv_str)
        return f"✅ 对话已保存到文件：{fname}"
    except Exception as e:
        return f"保存失败：{e}"


def _generate_portrait_from_file(uid: str, file_name: str, model, tokenizer):
    """从指定的历史文件生成用户画像"""
    if not uid or not file_name:
        return "错误：未提供有效对话文件，无法生成用户画像！"
    path = os.path.join(BASE_USER_DIR, uid, "history", file_name)
    if not os.path.isfile(path):
        return f"错误：选定的对话文件不存在！路径: {path}"

    with open(path, encoding="utf-8") as f:
        conv = f.read()

    messages = [
        {"role": "system",
         "content": (
             "你是一名专业心理咨询师和文字分析专家。"
             "下面 user 会给你一段完整对话，请你**仅**按以下九个要点提取信息，"
             "生成一份结构化用户画像（用 Markdown 列表均可）。"
             "不得续写对话、不得添加无关内容，记录中未提到的写“未提及”：\n"
             "① 基本信息(年龄，性别，职业，生活状况)\n"
             "② 情绪状态\n③ 心理需求与目标\n④ 应对机制\n⑤ 认知模式\n"
             "⑥ 社会支持与人际关系\n⑦ 生活质量\n⑧ 认知与情感倾向\n⑨ 咨询中反应"
         )},
        {"role": "user", "content": conv}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    resp = tokenizer.decode(outs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # 保存画像文件
    por_dir = os.path.join(BASE_USER_DIR, uid, "portrait")
    os.makedirs(por_dir, exist_ok=True)
    fname = f"用户画像_{_timestamp()}.txt"
    with open(os.path.join(por_dir, fname), "w", encoding="utf-8") as f:
        f.write(resp)

    print("\n" + "=" * 20 + " 用户画像已生成 " + "=" * 20)
    print(resp)
    print("=" * 58 + "\n")
    return f"✅ 用户画像已生成并保存到文件：{fname}"


def _inject_portrait(uid: str, portrait_file: str):
    """从指定的画像文件读取内容并注入到当前对话的 system prompt"""
    if not uid or not portrait_file:
        return "错误：未选择画像文件！"
    p_path = os.path.join(BASE_USER_DIR, uid, "portrait", portrait_file)
    if not os.path.isfile(p_path):
        return f"错误：画像文件不存在！路径: {p_path}"

    with open(p_path, encoding="utf-8") as f:
        content = f.read()

    hist = get_user_chat_history(uid)
    # 移除旧的画像（如果有的话），保留最基础的 system prompt
    base_sys = hist[0]["content"].split("### 用户画像（供参考", 1)[0].rstrip()

    # 注入新的画像
    hist[0]["content"] = base_sys + "\n### 用户画像（供参考，请勿直接展示）\n" + content.strip()
    return f"✅ 已将 {portrait_file} 的内容注入到当前对话中。"


# ──────────────────────────────────────────────────────────────
#  5. 主程序 & 命令行交互
# ──────────────────────────────────────────────────────────────
def main():
    """主交互循环"""
    print("🚀 正在加载模型，请稍候 …")
    model, tokenizer, streamer = _load_model()
    print("✅ 模型加载完毕")

    # 为当前命令行会话初始化用户目录和对话历史
    create_user_dirs(DEFAULT_USER_ID)
    history = get_user_chat_history(DEFAULT_USER_ID)

    print("\n" + "=" * 60)
    print("欢迎使用命令行心理咨询助手。")
    print("支持的命令:")
    print("  /save             - 保存当前对话到 history 目录")
    print("  /list history     - 列出已保存的对话文件")
    print("  /list portraits   - 列出已生成的画像文件")
    print("  /gen portrait     - （跟据提示）从指定对话文件生成画像")
    print("  /inject portrait  - （跟据提示）将指定画像注入当前对话")
    print("  /reset            - 重置当前对话历史")
    print("  /quit             - 退出程序")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("\nYou: ")
            cmd = user_input.lower()

            if cmd == "/quit":
                print("再见！")
                break

            elif cmd == "/reset":
                USER_CHAT_HISTORIES.pop(DEFAULT_USER_ID, None)
                history = get_user_chat_history(DEFAULT_USER_ID)
                print("\n[系统] 对话已重置。")
                continue

            elif cmd == "/save":
                status = _save_conversation(DEFAULT_USER_ID, history)
                print(f"\n[系统] {status}")
                continue

            elif cmd == "/list history":
                files = list_history_files(DEFAULT_USER_ID)
                print("\n[系统] 已保存的对话文件:")
                if not files: print("  - 无")
                for f in files: print(f"  - {f}")
                continue

            elif cmd == "/list portraits":
                files = list_portrait_files(DEFAULT_USER_ID)
                print("\n[系统] 已生成的画像文件:")
                if not files: print("  - 无")
                for f in files: print(f"  - {f}")
                continue

            elif cmd == "/gen portrait":
                files = list_history_files(DEFAULT_USER_ID)
                if not files:
                    print("\n[系统] 没有任何已保存的对话文件，请先使用 /save 保存。")
                    continue
                fname = input(f"[系统] 请输入要用于生成画像的对话文件名 (例如: {files[-1]}): ")
                status = _generate_portrait_from_file(DEFAULT_USER_ID, fname.strip(), model, tokenizer)
                print(f"\n[系统] {status}")
                continue

            elif cmd == "/inject portrait":
                files = list_portrait_files(DEFAULT_USER_ID)
                if not files:
                    print("\n[系统] 没有任何已生成的画像文件，请先使用 /gen portrait 生成。")
                    continue
                fname = input(f"[系统] 请输入要注入的画像文件名 (例如: {files[-1]}): ")
                status = _inject_portrait(DEFAULT_USER_ID, fname.strip())
                print(f"\n[系统] {status}")
                continue

            # --- 如果不是命令，则进行对话 ---
            history.append({"role": "user", "content": user_input})
            prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            generation_kwargs = dict(
                **inputs, streamer=streamer, max_new_tokens=1024,
                do_sample=True, top_p=0.95, temperature=0.8
            )
            Thread(target=model.generate, kwargs=generation_kwargs).start()

            print("\nAssistant: ", end="", flush=True)
            assistant_response = ""
            for chunk in streamer:
                print(chunk, end="", flush=True)
                assistant_response += chunk

            history.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n发生严重错误: {e}")
            break


if __name__ == "__main__":
    main()
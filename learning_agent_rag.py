import os
import json
import faiss
import numpy as np
import gradio as gr

from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI

from pypdf import PdfReader
from docx import Document


# ============================================================
# 1. 基础配置
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

MEMORY_FILE = "agent_memory.json"


# ============================================================
# 2. 工具函数：读取不同类型文件
# ============================================================

def read_txt_file(file_path: str) -> str:
    encodings = ["utf-8", "gbk", "ansi"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""


def read_pdf_file(file_path: str) -> str:
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"
    except Exception as e:
        text = f"PDF读取失败：{e}"
    return text


def read_docx_file(file_path: str) -> str:
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        text = f"Word文档读取失败：{e}"
    return text


def read_file_content(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".txt", ".py", ".cpp", ".c", ".h", ".hpp", ".java", ".js", ".html", ".css", ".md", ".csv"]:
        return read_txt_file(file_path)

    if ext == ".pdf":
        return read_pdf_file(file_path)

    if ext == ".docx":
        return read_docx_file(file_path)

    return f"暂不支持该文件类型：{ext}"


# ============================================================
# 3. 文本切分 Chunk
# ============================================================

def split_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    text = text.replace("\r", "\n")
    text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


# ============================================================
# 4. Embedding 与向量库
# ============================================================

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


class SimpleVectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.metadata = []
        self.dimension = None

    def build(self, chunks: List[str], metadata: List[Dict]):
        if not chunks:
            raise ValueError("没有可用于构建向量库的文本内容。")

        embeddings = []

        for chunk in chunks:
            emb = get_embedding(chunk)
            embeddings.append(emb)

        embeddings = np.array(embeddings).astype("float32")

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

        self.chunks = chunks
        self.metadata = metadata

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        if self.index is None:
            return []

        query_emb = np.array([get_embedding(query)]).astype("float32")
        distances, indices = self.index.search(query_emb, top_k)

        results = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue

            results.append({
                "content": self.chunks[idx],
                "metadata": self.metadata[idx],
                "distance": float(distance)
            })

        return results


vector_store = SimpleVectorStore()


# ============================================================
# 5. 简单长期记忆模块
# ============================================================

def load_memory() -> List[Dict]:
    if not os.path.exists(MEMORY_FILE):
        return []

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_memory(memory: List[Dict]):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def add_memory(question: str, answer: str):
    memory = load_memory()

    item = {
        "question": question,
        "answer": answer
    }

    memory.append(item)

    # 只保留最近 20 条，避免上下文过长
    memory = memory[-20:]

    save_memory(memory)


def retrieve_memory(query: str, max_items: int = 3) -> str:
    memory = load_memory()

    if not memory:
        return "暂无历史记忆。"

    # 简单关键词匹配版记忆检索
    # 课程项目展示足够，也可以后续升级成向量记忆
    scores = []

    for item in memory:
        text = item["question"] + item["answer"]
        score = 0

        for word in query:
            if word in text:
                score += 1

        scores.append((score, item))

    scores.sort(key=lambda x: x[0], reverse=True)

    selected = [item for score, item in scores[:max_items] if score > 0]

    if not selected:
        return "未检索到相关历史记忆。"

    memory_text = ""

    for i, item in enumerate(selected, 1):
        memory_text += f"\n【历史记忆{i}】\n"
        memory_text += f"问题：{item['question']}\n"
        memory_text += f"回答：{item['answer']}\n"

    return memory_text


# ============================================================
# 6. Agent 任务识别
# ============================================================

def classify_task(user_query: str) -> str:
    prompt = f"""
你是一个学习辅助 Agent，需要判断用户当前问题属于哪种任务类型。

任务类型只能从下面选择一个：
1. 文档问答
2. 代码解释
3. 实验总结
4. 问题定位
5. 通用学习问答

用户问题：
{user_query}

请只输出任务类型名称，不要输出其他内容。
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个任务分类助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ============================================================
# 7. Agent 主回答逻辑
# ============================================================

def build_prompt(
    user_query: str,
    task_type: str,
    context: str,
    memory_text: str
) -> str:
    prompt = f"""
你是一个基于 Agent 和 RAG 技术的学习辅助系统，主要帮助用户完成：
1. 文档问答
2. 代码解释
3. 实验总结
4. 问题定位
5. 学习内容梳理

当前任务类型：{task_type}

以下是从用户上传的文档或代码中检索到的相关内容：

{context}

以下是系统检索到的历史记忆：

{memory_text}

用户问题：
{user_query}

请根据以上信息进行回答。

回答要求：
1. 优先依据检索到的文档内容回答。
2. 如果资料中没有明确答案，请说明“根据当前资料无法完全确定”。
3. 如果是代码解释，请按“整体功能、关键模块、核心流程、可改进点”回答。
4. 如果是实验总结，请按“实验目的、实验过程、实验结果、遇到的问题、实验小结”回答。
5. 如果是问题定位，请给出可能原因和解决步骤。
6. 语言简洁清楚，适合学生写实验报告和项目展示。
"""

    return prompt


def ask_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "你是一个严谨、清晰、适合课程实验场景的学习辅助 Agent。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


def agent_answer(user_query: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    if not user_query.strip():
        return "", history

    # 1. 判断任务类型
    task_type = classify_task(user_query)

    # 2. 检索文档
    search_results = vector_store.search(user_query, top_k=4)

    if search_results:
        context = ""
        for i, item in enumerate(search_results, 1):
            file_name = item["metadata"].get("file_name", "未知文件")
            context += f"\n【资料片段{i}，来源：{file_name}】\n"
            context += item["content"] + "\n"
    else:
        context = "当前还没有上传资料，或没有检索到相关内容。"

    # 3. 检索历史记忆
    memory_text = retrieve_memory(user_query)

    # 4. 构造 Prompt
    prompt = build_prompt(
        user_query=user_query,
        task_type=task_type,
        context=context,
        memory_text=memory_text
    )

    # 5. 调用大模型
    answer = ask_llm(prompt)

    # 6. 保存记忆
    add_memory(user_query, answer)

    # 7. 更新对话历史
    final_answer = f"【任务类型】{task_type}\n\n{answer}"

    history.append((user_query, final_answer))

    return "", history


# ============================================================
# 8. 上传文件并构建知识库
# ============================================================

def upload_files(files):
    if not files:
        return "请先上传文件。"

    all_chunks = []
    all_metadata = []

    file_info = []

    for file in files:
        file_path = file.name
        file_name = os.path.basename(file_path)

        content = read_file_content(file_path)

        if not content.strip():
            file_info.append(f"{file_name}：读取失败或内容为空")
            continue

        chunks = split_text(content)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "file_name": file_name
            })

        file_info.append(f"{file_name}：读取成功，切分为 {len(chunks)} 个片段")

    if not all_chunks:
        return "没有可用于构建知识库的内容。"

    vector_store.build(all_chunks, all_metadata)

    result = "知识库构建完成。\n\n"
    result += "\n".join(file_info)
    result += f"\n\n共写入 {len(all_chunks)} 个文本片段。"

    return result


# ============================================================
# 9. 清空记忆
# ============================================================

def clear_memory():
    save_memory([])
    return "历史记忆已清空。"


# ============================================================
# 10. Gradio Web 界面
# ============================================================

with gr.Blocks(title="Agent + RAG 学习辅助系统") as demo:
    gr.Markdown(
        """
# Agent + RAG 学习辅助系统

本系统支持上传实验文档、代码文件或报告资料，并基于 RAG 检索增强生成技术进行问答。

功能包括：

- 文档问答
- 代码解释
- 实验总结
- 问题定位
- 历史记忆辅助回答
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="上传实验文档或代码文件",
                file_count="multiple"
            )

            upload_button = gr.Button("构建知识库")
            upload_status = gr.Textbox(
                label="知识库状态",
                lines=8
            )

            clear_memory_button = gr.Button("清空历史记忆")
            memory_status = gr.Textbox(
                label="记忆状态",
                lines=2
            )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="学习辅助 Agent",
                height=500
            )

            user_input = gr.Textbox(
                label="请输入你的问题",
                placeholder="例如：请解释这段代码的核心流程 / 根据文档生成实验总结 / 为什么运行报错？",
                lines=3
            )

            send_button = gr.Button("发送")

    upload_button.click(
        fn=upload_files,
        inputs=file_input,
        outputs=upload_status
    )

    clear_memory_button.click(
        fn=clear_memory,
        inputs=None,
        outputs=memory_status
    )

    send_button.click(
        fn=agent_answer,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    )

    user_input.submit(
        fn=agent_answer,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Agent + RAG 学习辅助系统")
    print("=" * 60)
    print("正在启动 Web 界面...")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )

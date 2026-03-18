
# SimpleTuner + SD3 LoRA 微调配置指南

> 基于 SimpleTuner 的 Stable Diffusion 3 统一标签微调技术记录

本文记录了在实验室服务器环境下，使用 SimpleTuner 对 Stable Diffusion 3 进行 LoRA 微调的完整配置流程，重点解决依赖版本冲突与训练参数配置问题。

---

## 一、 环境依赖与版本管理

### 1. 核心依赖冲突解决

**Python 版本要求**：SimpleTuner 要求 `Python >= 3.12`，建议使用 conda 创建独立环境。

**关键版本锁定**：

| 包 | 版本 | 说明 |
| :--- | :--- | :--- |
| `diffusers` | `0.36.0` | `0.37+` 版本 API 路径变更，会导致 `controlnet` 模块导入失败 |
| `torchao` | `0.14.0` | 与 `diffusers 0.36.0` 兼容，`0.16+` 会出现 logger 未定义错误 |
| `transformers` | `>=4.57.0` | SD3 支持所需 |
| `accelerate` | `>=0.30.0` | 分布式训练支持 |

**安装顺序**（版本锁定优先于最新版）：

```bash
pip install diffusers==0.36.0 torchao==0.14.0
pip install -e .  # 最后安装 SimpleTuner
```

### 2. 常见错误速查

- **错误 1：controlnet 路径变更**
  - **报错**: `ModuleNotFoundError: No module named 'diffusers.models.controlnet'`
  - **解决**: 降级到 `diffusers==0.36.0`
- **错误 2：torchao 兼容性**
  - **报错**: `NameError: name 'logger' is not defined`
  - **解决**: 降级到 `torchao==0.14.0`

---

## 二、 训练配置详解

### 1. 主配置文件 (`config.json`)

> **⚠️ 关键约束**：`num_train_epochs` 与 `max_train_steps` 互斥，**必须二选一**。

```json
{
  "--max_train_steps": 2000,
  "--num_train_epochs": 0,
  "--train_batch_size": 1,
  "--gradient_accumulation_steps": 4,
  "--learning_rate": 1e-4,
  "--mixed_precision": "bf16",
  "--resolution": 1024,
  "--lora_rank": 16,
  "--lora_alpha": 32,
  "--seed": null,
  "--validation_seed": null
}
```

**参数说明**：
- `num_train_epochs: 0`：显式使用步数控制而非轮数。
- `seed: null`：每轮使用随机种子（如需固定可设为具体数值）。
- `mixed_precision: bf16`：RTX 3090/4090 推荐，节省显存且数值稳定。

### 2. 数据配置 (`multidatabackend.json`)

**统一标签微调配置**：

```json
{
  "id": "main-dataset",
  "instance_data_dir": "/path/to/your/dataset",
  "instance_prompt": "a real image",
  "caption_strategy": "instanceprompt",
  "only_instance_prompt": true,
  "prepend_instance_prompt": false,
  "resolution": 1024,
  "resolution_type": "pixel"
}
```

**关键参数解析**：

| 参数 | 作用 | 推荐值 |
| :--- | :--- | :--- |
| `caption_strategy` | 标签策略 | `instanceprompt`：使用固定提示词<br>`filename`：使用文件名<br>`metadata`：从 JSON 读取 |
| `instance_prompt` | 统一提示词 | 根据微调目标设定，如 `"a real image"` |
| `only_instance_prompt` | 忽略其他 caption 源 | `true`：强制使用统一标签 |
| `prepend_instance_prompt`| 前置提示词 | `false`：不添加额外前缀 |

**Caption Strategy 选择建议**：
- **统一标签微调**：`instanceprompt` + `only_instance_prompt: true`
- **文件名作为提示词**：`filename`（要求图片文件名即为描述词）
- **已有标注数据**：`metadata`（需配合 `metadata_backend: discovery`）

---

## 三、 训练启动

配置完成后，在 SimpleTuner 目录下执行：

```bash
cd /path/to/SimpleTuner
bash simpletuner/train.sh
```

**验证启动成功**：
1. 正常应看到 `Epoch 1/7` 或进度条开始推进。
2. 首次启动会生成 VAE cache（预处理潜在空间编码），属于正常现象。
3. ⏳ **耗时参考**：RTX 3090 上 2000 steps 约需 30-40 分钟。

---

## 四、 推理测试

训练完成后，使用 `diffusers` 加载 LoRA 权重进行验证：

```python
import torch
from diffusers import StableDiffusion3Pipeline

# 1. 加载基础模型
pipe = StableDiffusion3Pipeline.from_pretrained(
    "models/sdv3-main",
    torch_dtype=torch.bfloat16,
    variant="fp16"
)

# 2. 加载 LoRA 权重
pipe.load_lora_weights("outputs/pytorch_lora_weights.safetensors")
pipe = pipe.to("cuda")

# 3. 生成测试
image = pipe(
    prompt="a real image",  # 使用训练时的统一标签
    negative_prompt="painting, drawing, illustration, cartoon, sketch, art",
    num_inference_steps=28,
    guidance_scale=7.0,
    height=1024,
    width=1024,
    generator=torch.manual_seed(42)  # 可选：固定种子便于复现
).images[0]

image.save("output.png")
```

**对比测试建议**：
- **固定种子** (`seed=42`)：用于验证训练的可复现性。
- **随机种子**（不设置 `generator`）：用于观察生成结果的多样性。

---

## 五、 训练参数参考表

| 参数 | 推荐值 | 说明 |
| :--- | :--- | :--- |
| `max_train_steps` | `2000-5000` | 根据数据集大小调整 |
| `train_batch_size` | `1-4` | 受显存限制，3090 24GB 可设为 4 |
| `gradient_accumulation_steps` | `2-4` | 等效 Batch Size = `batch_size` × `accumulation_steps` |
| `learning_rate` | `1e-4` ~ `4e-4` | LoRA rank 16 时 `1e-4` 较为稳定 |
| `resolution` | `1024` | SD3 原生推荐分辨率 |
| `lora_rank` | `16-32` | 16 为默认值，32 会占用更多显存 |
| `lora_alpha` | `32-64` | 通常设定为 Rank 值的 2 倍 |

---

## 六、 总结

- 📦 **版本锁定**：`diffusers 0.36.0` + `torchao 0.14.0` 为当前最稳定组合。
- ⚙️ **配置互斥**：`num_train_epochs` 与 `max_train_steps` 必须二选一，强烈建议使用步数（steps）控制。
- 🏷️ **统一标签**：使用 `instanceprompt` 策略并配合 `only_instance_prompt: true`，即可实现完美的固定提示词微调。
- 🎲 **随机种子**：根据实验需求，合理选择固定（可复现）或随机（验证多样性）。
- 💾 **模型输出**：训练完成后，LoRA 权重默认保存路径为 `outputs/pytorch_lora_weights.safetensors`。
- 💻 **环境信息**：`Ubuntu 20.04` / `Python 3.12` / `RTX 3090 24GB` / `SimpleTuner v4.0`
```

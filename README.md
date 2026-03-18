# SimpleTuner + SD3 LoRA 微调配置指南

> 基于 SimpleTuner 的 Stable Diffusion 3 统一标签微调技术记录

本文记录了在实验室服务器环境下，使用 SimpleTuner 对 Stable Diffusion 3 进行 LoRA 微调的完整配置流程，重点解决依赖版本冲突与训练参数配置问题。

## 环境依赖版本管理

### 核心依赖冲突解决

**Python 版本要求**：SimpleTuner 要求 Python >= 3.12，建议使用 conda 创建独立环境。

**关键版本锁定**：

| 包 | 版本 | 说明 |
|---|---|---|
| diffusers | 0.36.0 | 0.37+ 版本 API 路径变更，会导致 `controlnet` 模块导入失败 |
| torchao | 0.14.0 | 与 diffusers 0.36.0 兼容，0.16+ 会出现 logger 未定义错误 |
| transformers | >=4.57.0 | SD3 支持所需 |
| accelerate | >=0.30.0 | 分布式训练支持 |

**安装顺序**（版本锁定优先于最新版）：

```bash
pip install diffusers==0.36.0 torchao==0.14.0
pip install -e .  # 最后安装 SimpleTuner# simpletuner-notes

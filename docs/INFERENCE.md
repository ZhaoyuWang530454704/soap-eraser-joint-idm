# 使用完整模型包推理

## 1. 环境与模型包

计算在有 CUDA 的 Linux 服务器执行。原实验依赖版本见 [`environment_server.json`](../provenance/environment_server.json)，`requirements.txt` 记录同一版本组合。仓库整理验证复用原服务器环境，没有宣称重新验证过全新机器的依赖安装。

使用已经交付的 `complete_joint_IDM_model.zip`。其 SHA256 必须为：

```text
f981e65886e2de9d8e65d8c55ea549cd21f434ffefe315c55d931269c91748b6
```

包内包括 `model.pt`、`contract.json`、完整原推理代码、三个辅助源码、四个投影/归一化文件及 `package_manifest.json`。原 `model.pt` 的 state 包含完整 187 项，视觉骨干也在其中。

仓库不自动下载模型，不自动查找或占用 GPU。

## 2. 运行

```bash
python scripts/predict.py \
  --model-bundle /path/to/complete_joint_IDM_model.zip \
  --cache-dir /path/to/model_cache \
  --rgb /path/to/rgb.npy \
  --out /path/to/prediction.npz \
  --device cuda:0
```

选择已为本次计算分配的 GPU。`--device` 是可见 GPU 的逻辑编号，可配合 `CUDA_VISIBLE_DEVICES=GPU-UUID`。

封装先验证 ZIP、原逐文件 manifest 和仓库源码对应关系，再调用包内原 `public_predict.py`；没有重写模型前向或融合数学。

`--cache-dir` 可使用新目录或已经完整核验的本版解包目录。文件不完整/哈希不符则报错，保留证据，不自动覆盖。`--out` 必须以 `.npz` 结束且不存在。

只需检查解包、不做模型加载或推理时：

```bash
python scripts/predict.py \
  --model-bundle /path/to/complete_joint_IDM_model.zip \
  --cache-dir /path/to/model_cache \
  --unpack-only
```

## 3. 输入

- 单条记录 `rgb.npy`，`uint8 [N,256,640,3]`，`N>=2`。
- 左半第三人称，右半腕部，RGB 颜色顺序。
- 本版评测中 `N=376`，15 Hz；两路已同步。
- 不输入 GT、action、state、关节角或时间编号。

换任务、视角、分辨率或节拍不等于已验证支持。固定投影区域、颜色规则和监督分布均来自当前 SOAP TRAIN。

## 4. 输出 NPZ

| key | 形状/含义 |
|---|---|
| `raw` | `[N-1,10]`，重叠窗口融合后的原预测，主结果 |
| `ema05` | `[N-1,10]`，可选 EMA0.5 后处理 |
| `window_outputs` | `[K,17,10]`，融合前窗口预测 |
| `window_latents` | `[K,17,256]`，窗口时序特征 |
| `auxiliary` | `[N-1,1344]`，由 RGB 重算的辅助特征 |
| `starts` | `K` 个窗口起点；本实验 K=47 |
| `actual_batch_starts` | 包含最后 batch 补位的窗口起点 |
| `actual_batch_valid` | 对应的真实窗口标记 |

输出另外附 `prediction.receipt.json`，记录输入/输出 SHA、模型包 SHA 和调用信息。动作列与单位见 [数据合同](DATA_CONTRACT.md)。

RAW 并不是未经处理的单窗口输出，它已经融合多个窗口。EMA 是另加的一层输出处理，不能把 EMA 的误差填入 RAW 指标。

# 本版训练配置与损失

配置的权威来源是 [`experiment/contract.json`](../experiment/contract.json)，实际训练循环是 [`train.py`](../experiment/source/train.py)。以下说明对应已交付的旋转变化约束版。

## 1. 数据和 epoch

| 项目 | 本版设置 |
|---|---|
| 任务 | 仿真：把板擦放到洗手液瓶上 |
| TRAIN | 50 条，`sim000000`～`sim000049`，共 18750 个动作目标 |
| TEST | 10 条，`eval0000`～`eval0009`，共 3750 个动作目标 |
| 每条 | 376 帧双视角 RGB，375 行已对齐动作，15 Hz |
| 最终训练 | fresh 初始化，全部 50 条 TRAIN，60 epoch |
| 本轮 epoch 选型 | 不重新选；沿用冻结的 TRAIN 内部 40/10 选型所确定的 60 epoch |
| 初始化 | ImageNet ResNet18 + 固定 task seed；不载入旧任务 checkpoint |
| 初始 state 检查 | 优化前严格匹配冻结的全部 187 个 name/shape/dtype/bytes/value SHA |
| 随机种子 | 初始化 `20260812`；训练随机流 `seed + rank×1000003` |

内部选型的 10 条来自 TRAIN，与最终 TEST 的 10 条不同。该轮选型记录在 [`epoch_selection.json`](../provenance/epoch_selection.json)；它是复现依据，不是本版在 TEST 上挑 epoch 的记录。

一条 375 行记录切成 47 个长度 17、步长 8 的窗口，所以每 epoch 有 `50×47=2350` 个真实窗口。补 2 个零损失窗口，凑成 2352 个调度槽；每更新 8 窗，因此为 **294 次参数更新/epoch，60 epoch 共 17640 次**。

每个真实窗口每 epoch 使用一次。窗口重叠造成一帧出现多次，按原帧覆盖次数的倒数加权，使每个真实 TRAIN 动作行每 epoch 的总权重为 1。尾部复制帧与全局 batch 补位的损失为零。

## 2. 两卡训练与优化器

| 项目 | 设置 |
|---|---|
| GPU | 两张 A100；`world_size=2`，NCCL DDP |
| 每卡实际 microbatch | 2 个独立窗口 |
| 梯度累积 | 2 次 backward |
| 每次更新的全局窗口数 | `2卡×2窗×2累积=8` |
| 优化器 | AdamW，weight decay `1e-4` |
| 梯度裁剪 | 全参数 norm 上限 10 |
| 学习率日程 | cosine，60 epoch horizon，floor factor 0.05 |
| 精度 | FP32 master；ResNet 主视觉计算 BF16；head/fine/损失 FP32 |
| 数值设置 | TF32 关闭，确定性算法；BN running 统计固定 |
| 显存检查 | 每 rank 峰值余量与实际空闲均至少 10 GiB |

`DataLoader` 每卡取 4 窗；训练代码拆成两次各 2 窗的真实前向/反向。第一次使用 `DDP.no_sync()`，第二次同步梯度。两个 backward 共用一次全局有效行权重分母。

分组基础学习率：

| 参数组 | 学习率 |
|---|---:|
| ResNet18 backbone | `1e-5` |
| fine_visual / fine_closure | `1e-5` |
| views / visual_fuse / aux / shared / position / orientation 等 | `5e-5` |
| extra / closure_residual / dense_reduce / dense_embed | `3e-4` |
| temporal TCN | `3e-4` |

ResNet 没有冻结。BN 的 affine 参数也训练；只固定其 running mean/variance。

## 3. 联合损失

令 `p` 为 XYZ，`R` 为正交化后的旋转矩阵，`c` 为闭合度，上标 `*` 为标签。本节公式省略源码中的数值稳定 epsilon，精确计算见 [`common.py`](../experiment/source/common.py)。

### 3.1 每帧动作误差

```text
L_XYZ = ||p - p*||₂ / 0.02 m

L_rotation = sqrt(0.5 × ||R - R*||²_F) / 0.0872664626 rad

L_closure = SmoothL1(c,c*; beta=0.05) / 0.05
```

- XYZ 使用三维欧氏距离，直接优化末端空间位置。
- 旋转在合法旋转矩阵上计算 chordal 距离，避免把 6D 各分量当成独立角度。
- 夹爪使用 SmoothL1：小误差较平滑，大误差不会像纯平方误差那样快速放大。

### 3.2 XYZ 相邻帧变化

```text
L_XYZ_delta = ||(p_t-p_(t-1)) - (p*_t-p*_(t-1))||₂ / 0.01 m
```

这约束预测轨迹的位移变化接近标签，不是要求速度为零。沿用原训练循环的逐行覆盖权重；窗口第一个位置没有前一个位置，记零。

### 3.3 本版相对旋转变化约束

先计算每对相邻帧的 body-frame 相对旋转：

```text
DeltaR_pred = R_pred(t-1)^T × R_pred(t)
DeltaR_GT   = R_GT(t-1)^T   × R_GT(t)

L_rotation_delta =
  [sqrt(0.5 × ||DeltaR_pred-DeltaR_GT||²_F + 1e-12) - 1e-6]
  / 0.0436332313 rad
```

其意义是：“预测的转动方向和幅度，应与标签这一步的转动一致”。它不直接平滑六个旋转分量，也不强迫机械臂停止转动。固定 15 Hz 下，转角增量误差与角速度误差成比例。

每条记录有 374 对相邻动作，TRAIN 共 `50×374=18700` 对。旋转增量单独计算 pair coverage：同一原始相邻对在全部重叠窗口中的总权重为 1。跨记录、窗口首位、尾部重复行和全局补位不参与这一项。

如果整个姿态序列乘上同一个左侧旋转偏置，相对转动可能仍相同。因此逐帧姿态损失仍然保留，用来约束绝对姿态。

### 3.4 总权重

```text
L_total = L_XYZ + 0.5×L_XYZ_delta
        + 0.25×(L_rotation + 0.5×L_rotation_delta)
        + 3×L_closure
```

旋转增量的组合权重为 `0.25×0.5=0.125`。旋转 pair 权重经 multiplier 换算后加入原逐行训练总损失，最终沿用原全局行权重分母。

同一次 backward 包含各项损失。原实测控制检查验证了 XYZ、旋转、夹爪、旋转变化损失均能产生到共享视觉 conv1、融合层、细节 CNN 和 TCN 的非零有限梯度。

## 4. TRAIN-only 准备

继承的 [`prepare.py`](../preparation_reference/prepare.py) 包含 RGB 灰度/边缘、PCA、两组 PLS、颜色统计和归一化的完整准备过程。原始动作已经是统一动作合同下的 10 维目标；此步骤不重新做 FK 或修改标签。

本版直接复用由相同 SOAP TRAIN 拟合并冻结的准备文件。参照：

- [`TRAIN_projection_receipt.json`](../provenance/TRAIN_projection_receipt.json)：50 条 TRAIN 拟合范围和投影 SHA。
- [`prepared_training_files.json`](../provenance/prepared_training_files.json)：本版实际使用的准备文件大小与 SHA。
- [`expected_inputs.json`](../provenance/expected_inputs.json)：100 个 TRAIN 输入与 20 个 TEST video-only 输入的字节依据。

`prepare.py` 与 `choose_epoch.py` 放在 `preparation_reference/`，不是本轮新执行入口。其原始工程依赖 `common.py` 和根目录布局；不能在该参考目录直接启动。

## 5. 在服务器新目录重现 refit

仓库提供一个仅准备目录的入口，复用现有已冻结数据/投影并逐项验证 SHA。它不自动启动训练。

```bash
python scripts/stage_refit.py \
  --prepared-root /path/to/original_prepared_experiment \
  --run-dir /path/to/new_refit_run \
  --gpus GPU-UUID_A GPU-UUID_B
```

`prepared-root` 应具备本版 `dataset/data`、`features/TRAIN_meta.npz`、`features/refit/`、`pretrained/` 与 `epoch_selection.json`。本项目原服务器目录记录在准备文件元数据中。脚本只链接经过核验的 TRAIN 输入，复制 epoch 决策和科学源码；不带入 TEST GT，不改写原文件。

在已为本次训练分配的两张 A100 上，以原依赖环境启动：

```bash
CUDA_VISIBLE_DEVICES=GPU-UUID_A,GPU-UUID_B \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m torch.distributed.run \
  --standalone --nproc-per-node=2 --max-restarts=0 \
  /path/to/new_refit_run/source/train.py refit
```

新目录中的配置与科学源码 SHA 保持原值；初始 187 项严格指纹检查不删除。新数据、投影或初始化不一致会报错，应独立登记新实验。

此入口只覆盖同数据的 fresh refit。原 `owner.py` 是整个历史研究的实际入口；`orchestrate.py` 中的 `main()` 是继承的更早选型流程。二者有原服务器 runtime、GPU lease、历史评分与后处理路径依赖，保留用于审计，不作为本仓库的通用一键命令。

## 6. 已完成的原版验证

- 两卡实际前向、联合梯度、优化更新及跨 rank 参数/梯度一致性。
- 全 187 项严格加载，保存前后参数指纹一致，重载输出和 latent 逐位一致。
- 采样 ID、窗口行映射、补位掩码、原帧/旋转 pair 覆盖审计。
- 10 条完整 TEST RGB 预测先冻结；独立检查窗口、反归一化和 SciPy 融合。
- 完整推理包在实际 GPU 上从 RGB 重算后，与冻结预测核对；之后才读 TEST GT 评分。
- 评分、动作 ZIP、资源退出与最终仿真结果独立核验。

本次代码上传不重启原训练、预测或仿真任务。

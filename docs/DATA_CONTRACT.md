# 数据与动作合同

## RGB

每条记录的 `rgb.npy` 是 `uint8 [376,256,640,3]`，颜色顺序为 RGB：

```text
rgb[:, :, :320, :]  = 第三人称
rgb[:, :, 320:, :]  = 腕部
```

两路均为高 256、宽 320，按原 15 Hz 配对。模型拆分视角后分别做主骨干编码；细节 CNN 另外沿通道叠成 6 通道。

模型产生 375 行动作，使用前 375 个观察位置。每个时刻的动作由 17 帧离线上下文和 RGB 辅助特征决定。

## 动作格式

`action_raw.npy` 是 `float32 [375,10]`：

| 列 | 含义 |
|---|---|
| 0～2 | XYZ，米，`robot_base` 坐标系 |
| 3～5 | 旋转矩阵第一列 `[r11,r21,r31]` |
| 6～8 | 旋转矩阵第二列 `[r12,r22,r32]` |
| 9 | 原始 commanded closure；0 开、1 闭 |

旋转的第三列由前两列正交化后叉乘得到。不能把这 6 个数作为 6 个角度解读。

### action 与 state

本版监督是 **目标 action**。仿真原始 next-endpoint `action_joint` 的 7 个目标关节角，通过已准备好的统一 calibrated FK 转为固定 TCP 的绝对 XYZ/姿态；闭合度保留原始目标指令。

执行后的真实 state 用于后续仿真回放评估，不作为这里的训练标签。学习的是视觉对应的目标动作恢复。

### 固定 TCP

规格：`xarm7_fixed_tcp261983_abs_15hz_v1`。

```text
T_link7_fixed_tcp =
[[1,0,0,0],
 [0,1,0,0],
 [0,0,1,0.2619830737877168],
 [0,0,0,1]]
```

即 link7 沿自身 Z 轴平移约 261.9830738 mm，旋转为单位矩阵。所有闭合度使用同一个固定 TCP。完整准备依据与 URDF 等 SHA 见 [`canonical_action_metadata.json`](../provenance/canonical_action_metadata.json)。当前模型训练代码消费已准备好的 action，不重新执行 FK。

## 行对齐

```text
输入观察行 t  → 原目标 action 行 t+1
t = 0,...,374
```

目标时间为 `(t+1)/15` 秒；观察时间为 `t/15` 秒。canonical 数据已经完成该映射，训练和推理结果不得再向前或向后移动一行。

RAW/EMA 都保留这个对应关系。保存序列没有偏移搜索、裁剪、筛帧或重采样。

## 夹爪标签

本 SOAP 仿真版使用原始目标闭合指令，没有使用白板实验里的平滑代理标签。不同任务的标签语义与对齐方式不能混用。

## 数据布局

原工程使用：

```text
dataset/
  data/episodes/sim000000/
    rgb.npy
    action_raw.npy
  ... sim000049
  test_video_only/episodes/eval0000/
    rgb.npy
    timestamps.npy
  ... eval0009
features/
  TRAIN_meta.npz
  refit/
    auxiliary.npy
    projection/
      pca10.npz
      xyz10.npz
      all1010.npz
      normalization10.npz
```

`all1010.npz` 的历史命名表示 `all10` 这组 PLS 投影文件，不是 1010 维。实际是 48 维投影。

TEST 阶段从 video-only 目录取 RGB。GT 评分输入存放在独立目录，并在全部预测冻结、独立审计通过后才读取。仓库内只有数据格式与文件哈希，没有数组或 GT 数值。

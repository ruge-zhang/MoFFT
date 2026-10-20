# Linux AArch64／灵晟 SME1 适配说明

本补丁增加显式选择的 Linux SME1 路径。SME（Scalable Matrix Extension，可伸缩矩阵扩展）的第一版在本文称为 SME1，用于与 SME2 区分。默认仍使用上游 Apple SME2 路径，不自动把 SME2 机器改为 SME1。

## 选择与适配范围

| 项目 | 默认 SME2 路径 | 显式 SME1 路径 |
|---|---|---|
| 选择方式 | 原构建命令不变 | `MOFFT_SME1_PORT=ON`／生成器 `--sme1` |
| 平台与流式向量长度 | 保留原限制和设置 | Linux AArch64，512 位 |
| 指令与候选 | 原成组载入、ZA 成组读取、矩阵 FMA、宽度与旋转候选 | 禁止 SME2 成组指令；保留 SME 外积，使用已有 SVE 复数旋转因子和单行读取；仅采用已移植的宽度 2、无临时 ZA 旋转候选 |
| 公开 FFT 接口 | 原接口与运行时 | 同一公开接口与运行时 |
| 流式调用入口 | 原 Apple 汇编入口 | 编译器管理的 Linux C 入口及 private-ZA 状态 |
| 默认成本配置 | 原 Apple bootstrap | 独立且明确未校准的 Linux bootstrap |

SME1 候选限制是本补丁的实现边界，不表示硬件永远不能支持其他组织。此补丁保留 FP32／FP64、前向／反向的生成入口；生成成功不等于所有尺寸、布局和调用状态均已在目标机器验证。

## 环境与构建

需要支持 SME ACLE 属性和 `<arm_sme.h>` 的 AArch64 Clang 工具链、CMake 3.24 及 Python 3.10 或更高版本。须在目标环境核实操作系统开放 SME、所需浮点能力及实际 streaming VL＝64 字节。构建标志不能证明硬件或系统能力成立；本补丁没有修改系统向量长度。

Linux AArch64 shell，在 MoFFT 仓库根目录执行，指定已安装的编译器绝对路径：

```sh
cmake -S . -B build/linux-sme1 -DCMAKE_BUILD_TYPE=Release \
  -DMOFFT_SME1_PORT=ON -DMOFFT_BUILD_TOOLS=OFF \
  -DCMAKE_C_COMPILER=/absolute/path/to/clang \
  -DCMAKE_CXX_COMPILER=/absolute/path/to/clang++ \
  -DCMAKE_EXE_LINKER_FLAGS="--rtlib=compiler-rt -lgcc_s"
```

同一环境和工作目录执行：

```sh
cmake --build build/linux-sme1 --target mofft mofft_correctness
```

上述 compiler-rt 链接设置来自 Linux Clang 对 SME ABI 辅助例程的需要，必须与实际安装的运行时匹配，不改变 Apple 的原链接设置。只有确认目标指令与 ABI 支持后，才能执行生成的正确性程序。本补丁的完整 CMake 构建及目标测试状态以提交附带的实际回执为准，不把示例命令当作已通过。

## 校准与工具边界

`linux-sme1-512-bootstrap.json` 的 `measured=false`：数值继承自原 bootstrap 的模型假设，只用于启动生成和构建，不是灵晟 CPU 的测量配置。每种 CPU／系统／向量长度／工具链环境应独立校准，并保留环境身份；不能把某台 CPU 的配置称为所有机器的通用校准。

本补丁尚未移入 Linux 校准、候选数值运行器和计划搜索工具。原 Apple 工具保持原样，SME1 构建明确关闭这些工具和原 Apple 候选运行器。不要在 SME1 机器上直接运行原 SME2 微基准，也不要把本补丁称为已包含完整目标校准的版本。

## SME2 回归与性能边界

默认构建参数、Apple 汇编入口、运行时、公开接口和 Apple profile 保持。平台无关回归应同时运行原 `test_compiler.py` 与新增 `test_sme1_port.py`；还需在同 profile 和生成参数下，对比修改前后的默认 SME2 生成源及候选选择。生成器源哈希按上游规则更新，旧 kernel wisdom 会因身份不匹配被拒绝；不得伪造旧哈希或关闭检查来强行复用。

生成源一致是保护原路径的回归证据，不能代替 SME2 真机正确性与性能复验。在声称原 SME2 性能没有退化前，应使用相同 SME2 机器、工具链、配置与测量边界复验；无可用真机时明确记录未验证项。

Linux 状态入口采用 Arm [ACLE](https://arm-software.github.io/acle/main/acle.html) 的 `__arm_new("za")` 和 `__arm_locally_streaming` 属性，让编译器实现 [AAPCS64](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst) 的 private-ZA／lazy-save 协议；不手写简化的 SMSTART／SMSTOP 包装替代 ABI。

## 已完成的构建与兼容检查

在 Linux AArch64 登录环境，以 CMake 3.28.2、BiSheng Enterprise 5.2.0／Clang 19.1.7 和 Python 3.12.7 完成以下检查，没有在该登录环境执行 SME／SME2 指令：

| 检查 | 结果 |
|---|---|
| 原生成器／新增适配回归 | 原版 40 项、适配版 43 项通过 |
| Linux SME1 完整构建 | 默认 17 个基数、340 个发射内核的静态库及正确性程序构建／链接通过 |
| CMake 安装与外部调用 | 安装导出通过；独立工程用 `find_package(MoFFT)` 导入并链接成功 |
| SME1 反汇编 | 保留外积指令，未发现本检查禁止的 SME2 矩阵 FMA／VGX 指令 |
| 默认 SME2 生成源 | 同一 Apple bootstrap 下，340 个发射内核对应的四份 C／头文件与原版逐字节一致，选择记录保持 |
| SME2 内核对象 | 同一 Linux 编译器和参数下，原版与适配版的 FP32、FP64 和注册表三个对象逐字节一致 |
| SME2 真机数值及性能 | 尚未在本补丁提交方验证，交由具备 SME2 真机的维护者补测 |

原 Apple 构建的 `-mno-implicit-sme` 参数保持不变；本次 Linux 编译器不识别该参数，因此 Linux SME2 对象比对双方都使用其支持的相同参数。该比对不等同于 Apple Clang 生成物一致性，也不替代 Apple 设备的正确性及性能结果。

维护者在 SME2 设备复验时，应使用独立构建目录并保持 `MOFFT_SME1_PORT=OFF`，沿用原平台配置、工具链、校准与测量条件。先完成原公开正确性程序和候选检查，再比较基准；保留原样本及环境身份。若使用旧 kernel wisdom，应按原源码身份规则重新生成匹配的选择表，不关闭校验，也不静默回退到 bootstrap 后宣称调优条件相同。

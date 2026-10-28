# seekdb × hipVS CAGRA 构建优化报告

## 基线与实现

- 比赛冻结基线：`amd-cagra-contest-baseline-v2`
- 冻结 commit：`58ddbd7459843e979a1a02f9bb4be9a3edacd3f0`
- 基线 hipVS commit：`f96a663d0a4441d6f5344bd3ef552b71ae7b370b`
- 参赛分支：`contest/hipvs-cagra-optimization`

冻结基线已经具备 SQL `WITH (lib=cuvs)`、CAGRA 构建和查询路由。本实现不再针对
`lib=cuvs` 同步构建一份完整、但在正常 GPU 查询中不会使用的 VSAG HNSW sidecar。
数据仍进入一份连续的 row-major FP32 缓冲；只有过滤、删除、陈旧索引、无 GPU、按 ID 计算
距离、读取额外列等实际回退场景，才从该缓冲一次性物化 VSAG HNSW。`lib=vsag` 路径不变。

优化点：

- 避免端到端构建路径同时执行 CAGRA 构图和完整 CPU HNSW 构图。
- 以倍增策略预留连续向量缓冲，减少 `std::vector` 扩容和搬移。
- `get_index_number` 和 `get_vid_bound` 在 sidecar 延迟期间直接使用 CAGRA 缓冲元数据。
- CPU sidecar 物化后继续走原有增量 VSAG 路径，保持过滤、删除和无 GPU 回退语义。
- 增加 `cuvs_vsag_defer` 与 `cuvs_vsag_materialize` 轨迹，验收时可确认未发生提前 CPU 构图。

## 统一测试方法

新增脚本：

```text
docs/gpu-vector-index-hipvs-cuvs/validation/run_build_compare.sh
```

冻结基线和候选版本分别从源码构建为两个 observer，均使用：

- `OB_BUILD_CUVS=ON`
- `OB_BUILD_CUVS_TRACE=ON`
- 同一 `libseekdb_cuvs_bridge.so`
- 同一 GPU、ROCm、hipVS、数据集、维度、M、ef 参数和 CPU 配额

每轮先清除 `lib=cuvs` 向量索引，再从
`CREATE VECTOR INDEX` 开始计时，在同一 SQL 调用内完成第一条固定 `APPROXIMATE` 查询。
脚本至少执行 3 轮并交替 baseline/candidate 顺序，结束后取两边构建耗时中位数计算加速比。
脚本同时检查轨迹中确实发生 `cuvs_build` 与 `cuvs_serve`，并根据固定精确 Top-100 分别
计算 Recall@10 与 Recall@100。原始耗时和汇总写入 `summary.json`。

完整命令见 `docs/gpu-vector-index-hipvs-cuvs/README.md`。

## 当前验证状态

当前机器没有 `/dev/kfd`，也没有 `rocminfo`、`hipcc` 或可用 ROCm 环境，因此无法执行 AMD GPU
构建、首条 GPU 查询或正式性能测试。

已完成：

- 从冻结 tag `amd-cagra-contest-baseline-v2` 创建参赛分支。
- `OB_BUILD_CUVS=OFF` 的完整源码构建成功，生成 1.3 GiB `seekdb` 可执行文件。
- 使用无 GPU 的桥接桩打开 `OB_BUILD_CUVS=ON` 后，包含 GPU 分支的 `oblib_lib` 目标编译通过。
- 编译产物完成 `lib=vsag` 的 HNSW 建表、插入和 `APPROXIMATE` 查询冒烟，返回预期 ID `1,2,3`。
- 最终候选源码使用 Clang 18 和冻结依赖分别配置 `OB_BUILD_CUVS=OFF/ON`，
  `oblib_lib` 均编译通过；`ON` 配置使用仅用于编译验证的无 GPU 桥接桩。
- 按冻结基线现有 CAGRA 集成实现构建路径优化。
- GPU smoke 脚本增加 deferred sidecar 和按需物化检查。
- 新增 baseline/candidate 端到端 A/B 构建脚本，同时输出 Recall@10 与 Recall@100。
- Shell 语法检查通过。

未完成：

- hipVS/cuVS 真实桥接库编译和 AMD GPU 运行。
- 3 轮 CPU/GPU 路径性能数据。
- Recall@10、Recall@100 和查询吞吐实测。

## 结果表

| 数据量 / 维度 | 冻结基线构建耗时 | 候选版本构建耗时 | 构建加速比 | 基线 / 候选 Recall@10 | 基线 / 候选 Recall@100 | 是否达标 |
| --- | --- | --- | --- | --- | --- | --- |
| 待 AMD GPU 云端实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待确认 |

本报告不会把 Notebook 中约 12.65 倍的批量查询加速当作索引构建成绩，也不会用自测耗时替代
组织方云端统一评测。

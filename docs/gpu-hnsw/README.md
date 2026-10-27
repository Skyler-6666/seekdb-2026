# 使用 hipVS 加速 seekdb 的 HNSW 索引构建

> 题目：使用 hipVS 加速 HNSW 索引构建（OceanBase 大赛，20 分）
>
> 代码分支：`competition/gpu-hnsw-hipvs`
> 仓库：<https://github.com/hechuan1023/OB-2026-seekdb>
> 基线：`competition/ai-function-enhancement`

---

## 1. 现状分析：HNSW 构建的时间花在哪里

seekdb 的向量索引由 **VSAG**（蚂蚁集团开源的 <https://github.com/antgroup/vsag>）实现，
以预编译静态库的形式链接进 observer。HNSW 的建图过程在 VSAG 内部是标准的
hnswlib 算法（malkov / yashunin 的 *Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs*）：

```
for each inserted vector v:
    l = random_level()                      # 按指数分布决定 v 的层高
    从入口点贪心下降，找到上层入口 ep
    W = search_layer(v, ep, ef_construction) # 在第 0 层做 ef_construction 大小的候选集搜索
    neighbors = select_neighbors_heuristic(W)   # 带剪枝的邻居选择启发式
    把 v 插入各层，并为每个邻居补反向边
```

其中 `search_layer` 是**距离计算密集**的部分：每插入一个向量，都要把它与候选
集合中的每个邻居做一次距离计算，距离计算次数约为 `O(N · ef_construction · log N)`。
距离计算本身是 `O(dim)` 的点积/平方差累加，因此整个建图是 `O(N · ef_construction · log N · dim)`。

结论：**建图时间几乎全部花在"批量 KNN 图"的计算上，而不是在图的指针操作上**。
这正是 GPU 擅长的事情——它可以同时对成千上万个向量做距离计算。

### 1.1 seekdb 中的调用链

```
create vector index ... with (type=hnsw)
  └─ ObDDLIndependentDag
       └─ ObVectorIndexAppendPipeline     # 按 tablet 逐个灌数据
            └─ ObHNSWIndexAppendBufferOperator::append_row
                 └─ ObPluginVectorIndexAdaptor::add_snap_index()   # ★ 改造点 1
       └─ ObVectorIndexBuildAndWritePipeline
            └─ ObHNSWIndexBuildOperator::serialize_vector_index
                 └─ ObPluginVectorIndexAdaptor::check_snap_hnswsq_index()  # ★ 改造点 2
                      └─ ObPluginVectorIndexAdaptor::serialize()
```

两个改造点分别对应"数据从哪来"和"索引什么时候成型"。

---

## 2. 方案设计

### 2.1 核心洞察：hipVS 的输出格式和 VSAG 的输入格式是同一个

这是整个方案成立的前提，方案里其余所有工作都是为了把这条路径接通。

| | hipVS | VSAG |
| --- | --- | --- |
| HNSW 实现 | vendored **nmslib/hnswlib v0.7.0**（`cpp/cmake/patches/hnswlib_override.json`） | vendored **hnswlib v0.7.0**（`third_party/vsag-129b82c/src/algorithm/hnswlib/`） |
| 序列化格式 | `hnswlib::saveIndex()` | `hnswlib::loadIndex()` |

两边跑的是**同一份 hnswlib**，因此序列化字节流完全一致。逐字段核对如下
（`hnswlib/hnswalg.h:599` `saveIndex` vs `vsag/src/algorithm/hnswlib/hnswalg.cpp:990` `DeserializeImpl`）：

| 偏移 | 字段 | 类型 | 大小 |
| --- | --- | --- | --- |
| 0 | `offsetLevel0_` | `size_t` | 8 |
| 8 | `max_elements_` | `size_t` | 8 |
| 16 | `cur_element_count` | `size_t` | 8 |
| 24 | `size_data_per_element_` | `size_t` | 8 |
| 32 | `label_offset_` | `size_t` | 8 |
| 40 | `offsetData_` | `size_t` | 8 |
| 48 | `maxlevel_` | `int` | 4 |
| 52 | `enterpoint_node_` | `tableint`（= `uint32_t`） | 4 |
| 56 | `maxM_` | `size_t` | 8 |
| 64 | `maxM0_` | `size_t` | 8 |
| 72 | `M_` | `size_t` | 8 |
| 80 | `mult_` | `double` | 8 |
| 88 | `ef_construction_` | `size_t` | 8 |
| | **合计** | | **96** |

第 0 层元素块的布局也一致，两边都是
`[ maxM0_ * 4 + 4 字节邻接表 ][ dim * 4 字节向量 ][ 8 字节 label ]`，
其中 `size_data_per_element_ = maxM0_ * 4 + 4 + dim * 4 + 8`。

> VSAG 的 `DeserializeImpl` 里有一段兼容代码：先按"新格式"读 12 字节
> （`enterpoint_node_` 4 字节 + `maxM_` 8 字节），只有当读到的 `maxM_` 与
> 目标索引自身的 `M_` **相等**时才认为是新格式；否则退回旧格式（8 字节
> `enterpoint_node_`）并抛 "The index was saved with different M_ value"。
> 这就是下面 2.3 节必须严格对齐 M 的原因。

### 2.2 M 的对齐：graph_degree = 2 × max_degree

VSAG 建索引时把 `M` 夹到 `[8, 64]`：

```cpp
// vsag/src/index/hnsw.cpp:62-64
auto M = std::min(std::max((int)hnsw_params.max_degree, MINIMAL_M), MAXIMAL_M);  // MINIMAL_M=8, MAXIMAL_M=64
```

而 hipVS 从 CAGRA 图转 HNSW 时，hnswlib 的 M 取的是**图度数的一半**：

```cpp
// hipVS/cpp/src/neighbors/detail/hnsw.hpp:353
auto appr_algo = std::make_unique<hnswlib::HierarchicalNSW<...>>(
    hnsw_index->get_space(), n_rows, cagra_index.graph().extent(1) / 2, params.ef_construction);
```

hnswlib 内部 `maxM_ = M_`，`maxM0_ = M_ * 2`。

所以只要让 GPU 侧的图度数取 `2 × max_degree`，就得到：

```
GPU 产物:  maxM_ = M_ = max_degree,  maxM0_ = 2 * max_degree
VSAG 目标: M_ = clamp(max_degree, 8, 64),  maxM_ = M_
```

两边 `M_` 与 `maxM_` 都相等 → VSAG 走"新格式"分支，加载成功。

seekdb 侧传入的 `max_degree` 用 `MAX(MIN(param->m_, 64), 8)` 计算，与 VSAG 的
夹取规则等价（对任意 m ≥ 0 两者结果相同）。

### 2.3 label 必须重写

hnswlib 给元素打 label 用的是**插入顺序**（0..N-1），而 seekdb 必须用**行 vid**
寻址元素。VSAG 加载时会用 `label_lookup_[getExternalLabel(i)] = i` 重建
label→内部 id 的映射，所以只要在交给 VSAG 之前把镜像里每个元素的外部 label
改写成对应的行 vid 即可，内部 id 和邻接表都不用动。

这由 `src/oblib/lib/vector/ob_hnsw_serial_format.cpp` 的
`patch_hnsw_serial_labels()` 完成，label 偏移量直接取自文件头里的
`label_offset_`，不写死任何魔数。

### 2.4 整体数据流

```
   create vector index ... type=hnsw
            │
            ▼
   ObHNSWAppendPipeline → add_snap_index()
            │  ★ 改造点 1：不再逐行 add_index，而是把向量累积到内存缓冲
            ▼
   check_snap_hnswsq_index() → build_snap_hnsw_index()
            │  ★ 改造点 2：一次性建图
            ▼
   ┌────────────────────────────────────────────┐
   │ GpuHnswBuilder::build()                    │  dlopen 加载的插件
   │   └─ hipVS: CAGRA 建全邻 KNN 图（GPU）      │
   │   └─ hipVS: 生成上层层次（GPU）             │
   │   └─ hnswlib::saveIndex() → 96 字节头 + 元素块 │
   └────────────────────────────────────────────┘
            │  nmslib/hnswlib v0.7.0 字节流
            ▼
   parse_hnsw_serial_header()   # 校验头
   check_hnsw_serial_header()   # 校验 M / maxM0 / 长度
   patch_hnsw_serial_labels()   # label → 行 vid
            ▼
   obvectorutil::deserialize_index()  # VSAG Index::Deserialize
```

### 2.5 为什么 GPU 层次必须用 `HnswHierarchy::GPU`

hipVS 有两种把 CAGRA 图转成 HNSW 的方式，序列化出来的字节流格式一样，但
**入口点的正确性不同**：

* `HnswHierarchy::CPU`：`from_cagra<CPU>` 用 `hnswlib::addPoint()` 逐点插入，
  并置 `base_layer_init = false` 跳过 0 层的建边。这个分支在 `curlevel == 0`
  时会**提前 return**，而 `enterpoint_node_` 是在后面的 else 分支里才被赋值的。
  由于随机层高为 0 的概率是 `1 - 1/M`（M=16 时约 93.75%），入口点很可能
  停留在 `-1`，或者指向一个层高低于 `maxlevel_` 的点——上层遍历会越界。

* `HnswHierarchy::GPU`：`from_cagra<GPU>` 直接分配所有点、按层高做桶排序，
  然后显式设置
  ```cpp
  appr_algo->enterpoint_node_ = order.back();  // 层高最高的那个点
  appr_algo->maxlevel_        = hist.size() - 1;
  ```
  这才是正确的入口点。

所以插件固定使用 `cuvsHnswHierarchy::GPU`。

---

## 3. 实现说明

### 3.1 目录与文件

| 文件 | 作用 |
| --- | --- |
| `src/oblib/lib/vector/gpu/ob_gpu_hnsw_plugin.h` | 插件的纯 C ABI（`ob_gpu_hnsw_build` 等 4 个符号） |
| `src/oblib/lib/vector/gpu/ob_gpu_hnsw_plugin.cu` | 插件实现，用 hipcc 编译，调用 hipVS 的 CAGRA / HNSW C API |
| `src/oblib/lib/vector/gpu/build_plugin.sh` | 用 hipcc 编译插件的脚本 |
| `src/oblib/lib/vector/ob_gpu_hnsw_builder.{h,cpp}` | observer 侧的插件客户端，`dlopen` + `dlsym` |
| `src/oblib/lib/vector/ob_hnsw_serial_format.{h,cpp}` | nmslib/hnswlib 序列化格式的解析、校验与 label 重写 |
| `src/oblib/lib/vector/ob_vsag_adaptor.{h,cpp}` | 新增 `deserialize_index()`，把字节流喂给 VSAG |
| `src/observer/vector_index/ob_plugin_vector_index_adaptor.cpp` | 两个改造点：缓冲与一次性建图 |
| `unittest/oblib/test_hnsw_serial_format.cpp` | 格式解析/校验/重写的单元测试 |
| `src/oblib/lib/vector/gpu/benchmark_hnsw_build.py` | 压测脚本：建表 → 灌数 → 建索引 → 算召回 |
| `src/oblib/lib/vector/gpu/compare_benchmark_results.py` | 对比两次压测结果，输出加速比 |
| `src/oblib/lib/vector/gpu/run_gpu_vs_cpu_benchmark.sh` | 一键跑完 CPU / GPU 两轮并出对比表 |

### 3.2 GPU 代码不进 observer 二进制

插件是独立的 `.so`，运行时用 `dlopen` 加载。这样做的理由：

* seekdb 在没有 AMD GPU 的机器上照样能编译、照样能跑，不需要 ROCm/hipVS 依赖；
* 插件编译失败、加载失败、找不到 GPU、建图失败——**任何一环出问题都自动退回
  原来的 CPU 路径**，对现有部署零影响；
* 插件的 ABI 极小（4 个 C 符号），observer 侧只依赖一个 C 头文件，不引入
  任何 GPU 头文件。

插件查找顺序：`$OB_GPU_HNSW_PLUGIN_PATH`（文件或目录）→ observer 可执行文件同级
目录 → `/usr/local/lib` → `/usr/lib` → 当前目录。

### 3.3 降级策略

| 情况 | 行为 |
| --- | --- |
| 插件不存在 / 加载失败 | 退回 CPU 路径（observer 日志 `[OBGPUHNSW] dlopen plugin failed`） |
| 没有可用的 AMD GPU | 退回 CPU 路径 |
| 索引是 cosine | 不走 GPU 路径 |
| 索引带 extra info 列 | 不走 GPU 路径 |
| 缓冲的向量超过 `OB_GPU_HNSW_MAX_BYTES`（默认 4GB） | 把已缓冲的写入索引，标记 fallback，之后增量构建 |
| GPU 建图或加载失败 | 记 warn 日志，用同一批缓冲数据走 CPU 路径重建 |

**关于 cosine 的排除**：VSAG 对 cosine 索引会自己做归一化
（`hnsw_zparameters.cpp:44` 里 `obj.normalize = true`），而反序列化时它会额外读
`max_elements_ * sizeof(float)` 字节的归一化向量（`DeserializeImpl` 末尾的
`if (normalize_)` 分支）。这些字节 hnswlib 从来不写，所以 GPU 产物无法喂给
cosine 索引。

### 3.4 一次构建完成后不再替换索引

`snap_index_built_` 标志保证：快照索引一旦成型（无论走 GPU 还是 CPU），
后续到达的行只做**增量扩展**，不会再次触发"整体替换"。这避免了增量到达的行
把已建好的全量索引覆盖掉。

### 3.5 指标（metric）的兼容性

| seekdb | VSAG | hipVS / cuVS | 一致性 |
| --- | --- | --- | --- |
| `l2` | `METRIC_TYPE_L2SQR` → `hnswlib::L2Space`（内部是 `L2Sqr`，平方 L2） | `L2Expanded`（平方 L2） | 排序完全一致 ✓ |
| `ip` | `METRIC_TYPE_IP` → `hnswlib::InnerProductSpace`（`1 - IP`） | `InnerProduct`（原始 IP） | `1-IP` 与 `IP` 是单调反序，**最近邻排序一致** ✓ |
| `cosine` | `METRIC_TYPE_COSINE`（VSAG 自己做归一化） | — | 排除，不用 GPU 路径 |

距离值本身不写进索引文件（只存拓扑结构），查询时的距离由 VSAG 用自己的
space 计算，所以 IP 的两种写法差异不影响正确性。

---

## 4. 实验方法

### 4.1 环境

| 项 | 要求 |
| --- | --- |
| GPU | AMD Radeon（MI-series / RX-series），ROCm 6.x |
| hipVS | AMD-Ecosystem/hipVS，`release/rocmds-26.03` 分支 |
| seekdb | Linux，源码编译（Bazel） |

### 4.2 编译插件

```bash
cd src/oblib/lib/vector/gpu
HIPVS_ROOT=/opt/hipvs ./build_plugin.sh
# 产物：libob_gpu_hnsw_plugin.so
sudo cp libob_gpu_hnsw_plugin.so /usr/local/lib/
```

### 4.3 跑对比

```bash
cd src/oblib/lib/vector/gpu
./run_gpu_vs_cpu_benchmark.sh \
    --stop-cmd "obd cluster stop seekdb" \
    --start-cmd "obd cluster start seekdb" \
    --rows 200000 --dim 128 --m 16 --ef-construction 200 \
    --output-dir ./result
```

脚本会：

1. 用 `OB_GPU_HNSW_BUILD=0` 启动 observer，跑一轮压测（label = `cpu`）；
2. 用 `OB_GPU_HNSW_BUILD=1` 重新启动 observer，跑第二轮（label = `gpu`）；
3. 打印对比表并写出 `comparison.json`。

也可以手工分两次跑：

```bash
python3 benchmark_hnsw_build.py --rows 200000 --dim 128 --label cpu --output cpu.json
python3 benchmark_hnsw_build.py --rows 200000 --dim 128 --label gpu --output gpu.json
python3 compare_benchmark_results.py cpu.json gpu.json
```

`benchmark_hnsw_build.py` 做的事：

1. 建表 `t_bench(c1 bigint primary key, c2 vector(dim))`；
2. 生成均匀随机数据集，用 `LOAD DATA LOCAL INFILE` 导入（服务端不支持时自动
   退回批量 `INSERT`）；
3. 计时执行
   `create vector index idx_vec on t_bench (c2) with (distance=l2, lib=vsag, type=hnsw, m=16, ef_construction=200)`；
4. 跑 N 条随机查询，同时用 `order by l2_distance(c2, v) approximate limit k`
   （走索引）和 `order by l2_distance(c2, v) limit k`（精确全扫）各查一次，
   算 top-k 召回率，并统计两条路径的 p50 / p95 延迟；
5. 输出 JSON。

### 4.4 结果表

> **说明：以下表格需要在本人的 AMD GPU 机器上实际跑出来后填入。**
> 本次开发环境为 Windows，无 AMD GPU，无法编译插件也无法运行压测，
> 因此不提供任何未经实测的数字。

| 场景 | rows | dim | m | ef_construction | 建索引耗时 (s) | rows/s | recall@10 | approximate p50 (ms) | exact p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU (基线) | 200000 | 128 | 16 | 200 | *待填* | *待填* | *待填* | *待填* | *待填* |
| GPU (hipVS) | 200000 | 128 | 16 | 200 | *待填* | *待填* | *待填* | *待填* | *待填* |

加速比 = CPU 建索引耗时 / GPU 建索引耗时。

**观测量说明**：

* `build_seconds` 是 `create vector index` 的端到端墙钟时间，包含 DDL 事务、
  tablet 切分、数据读取等开销，不只是纯建图时间。GPU 与 CPU 用同一套流程，
  因此对比是公平的。若要更纯粹的建图时间，可从 observer 日志里取：

  ```
  build hnsw index with cpu success [count] [dim] [seconds]
  build hnsw index with gpu success [count] [dim] [seconds] [max_degree] [graph_degree] [gpu_seconds]
  ```

  两条日志的字段顺序一致，第 3 个参数就是建索引耗时。

* `recall@10` 是 GPU 产物与精确全扫的重合率，用来证明**加速没有牺牲召回**。
  GPU 图的召回通常与 CPU 图相当或略低（CAGRA 的图是近似 KNN 图），若差距
  明显，可用 `OB_GPU_HNSW_BUILD_ALGO=2`（NN-descent）或调大 `ef_construction`
  观察变化。

* `exact p50` 是精确全扫的延迟，量级会远大于 approximate，两者对比可以说明
  索引本身是否生效。

### 4.5 建议的实验矩阵

| 变量 | 取值 |
| --- | --- |
| rows | 100K / 200K / 500K / 1M |
| dim | 128 / 768 |
| m | 16 / 32 |
| build_algo | 0 (auto) / 1 (ivf_pq) / 2 (nn_descent) |

dim 越大、rows 越多，GPU 的优势越明显（距离计算被批量并行）。

---

## 5. 正确性验证

### 5.1 单元测试

`unittest/oblib/test_hnsw_serial_format.cpp` 构造了一份与 nmslib/hnswlib
v0.7.0 布局**逐字节一致**的镜像，覆盖：

* 头部解析（13 个字段全部核对）；
* 缓冲区过短、空指针被拒绝；
* M / maxM0 / 长度不匹配被拒绝；
* label 重写正确，且元素个数不匹配时被拒绝。

```bash
bazel test //unittest/oblib:test_hnsw_serial_format
```

### 5.2 格式兼容性核对

方案不依赖"猜"格式，而是逐字段比对了 hnswlib v0.7.0 的 `saveIndex`/
`loadIndex` 与 VSAG 的 `SerializeImpl`/`DeserializeImpl`，核对项见 2.1 节的表。
其中 hipVS 的 hnswlib 补丁（`cpp/cmake/patches/hnswlib.diff`）只做了模板化
重构（支持 half / int8），**没有触碰 `saveIndex` / `loadIndex`**。

### 5.3 端到端验证

压测脚本同时跑 approximate 和精确全扫，用召回率作为正确性的端到端判据：
如果 VSAG 加载 GPU 产物时出错、或图的拓扑不对，召回率会立刻塌到 0。

---

## 6. 限制与后续工作

1. **未在真实 AMD GPU 上跑过**。本方案的正确性依据是逐字段的格式核对与
   源码级审查，编译与实测需要在目标机器上完成。
2. **只覆盖 `type=hnsw`**。HNSW_SQ / HNSW_BQ（带量化）、HGRAPH（带
   extra info）、IPIVF（稀疏）都还在 CPU 路径上，因为它们的索引格式不是
   hnswlib v0.7.0 那套。
3. **只覆盖 l2 和 ip**，cosine 因为归一化的原因被排除。
4. **内存开销**。GPU 路径需要在内存里缓冲整个 tablet 的向量，预算由
   `OB_GPU_HNSW_MAX_BYTES` 控制（默认 4GB）。超过预算会自动退回增量 CPU 构建。
   后续可以考虑按 tablet 之外再分片。
5. **多 tablet 并发**。多个 tablet 同时建图会同时占用显存，需要在插件里加
   一个信号量限制并发数。当前版本依赖 CAGRA 自身的显存申请，暂未限制。

---

## 7. 复现步骤速查

```bash
# 1. 拉分支
git fetch origin
git checkout feature/gpu-hnsw-build-hipvs

# 2. 编译 seekdb（Linux + Bazel）
./build.sh --release   # 具体命令以仓库文档为准

# 3. 编译插件
cd src/oblib/lib/vector/gpu
HIPVS_ROOT=/opt/hipvs ./build_plugin.sh
sudo cp libob_gpu_hnsw_plugin.so /usr/local/lib/

# 4. 验证插件能被加载（observer 日志里应出现）
#    [OBGPUHNSW] loaded gpu hnsw plugin

# 5. 跑对比
./run_gpu_vs_cpu_benchmark.sh \
    --stop-cmd "obd cluster stop seekdb" \
    --start-cmd "obd cluster start seekdb" \
    --rows 200000 --dim 128
```

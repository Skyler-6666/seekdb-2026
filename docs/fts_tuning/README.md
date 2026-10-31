# 全文索引构建、分词与查询优化

本分支以比赛基线 `5bce9e06d1281347ad61344e4b01fab9158c58a2` 为基础，
优化全文索引的解析、倒排构建和匹配查询热路径。实现不修改测试数据、
benchmark SQL、计时逻辑、评分脚本、固定基线或 CI 入口，也不缓存查询答案。

## 主要改动

### 解析和分词

- IK 字符分类只解码一次，并按原分类优先级识别 ASCII、全角字符、中文、
  其他 CJK 和 UTF-16 surrogate。
- IK 主循环对单字节 ASCII 直接确定长度和字母/数字类别；非 ASCII、非法
  或截断 UTF8 继续使用原解码路径。
- IK 主循环将已取得的字符、长度和类型直接交给固定处理器，避免逐字符
  重建状态和重复分类。
- `ObFTParseHelper` 在相同 parser 下复用 IK/BENG token iterator；每次仍
  读取并完整分词当前文档。parser 元数据使用独立 arena，当前结果词和 JSON
  仍由每次调用的 allocator 产生。
- TOKENIZE 按线程、parser 名称和属性复用 parser 实例，不缓存分词结果。
- TOKENIZE 的线程缓存同时保留 IK 词典加载状态和 loader 句柄；parser、
  字符集或属性变化时失效，避免每次调用重复检查并尝试加载同一词典。
- 服务在打开 SQL 入口前、`dict_cache` 初始化完成后预构建内置 IK 主词典、
  量词词典和停用词词典，避免把首次索引构建中的一次性缓存成本计入 DDL。
- 空 parser 属性直接使用内置默认值；ASCII 停用词、已小写 ASCII 和词频
  聚合分别使用无分配查找、恒等转换快路径和单次哈希更新。大小写检查在
  每个文档内复用字符集和 case table 指针，并对 ASCII 与常见 CJK UTF-8
  字符使用保持回退语义的范围快路径。
- 范围词典二分定位，DAT 词典查找减少临时字符串、除法和重复遍历。

### 索引构建

- IK 与 BENG 解析器在文档间复用，减少每行初始化、析构和临时分配。
- DAT 构建仅在取得第一组 child code 时完整编码；后续冲突候选立即结束。
- FTS DDL 行扫描在严格匹配输出列和本地 DAS 形态时批量输出 token 行，
  其余计划继续使用标量路径。
- DDL 的稳定全文副本使用独立可复用缓冲；行消费完成前不发生覆盖，
  文档增长时只扩容，避免每个文档重复走通用字符串写入路径。
- 固定长度 16 字节字符串和 64 位整数差分解码增加批量快路径。

### MATCH 查询

- 只有相关性值不可观察、自然语言模式、单 MATCH 正向谓词且无加权、排序、
  top-k 消费时，才走无分数倒排合并；布尔模式和分数消费路径保持原实现。
- 小型 DAAT 倒排流按 ID 批量归并并去重；独占的标量 `COUNT(*)` 可只返回
  行数，不物化未使用的结果表达式。
- `COUNT(*)` 消费的单表有限派生查询会移除不可观察的隐式相关性排序，
  并将派生表投影替换为常量，避免仅为计数而回表读取原列。
- 无分数 DAAT 对非空整数文档 ID 直接比较，其他类型继续使用原 Datum
  比较器。
- 无过滤、无聚合、无采样的部分回表请求按实际容量预取，避免只要少量结果
  却先读取完整批次。
- TAAT 哈希桶按分区预计文档数配置，并用 `set_or_update` 合并相关性累加。
- UTF8MB4 排序规则增加 ASCII 权重快路径，保持原有大小写折叠语义。

## 正确性

回归入口位于 `unittest/fts_tuning/`，新增 FTS 用例同时加入既有
`tools/deploy/mysql_test/test_suite/fts_index/`。验证覆盖：

- IK smart/max_word、BENG、space、ngram、ngram2 的完整 token 和词频；
- 中文、英文、全角、Unicode 补充平面、NUL、空串和长文本；
- MATCH 的完整 ID、顺序、浮点分数、COUNT、LIMIT/OFFSET 和布尔模式；
- 索引创建、更新、删除、事务提交/回滚及索引后置创建；
- DAAT 多倒排流、去重、分页、回表容量和缓存回表；
- parser 属性复用、输入生命周期、DAT/DAT 解码边界。

## 复现

```bash
python3 -m unittest discover -s unittest/fts_tuning -p test_validation.py -v
python3 unittest/fts_tuning/official_benchmark.py verify-contract
```

构建并启动隔离实例后，执行原样性能脚本：

```bash
REPORT_FILE=$(mktemp /tmp/fts-bench-report.XXXXXX)
LABEL=after ROWS=20000 BATCH=500 ROUNDS=3000 QUERY_ROUNDS=200 \
  SAMPLES=3 WARMUP=30 SKIP_LOAD=0 OUTPUT="$REPORT_FILE" \
  bash tools/benchmark/fts_large_bench.sh
python3 tools/benchmark/fts_large_bench_score.py "$REPORT_FILE"
```

每次完整测量必须使用新报告文件、相同硬件和资源、相同编译方式及独立
数据库实例。正式成绩以比赛统一环境为准。

# Round-two classifier fast paths

The production change remains limited to `ObFTCharUtil::do_classify`. Decoding
still happens first, using the same decoder and return-code check. The change
then partitions the existing category predicates:

- ASCII A-Z/a-z is English; ASCII 0-9 is Arabic; all other ASCII is USELESS.
- Every code point FF00..FFEF is already OTHER_CJK. Fullwidth A-Z/a-z/0-9 within
  that block retain their earlier English/Arabic categories, using their fixed
  FEE0 offset from ASCII. This branch also preserves punctuation and halfwidth
  Katakana as OTHER_CJK.
- Outside those two ranges, the upstream `is_alpha` and `is_arabic` predicates
  cannot match. Keep the existing Chinese, other-CJK, and UTF16-only surrogate
  predicates and skip the two impossible out-of-line calls.

No dictionary, segmentation rule, token offset, case conversion, or charset
validation is changed. The first-valid-character scan still runs in the caller.
Malformed UTF8 and the current generic UTF16/UTF16LE decoder behavior are
intentionally preserved. If upstream Unicode category ranges change, run the
exhaustive differential again before updating this fast path.

Run from the repository root:

```sh
python3 unittest/fts_tuning/tokenizer_round2_regression.py --build-dir /tmp/tokenizer-round2 --compiler g++
```

```powershell
python unittest/fts_tuning/tokenizer_round2_regression.py --build-dir ../tokenizer-round2 --compiler msvc
```

The default comparison is against round-one commit
`32cf87c11ae460301394f539ae25896748bee73d`. Pass
`--baseline 5bce9e06d1281347ad61344e4b01fab9158c58a2` for the original upstream
classifier. Both commits must exist in the local Git object database.

The existing source-extraction harness compiles the real old/new classifier
bodies, actual UTF8 decoder/helper templates, and actual Unicode predicates in
separate translation units. It compares 8,272,460 cases, including every Unicode
code point and every truncated encoding, every two-byte input, exhaustive
E0..EF three-byte inputs, deterministic random four-byte input, null input,
unsupported charsets, and UTF16/UTF16LE inputs. Independent expected categories
also check overlapping fullwidth ranges. See `tokenizer_README.md` for the exact
dependency-glue boundary.

After correctness passes, five synthetic timing datasets cover all 128 ASCII
values, a spread of basic Chinese characters, the entire FF00..FFEF block,
fullwidth letters/digits, and mixed scripts. Each dataset uses five interleaved
old/new samples of 12,288,000 calls and verifies identical checksums. These are
local classification timings, **not native seekdb, SQL, official benchmarks, or
competition scores**. No Linux server or modified seekdb binary was built by
this harness. Native compilation, SQL/index regression, and the official CI
workload remain required.

The build directory receives `tokenizer_round2_result.json` with every raw
sample, medians, platform/compiler labels, baseline commit, classifier source
hashes, and the differential result. Keep generated artifacts outside the
repository.

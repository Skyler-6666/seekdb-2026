# IK character-classification regression

From the repository root, with Python 3 and a C++17 compiler:

```sh
python3 unittest/fts_tuning/tokenizer_differential.py --build-dir /tmp/seekdb-tokenizer-test --compiler g++
```

Windows with Visual Studio 2022 Community C++ tools:

```powershell
python unittest/fts_tuning/tokenizer_differential.py --build-dir ../tokenizer-build --compiler msvc
```

Add `--benchmark` for an **unofficial character-classification microbenchmark**.
It uses five interleaved old/new timing rounds and verifies equal checksums.
This timing cannot establish TOKENIZE, index-build, MATCH, official benchmark,
or competition-score improvements. The compile uses separate translation units
for Unicode predicates and does not enable link-time optimization.

The script reads the old header from immutable competition baseline
`5bce9e06d1281347ad61344e4b01fab9158c58a2` using `git show`, and the new header
from the working tree. It extracts the actual upstream UTF-8 decoder, charset
helper templates, error values, and Unicode predicate implementations. It does
not substitute a hand-written Unicode decoder or copy category tables. The
generated dependency glue disables logging/branch hints and maps test collation
IDs directly to their charset, so this remains an isolated source-level test.
A real server build, SQL regression, and official Linux benchmark are separate
acceptance requirements.

The test compares return values **and** output types for 8,272,460 cases:

- Every code point 0..0x10FFFF encoded by the real helper, at every prefix length,
  including UTF-8 encodings of surrogate values to preserve malformed-input
  behavior.
- Every 16-bit byte combination at lengths 0..2 under each supported classifier
  charset: UTF8MB4, UTF16, and UTF16LE.
- Every three-byte sequence with lead byte E0..EF; one million deterministic
  four-byte random inputs; explicit Unicode boundary and out-of-range values.
- Null buffers, unsupported charsets, empty buffers, odd UTF16 lengths, and
  UTF16/UTF16LE four-byte inputs.

Independent assertions also check fullwidth letter/digit precedence over the
overlapping CJK block, Chinese/Japanese/Korean categories, emoji, punctuation,
and the existing surrogate handling.

Two baseline behaviors are intentionally retained: the classifier treats only
negative decoder returns as errors (`OB_CS_ILSEQ` is zero), and the current
charset helper has no specialized UTF16/UTF16LE Unicode decoder. The test uses
those actual upstream helpers; it does not claim to validate an idealized UTF16
implementation. No segmentation rules, dictionary, case conversion, token
frequency, offset, or SQL behavior is modified by the optimization.

Generated sources, executable, and `tokenizer_result.json` are written only to
`--build-dir`. The JSON records both source hashes and labels its limited scope.
Keep this build directory outside the repository when preparing a patch.

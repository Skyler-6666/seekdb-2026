# BENG normalization regression

`english_regression.py` compares the complete production English normalizer
against commit `32cf87c11ae460301394f539ae25896748bee73d`. Run in a C++17 compiler
environment, from the repository root:

```sh
python unittest/fts_tuning/english_regression.py --build-dir ../english-test-build
```

On Windows, activate the Visual Studio x64 Native Tools environment first.
The repository must contain the baseline commit. The build directory contains
generated source and an executable; neither belongs in a source submission.

The candidate borrows the trimmed input token only for UTF8MB4 tokens containing
ASCII bytes without `A` through `Z`. All other tokens use the unchanged charset
conversion. The extra ASCII scan is intentional; whether this improves the
official workload still requires a native benchmark.

Ownership was traced through the actual consumers:

- `ObTextWhitespaceTokenizer` returns slices of the document. Its next call
  advances offsets without changing document bytes.
- `ObBEngFTParser::get_next_token` continues to allocate and copy each result
  into the parser caller's allocator before requesting another token.
- `ObTextTokenGroupNormalizer::build_grouping_map` copies each first unique key
  into its own allocator. Repeated keys update only the frequency.
- A normalizer's previous output already becomes invalid at its next call,
  which reuses the normalization allocator; reset/reuse do not extend that
  lifetime. No retaining consumer is allowed to rely on a borrowed span later.

The regression extracts the real old/new normalizer functions, real
`ObCharset::tolower`, real UTF8 decoder/encoder and lowercase loop, upstream
Unicode case tables and ctype table, and the real BENG/grouping consumer
functions. It exercises 99,856 byte/frequency cases, all ASCII two-byte
combinations in a token, all 256 middle-byte values, embedded NUL/control bytes,
uppercase and multibyte/invalid UTF8, empty and not-initialized states, mixed
allocator reuse, consumer ownership, reset and failure propagation.

For a synthetic 10,000-token stream containing 8,000 eligible tokens, the
normalizer allocation count is 10,000 before and 2,000 after. BENG's stable-copy
allocation remains in place. These are adapter operation counts, not elapsed
time, peak memory measurements, or an official score.

String/datum, allocator, upstream stream, and map boundaries use portable
adapters. UTF8 case conversion uses extracted upstream code; other encodings
only verify that the original conversion path is selected. Actual collation
dispatch, other charset conversion semantics, SQL results, native allocations,
Linux compilation and official benchmark performance remain to be verified.

# Native regression coverage for the later FTS optimizations

`native_extended_regression.py` captures complete typed results from a real
seekdb server. Run the same file once against the official baseline and once
against the candidate, using the same server configuration. Instances can run
sequentially, which avoids concurrent benchmark load on one machine.

```bash
python3 unittest/fts_tuning/native_extended_regression.py capture --port 38680 --output /tmp/fts-before-new.json
# Stop the baseline instance and start the candidate on the same test port.
python3 unittest/fts_tuning/native_extended_regression.py capture --port 38680 --output /tmp/fts-after-new.json
python3 unittest/fts_tuning/native_extended_regression.py compare /tmp/fts-before-new.json /tmp/fts-after-new.json
```

The Python dependency is PyMySQL. If authentication is configured, supply the
password through `FTS_TEST_PASSWORD`, never as a command-line argument. The
runner creates and drops only its uniquely named `fts_extended_<uuid>` database.
Use a dedicated test instance. Output paths must not already exist.

Coverage includes all ASCII bytes, empty/NULL values and Unicode comparisons,
sorting/grouping under general, binary and Unicode collations; complete raw
TOKENIZE JSON for five parsers with case, accents, fullwidth letters, NUL,
supplementary characters, long input and IK modes; and three real fulltext
indexes with full document identities, exact scores, natural/Boolean MATCH,
thresholds, compound predicates, score ordering, aggregation, DISTINCT and LIMIT.
No score rounding or token sorting is performed. Differences are failures to
investigate, not automatically discarded as timing noise. SQL errors abort the
capture; warning rows are retained and compared. These are correctness checks,
not official performance measurements or evidence of a competition score.

`test_parser_property_reuse.cpp` is a focused GoogleTest of the production
`ObFTParserProperty` API. It checks that implicit IK defaults agree with explicit
smart mode and that parsing empty properties after max_word restores those
defaults. The first test fails on the previous v6 implementation, which leaves
the prior mode unchanged. Compile/link this test against the same production
objects and dependencies as the tested seekdb build, with GoogleTest. This
repository snapshot has no native FTS unit-test CMake target, so no existing
target was silently repurposed. Record the actual build/test command with the
validation evidence; merely adding this source is not a claim that it ran.

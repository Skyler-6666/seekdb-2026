# One-probe grouped insertion/update

`ObAddWord::groupby_word` now uses the existing `ObHashMap::set_or_update`
interface instead of `atomic_refactored` followed by `set_refactored` on a miss.
The initial value is still **one**, even when the first input token frequency is
larger. Existing entries still add the supplied `word_freq` and retain the latest
collation-equivalent spelling. Neither the existing node's bucket-chain position
nor the underlying arena token lifetime changes. The callback only assigns the
existing key and frequency; it does not call back into the map.

The real map implementation initializes a temporary pair before delegating to
the hash table. Its three-argument forwarder supplies the hash table's identity
pair preprocessor; the callback receives the full `HashMapPair&`. A single hash
and bucket traversal either update the node or call `internal_set`, which
prepends a new node and propagates allocation failure. The extra temporary pair
copies the actual `ObFTWord` shallowly; it does not copy or free token bytes.

Run the regression from an activated C++17 compiler environment:

```sh
python3 unittest/fts_tuning/grouping_regression.py --build-dir /tmp/grouping-test --cxx g++
```

```powershell
python unittest/fts_tuning/grouping_regression.py --build-dir ../grouping-test --cxx cl
```

The harness extracts all three production grouping bodies: official baseline
`5bce9e06d1281347ad61344e4b01fab9158c58a2`, the previous submission
`32cf87c11ae460301394f539ae25896748bee73d`, and the working tree. It also extracts
the real pair, identity preprocessor, atomic/upsert map/hash-table forwarders,
bucket traversal bodies, and `internal_set`. The allocator, locks, word storage,
hash/equality, and ordinary baseline get/set are explicitly portable adapters.
They cannot establish native charset, concurrency, or SQL correctness.

Tests compare 36,000 three-way prefixes, independent frequency/spelling oracles,
bucket order under collisions, invalid arguments, hash failure, allocation
failure on a new key, allocator-free duplicate update, missing allocator and
uninitialized buckets. Two 10,000-token operation-count workloads cover 32
distinct words and all 10,000 distinct words. If N is token count and U is unique
word count, the observed adapter counts are:

| Implementation | Hash calls | Hash-node allocations |
| --- | ---: | ---: |
| Official get + overwrite set | 2N | N |
| Previous atomic + miss insert | N + U | U |
| Current set_or_update | N | U |

These counts follow the extracted control flow. They are not elapsed-time
performance measurements or official scores; the adapter uses `std::string`
instead of the native shallow `ObFTWord`, so timing it would misrepresent native
copy cost. The build directory contains `grouping_result.json`, raw success and
count output, baseline IDs, platform/compiler labels, and source hashes. Native
seekdb compilation, collation/SQL regression and official CI remain required.

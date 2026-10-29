# Bridge buffer fault-injection test

From the repository root, with a C compiler:

```bash
gcc -std=c11 -Wall -Wextra -Werror \
  -Iunittest/oblib/vector/hipvs_bridge_mock \
  unittest/oblib/vector/test_hipvs_bridge_buffers.c \
  -o /tmp/test_hipvs_bridge_buffers
/tmp/test_hipvs_bridge_buffers
```

The shim tests production bridge allocation ownership and error propagation on
CPU. It does not test the real cuVS ABI, GPU execution, recall, or performance.
Do not add this include directory to a product build.

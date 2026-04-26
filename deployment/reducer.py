#!/usr/bin/env python3
import sys

# Identity Reducer
# The mapper already emits the final JSON representation of the processed page.
# The reducer simply collects these and outputs them.
if __name__ == "__main__":
    for line in sys.stdin:
        # line is already a JSON string emitted by the mapper
        print(line.strip())

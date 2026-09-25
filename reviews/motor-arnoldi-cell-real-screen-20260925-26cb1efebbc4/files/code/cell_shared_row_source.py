"""Generate a shared-row storage variant of the unchanged FP64 cell equations."""
from __future__ import annotations

import re


def shared_row_source(original):
    start = original.index('__device__ __noinline__ void warp_midpoint')
    end = original.index('__device__ void commit_event', start)
    segment = original[start:end]
    declaration = 'double A[17],original[17],b,br'
    if segment.count(declaration) != 1:
        raise ValueError('unexpected solve declaration')
    segment = segment.replace(declaration,
                              '__shared__ double A_shared[17*17],original_shared[17*17];double b,br')
    segment, count_a = re.subn(r'\bA\[([^\]]+)\]', r'A_shared[i*17+\1]', segment)
    segment, count_o = re.subn(r'\boriginal\[([^\]]+)\]',
                               r'original_shared[i*17+\1]', segment)
    if (count_a, count_o) != (11, 3) or 'double A[17]' in segment:
        raise ValueError('unexpected solve access count')
    return original[:start] + segment + original[end:]

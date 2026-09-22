"""Finite volumes on an explicit subset of known, unchanged PN labels.

An omitted same-label neighbour is an open conductor face, including INSIDE
the downloaded array. It must never become a membrane wall. This permits
irregular domain extensions without changing segmentation or inventing a
voltage condition on the remaining interface. No CNS admission is implied.
"""
import numpy as np
from neck_distributed_conductor import voxel_topology


def selected_voxel_topology(labels, selected, ownership, *, halo_mask):
    """Reuse the full-volume enumerator, then restore omitted interior faces.

    Every array-edge neighbour must be known from the original one-cell halo.
    Returned node order remains ascending flat voxel order. Outer faces can
    lie inside the context: callers must match physical faces explicitly,
    rather than pass them to a box-plane-only surface graph.
    """
    labels = np.asarray(labels); selected = np.asarray(selected)
    halo = np.asarray(halo_mask)
    if (labels.ndim != 3 or labels.dtype != np.bool_
            or selected.dtype != np.bool_ or selected.shape != labels.shape
            or not selected.any() or np.any(selected & ~labels)
            or halo.dtype != np.bool_
            or halo.shape != tuple(np.array(labels.shape)+2)
            or not np.array_equal(halo[1:-1, 1:-1, 1:-1], labels)):
        raise ValueError('Explicit selected PN cells and exact original label halo required')
    temporary_halo = halo.copy()
    temporary_halo[1:-1, 1:-1, 1:-1] = selected
    t = voxel_topology(selected, ownership, halo_mask=temporary_halo)
    del temporary_halo
    omitted = labels & ~selected
    nodes = [t['outer_nodes']]; axes = [t['outer_axis']]; signs = [t['outer_sign']]
    for axis in range(3):
        for sign in (-1, 1):
            here = [slice(None)]*3; there = here.copy()
            here[axis] = slice(1, None) if sign < 0 else slice(None, -1)
            there[axis] = slice(None, -1) if sign < 0 else slice(1, None)
            hit = selected[tuple(here)] & omitted[tuple(there)]
            xyz = np.column_stack(np.nonzero(hit)); del hit
            if sign < 0: xyz[:, axis] += 1
            flat = np.ravel_multi_index(xyz.T, labels.shape)
            ni = np.searchsorted(t['flat_voxel'], flat)
            assert np.array_equal(t['flat_voxel'][ni], flat)
            # One face per node for this axis/sign. No unsigned underflow.
            assert np.all(t['membrane_face_count'][ni] > 0)
            t['membrane_face_count'][ni] -= 1
            nodes.append(ni.astype(np.int32)); axes.append(np.full(len(ni), axis, np.int8))
            signs.append(np.full(len(ni), sign, np.int8))
    t['outer_nodes'] = np.concatenate(nodes)
    t['outer_axis'] = np.concatenate(axes)
    t['outer_sign'] = np.concatenate(signs)
    t['outer_faces'] = len(t['outer_nodes'])
    t['explicit_selected_domain'] = True
    return t

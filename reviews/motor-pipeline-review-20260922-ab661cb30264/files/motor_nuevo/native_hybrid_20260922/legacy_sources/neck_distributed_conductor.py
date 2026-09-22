"""Experimental distributed intracellular conductor on unchanged cubic labels.

Cell-centred finite volumes, isotropic cytoplasm, staircase wall membrane.
An ownership boundary is a bookkeeping cut: its shared faces conduct exactly
once and carry NO membrane. No assignment to a cable point is made here.
Acquisition faces require explicit, finite boundary potentials for assembly;
zero-flux closure is never inferred. This candidate is outside the CNS.
"""
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix, diags


@njit(cache=False)
def _enumerate(halo, ids, owner, nx, ny, nz, edge, cut, wall, outer_node, outer_axis, outer_sign):
    ne = nc = no = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                flat = (x*ny+y)*nz+z
                i = ids[flat]
                if i < 0:
                    continue
                for axis in range(3):
                    coord = x if axis == 0 else y if axis == 1 else z
                    length = nx if axis == 0 else ny if axis == 1 else nz
                    stride = ny*nz if axis == 0 else nz if axis == 1 else 1
                    for sign in (-1, 1):
                        if coord+sign < 0 or coord+sign >= length:
                            hx, hy, hz = x+1, y+1, z+1
                            if axis == 0: hx += sign
                            elif axis == 1: hy += sign
                            else: hz += sign
                            if not halo[hx, hy, hz]:
                                if len(wall): wall[i] += 1
                                continue
                            if len(outer_node):
                                outer_node[no] = i; outer_axis[no] = axis; outer_sign[no] = sign
                            no += 1
                            continue
                        j = ids[flat+sign*stride]
                        if j < 0:
                            if len(wall): wall[i] += 1
                        elif sign == 1:
                            if len(edge): edge[ne, 0] = i; edge[ne, 1] = j
                            if owner[i] != owner[j]:
                                if len(cut): cut[nc] = ne
                                nc += 1
                            ne += 1
    return ne, nc, no


def voxel_topology(mask, ownership, *, halo_mask=None):
    """Own every labelled cell exactly once; preserve all face connections.

    The mask is the ENTIRE downloaded context, or an explicitly declared
    subwindow. Labels outside its array bounds are unknown, not membrane,
    unless a one-cell halo explicitly supplies the original neighbouring
    labels. A same-label halo neighbour remains an open conductor face.
    """
    mask = np.asarray(mask); own = np.asarray(ownership)
    if (mask.ndim != 3 or min(mask.shape) < 1 or mask.dtype != np.bool_
            or own.shape != mask.shape or own.dtype.kind not in 'iu'
            or not mask.any() or np.any(own[mask] < 0)):
        raise ValueError('Nonempty original Boolean labels and complete integer ownership required')
    if halo_mask is None:
        halo = np.pad(mask, 1, constant_values=True)
    else:
        halo = np.asarray(halo_mask)
        if (halo.dtype != np.bool_ or halo.shape != tuple(np.array(mask.shape)+2)
                or not np.array_equal(halo[1:-1, 1:-1, 1:-1], mask)):
            raise ValueError('Halo must contain the exact original interior labels')
    n = int(mask.sum())
    if n >= np.iinfo(np.int32).max:
        raise ValueError('Context exceeds explicit int32 node bound')
    flat = np.flatnonzero(mask).astype(np.int64)
    ids = np.full(mask.size, -1, np.int32); ids[flat] = np.arange(n, dtype=np.int32)
    owners = own.ravel()[flat].copy()
    args = (halo, ids, owners, *mask.shape)
    empty = np.empty(0, np.int32)
    ne, nc, no = _enumerate(*args, np.empty((0, 2), np.int32), empty,
                            np.empty(0, np.uint8), empty, empty, empty)
    edges = np.empty((ne, 2), np.int32); cuts = np.empty(nc, np.int64)
    wall = np.zeros(n, np.uint8); outer = np.empty(no, np.int32)
    axis = np.empty(no, np.int8); sign = np.empty(no, np.int8)
    assert _enumerate(*args, edges, cuts, wall, outer, axis, sign) == (ne, nc, no)
    return dict(shape=mask.shape, flat_voxel=flat, ownership=owners, edges=edges,
                interface_edge_indices=cuts, membrane_face_count=wall,
                outer_nodes=outer, outer_axis=axis, outer_sign=sign,
                voxel_count=n, outer_faces=no, interfaces=nc)


def physical_terms(topology, *, voxel_um, Ri_ohm_cm, Rm_ohm_cm2, Cm_uF_cm2):
    if any(not np.isfinite(x) or x <= 0 for x in [voxel_um, Ri_ohm_cm, Rm_ohm_cm2, Cm_uF_cm2]):
        raise ValueError('Positive physical coefficients required')
    area = topology['membrane_face_count'].astype(float)*voxel_um**2
    return dict(g_nS=1e5*voxel_um/Ri_ohm_cm, wall_area_um2=area,
                C_nF=area*Cm_uF_cm2*1e-5, leak_nS=area*10/Rm_ohm_cm2)


@njit(cache=False)
def _flux(edges, g, voltage, cut_indices):
    """Two independently accumulated partitions and the undivided operator."""
    full = np.zeros(len(voltage)); local = np.zeros(len(voltage)); exchange = np.zeros(len(voltage))
    k = 0; dissipation = 0.; interface_power = 0.
    for e in range(len(edges)):
        i, j = edges[e]; dv = voltage[i]-voltage[j]; current = g*dv
        full[i] += current; full[j] -= current; dissipation += current*dv
        if k < len(cut_indices) and e == cut_indices[k]:
            exchange[i] += current; exchange[j] -= current
            interface_power += current*dv; k += 1
        else:
            local[i] += current; local[j] -= current
    return full, local, exchange, dissipation, interface_power


def partition_flux(topology, g_nS, voltage_mV):
    v = np.asarray(voltage_mV, dtype=float)
    if v.shape != (topology['voxel_count'],) or not np.isfinite(v).all() or not np.isfinite(g_nS) or g_nS <= 0:
        raise ValueError('Finite voltage on every cell and positive conductance required')
    return _flux(topology['edges'], g_nS, v, topology['interface_edge_indices'])


def assemble(topology, terms, *, outer_potential_mV, partitioned=False):
    """Explicit Dirichlet faces at half a cell from its centre (2*g).

    Returns G and boundary RHS. These supplied potentials are boundary data,
    NOT measured PN voltages. For unknown boundary data, keep the open
    topology and do not call this function. The local/partitioned path adds
    the cut-face Laplacian separately, without membrane, for verification.
    """
    boundary = np.asarray(outer_potential_mV, dtype=float)
    if boundary.shape != (topology['outer_faces'],) or not np.isfinite(boundary).all():
        raise ValueError('Every acquisition face requires an explicit finite potential')
    n = topology['voxel_count']; edges = topology['edges']; g = float(terms['g_nS'])
    if (not np.isfinite(g) or g <= 0 or np.asarray(terms['C_nF']).shape != (n,)
            or np.asarray(terms['leak_nS']).shape != (n,)):
        raise ValueError('Physical terms do not match the conductor')
    def lap(es):
        i, j = es.T
        degree = np.bincount(np.r_[i, j], minlength=n)
        off = coo_matrix((np.full(2*len(i), -g), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
        return off+diags(g*degree)
    if partitioned:
        keep = np.ones(len(edges), bool); keep[topology['interface_edge_indices']] = False
        L = lap(edges[keep])+lap(edges[~keep])
    else:
        L = lap(edges)
    outer = topology['outer_nodes']
    clamp = 2*g*np.bincount(outer, minlength=n)
    rhs = 2*g*np.bincount(outer, weights=boundary, minlength=n)
    return (L+diags(terms['leak_nS']+clamp)).tocsr(), rhs


def boundary_centres(topology):
    p = np.column_stack(np.unravel_index(topology['flat_voxel'][topology['outer_nodes']], topology['shape'])).astype(float)+.5
    p[np.arange(len(p)), topology['outer_axis']] += .5*topology['outer_sign']
    return p

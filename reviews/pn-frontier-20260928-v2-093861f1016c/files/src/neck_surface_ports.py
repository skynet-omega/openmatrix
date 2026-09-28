"""Experimental complete-surface coupling to explicitly ordered cable ports.

Harmonic coordinates are a PROVISIONAL boundary voltage interpolation, not a
measurement of a 3-D cross-section. Label witnesses can justify connectivity
of the interpolation graph; they do not determine its electrical weights.
The conjugate current map is the transpose of the voltage map. The coupling
is a positive semidefinite congruence, possibly with positive off-diagonals;
it must not be passed as an ordinary resistor graph to an M-matrix validator.
No membrane, synapse, CNS admission, or physiology is inferred here.
"""
import hashlib
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, hstack
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


def surface_graph(shape, flat_voxel, axis, sign):
    """Face neighbours on box planes, and folds sharing the SAME inner voxel.

    A diagonal pair on one plane is never connected. Outside corner labels
    are not needed: the fold passes through an explicitly shared inner cell.
    No path through another label or unknown padded corner is introduced.
    """
    shape = np.asarray(shape); f = np.asarray(flat_voxel)
    axis = np.asarray(axis); sign = np.asarray(sign)
    if (shape.shape != (3,) or shape.dtype.kind not in 'iu' or np.any(shape < 1)
            or f.ndim != 1 or f.dtype.kind not in 'iu' or not len(f)
            or axis.shape != f.shape or sign.shape != f.shape
            or axis.dtype.kind not in 'iu' or sign.dtype.kind not in 'iu'
            or not np.isin(axis, [0, 1, 2]).all() or not np.isin(sign, [-1, 1]).all()
            or np.any(f < 0) or np.any(f >= np.prod(shape))):
        raise ValueError('Explicit unique box faces required')
    xyz = np.column_stack(np.unravel_index(f, shape))
    if (np.any(xyz[np.arange(len(f)), axis] != np.where(sign < 0, 0, shape[axis]-1))
            or len(np.unique(np.column_stack([f, axis, sign]), axis=0)) != len(f)):
        raise ValueError('Duplicate or non-boundary face')
    batches = []
    for a in range(3):
        other = [i for i in range(3) if i != a]
        for s in [-1, 1]:
            chosen = np.flatnonzero((axis == a) & (sign == s))
            ids = np.full(shape[other], -1, np.int64)
            ids[tuple(xyz[chosen][:, other].T)] = chosen
            for d in range(2):
                left = [slice(None)]*2; right = left.copy()
                left[d] = slice(0, -1); right[d] = slice(1, None)
                i = ids[tuple(left)]; j = ids[tuple(right)]; hit = (i >= 0) & (j >= 0)
                batches.append(np.column_stack([i[hit], j[hit]]))
    order = np.argsort(f, kind='stable')
    # Every pair of incident faces shares the inner cell, including all
    # three folds of a box corner. A chain would depend on sort tie order.
    largest = int(np.unique(f, return_counts=True)[1].max())
    for step in range(1, largest):
        hits = np.flatnonzero(f[order][step:] == f[order][:-step])
        batches.append(np.column_stack([order[hits], order[hits+step]]))
    edges = np.concatenate(batches)
    graph = coo_matrix((np.ones(len(edges)), edges.T), shape=(len(f), len(f))).tocsr()
    _, components = connected_components(graph, directed=False)
    return edges, components


def harmonic_coordinates(face_count, edges, weights, seed_faces):
    """Dirichlet shape functions on a declared, fully seeded surface graph.

    Unitless graph weights specify an interpolation prior, NOT cytoplasmic
    conductance. Separate seeds retain distinct columns even in one surface
    component. Unseeded components and overlapping terminal seeds fail closed.
    """
    e = np.asarray(edges); w = np.asarray(weights, dtype=float)
    if (type(face_count) is not int or face_count < 1 or e.ndim != 2 or e.shape[1] != 2
            or e.dtype.kind not in 'iu' or np.any(e < 0) or np.any(e >= face_count)
            or np.any(e[:, 0] == e[:, 1]) or w.shape != (len(e),)
            or not np.isfinite(w).all() or np.any(w <= 0) or not len(seed_faces)):
        raise ValueError('Invalid explicit interpolation graph')
    seed_column = np.full(face_count, -1, np.int64)
    for col, values in enumerate(seed_faces):
        values = np.asarray(values)
        if (values.ndim != 1 or not len(values) or values.dtype.kind not in 'iu'
                or np.any(values < 0) or np.any(values >= face_count)
                or len(np.unique(values)) != len(values) or np.any(seed_column[values] >= 0)):
            raise ValueError('Port seeds missing, repeated, or overlapping')
        seed_column[values] = col
    i, j = e.T
    graph = coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])),
                       shape=(face_count, face_count)).tocsr()
    count, components = connected_components(graph, directed=False)
    boundary = np.flatnonzero(seed_column >= 0)
    if len(np.unique(components[boundary])) != count:
        raise ValueError('Unseeded surface: do not seal or assign by proximity')
    L = diags(np.asarray(graph.sum(1)).ravel())-graph
    P = np.zeros((face_count, len(seed_faces)))
    P[boundary, seed_column[boundary]] = 1.
    inner = np.flatnonzero(seed_column < 0)
    residual = 0.
    if len(inner):
        A = L[inner][:, inner].tocsc(); rhs = -L[inner][:, boundary]@P[boundary]
        P[inner] = splu(A).solve(rhs)
        residual = float(np.max(abs(A@P[inner]-rhs)))
    unity = float(np.max(abs(P.sum(1)-1)))
    if unity > 1e-8 or P.min() < -1e-10 or P.max() > 1+1e-10 or not np.isfinite(P).all():
        raise ValueError('Interpolation solve failed; no silent renormalization')
    return P, dict(partition_of_unity_error=unity, maximum_residual=residual,
                   minimum_weight=float(P.min()), maximum_weight=float(P.max()),
                   seeded_components=count, faces=face_count, ports=len(seed_faces),
                   normalized_or_clipped=False, biological_admission=False)


class SurfacePortCoupler:
    """I_faces=g*(S*v-P*u); I_cells=S.T*I_faces; I_ports=-P.T*I_faces.

    Positive current leaves its own subsystem. Coupling loss is nonnegative;
    terminal power exchange equals this loss. g is the FV half-cell value
    (2*sigma*h), not a fitted port gain. Ports have no artificial membrane.
    All face rows must be provided, in a persisted explicit ordering.
    """
    def __init__(self, volume_node_count, outer_nodes, voltage_map, face_g_nS):
        nodes = np.asarray(outer_nodes); P = np.asarray(voltage_map, dtype=float)
        g = np.asarray(face_g_nS, dtype=float)
        if (type(volume_node_count) is not int or volume_node_count < 1
                or nodes.ndim != 1 or nodes.dtype.kind not in 'iu' or not len(nodes)
                or np.any(nodes < 0) or np.any(nodes >= volume_node_count)
                or P.ndim != 2 or P.shape[0] != len(nodes) or P.shape[1] < 1
                or not np.isfinite(P).all() or P.min() < -1e-10
                or np.max(abs(P.sum(1)-1)) > 1e-8
                or np.any(np.max(P, axis=0) < 1e-12)
                or g.shape != (len(nodes),) or not np.isfinite(g).all() or np.any(g <= 0)):
            raise ValueError('Complete conservative face map and positive half-cell conductance required')
        self.n = volume_node_count; self.nodes = nodes.astype(np.int64, copy=True)
        self.P = P.copy(); self.g = g.copy()
        h = hashlib.sha256(str((self.n, self.P.shape)).encode())
        for value in [self.nodes, self.P, self.g]:
            h.update(value.tobytes()); value.flags.writeable = False
        self.identity = h.hexdigest()

    def flux(self, volume_mV, port_mV):
        v = np.asarray(volume_mV, dtype=float); u = np.asarray(port_mV, dtype=float)
        if (v.shape != (self.n,) or u.shape != (self.P.shape[1],)
                or not np.isfinite(v).all() or not np.isfinite(u).all()):
            raise ValueError('Finite complete volume and port voltages required')
        gap = v[self.nodes]-self.P@u; q = self.g*gap
        return dict(cell_outward_pA=np.bincount(self.nodes, weights=q, minlength=self.n),
                    port_outward_pA=-self.P.T@q, face_outward_pA=q,
                    dissipated_pA_mV=float(np.dot(q, gap)))

    def matrix(self):
        """Positive semidefinite congruence; for bounded explicit assemblies."""
        T, g = self.congruence()
        return (T.T@diags(g)@T).tocsr()

    def congruence(self):
        """Factor certificate: no eigenvalue fit or M-matrix assumption."""
        S = coo_matrix((np.ones(len(self.nodes)), (np.arange(len(self.nodes)), self.nodes)),
                       shape=(len(self.nodes), self.n)).tocsr()
        T = hstack([S, -csr_matrix(self.P)], format='csr')
        return T, self.g.copy()


def exterior_collar(split, selected_source_edges, original_ports, *, voxel_um,
                    Ri_ohm_cm, Rm_ohm_cm2, Cm_uF_cm2):
    """Keep exterior pieces of wholly retired frusta, using exact split IDs.

    Boundary points coinciding with original ports share that node. Each
    physical collar area is included once; its original half-edge membrane
    must already have been withdrawn by withdraw_source_block in the caller.
    """
    from dm1_native_passive import frustum_parameters
    from neck_multiport import laplacian
    original_ports = np.asarray(original_ports)
    if (original_ports.ndim != 1 or not len(original_ports) or original_ports.dtype.kind not in 'iu'
            or len(np.unique(original_ports)) != len(original_ports)
            or np.any(original_ports < 0) or np.any(original_ports >= len(split['xyz']))):
        raise ValueError('Unique valid original ports required')
    selected = np.isin(split['source_edge'], selected_source_edges)
    inside = np.zeros(len(selected), bool); inside[split['removed_edges']] = True
    if not selected[inside].all():
        raise ValueError('A volume-owned source edge was not retired')
    keep = selected & ~inside
    cuts = np.array([c['node'] for c in split['crossings']], dtype=np.int64)
    nodes = np.unique(np.r_[split['edges'][keep].ravel(), original_ports, cuts])
    edges = np.searchsorted(nodes, split['edges'][keep])
    if (original_ports.ndim != 1 or original_ports.dtype.kind not in 'iu'
            or len(np.unique(original_ports)) != len(original_ports)
            or not np.isfinite(voxel_um) or voxel_um <= 0):
        raise ValueError('Unique ordered original ports and physical scale required')
    if any(not np.isfinite(x) or x <= 0 for x in [Ri_ohm_cm, Rm_ohm_cm2, Cm_uF_cm2]):
        raise ValueError('Positive collar coefficients required')
    # A crossing exactly at an original endpoint has no exterior frustum;
    # it is an algebraic alias, not an isolated membrane-bearing cable node.
    C = np.zeros(len(nodes)); leak = C.copy(); area = 0.
    if len(edges):
        occupied = np.unique(edges); active_nodes = nodes[occupied]
        terms = frustum_parameters(split['xyz'][active_nodes]*voxel_um,
            split['radius'][active_nodes]*voxel_um, np.searchsorted(occupied, edges),
            Rm_ohm_cm2=Rm_ohm_cm2, Cm_uF_cm2=Cm_uF_cm2, Ri_ohm_cm=Ri_ohm_cm)
        C[occupied] = terms['C_nF']; leak[occupied] = terms['leak_nS']
        axial = terms['axial_nS']; area = float(terms['edge_area_um2'].sum())
    else:
        axial = np.empty(0)
    G = laplacian(len(nodes), edges, axial)+diags(leak)
    return dict(G_nS=G.tocsr(), C_nF=C, original_split_nodes=nodes,
                source_port_nodes=original_ports.copy(),
                source_port_indices=np.searchsorted(nodes, original_ports),
                surface_port_indices=np.searchsorted(nodes, cuts), surface_port_nodes=cuts,
                source_edges=split['source_edge'][keep], intervals=split['source_interval'][keep],
                area_um2=area, cut_membrane_area_um2=0.)


class CoupledVolumeCollar:
    """Actual volume membrane + distributed interface + exterior cable collar.

    Full-volume action avoids a dense/sparse factorization of millions of
    unknowns. This is the local patch ONLY: an exterior source cable must have
    its old block withdrawn before sharing source_port_indices with this patch.
    The anatomical/provisional status of P remains the caller's responsibility.
    """
    def __init__(self, topology, terms, voltage_map, collar):
        self.t = topology; self.terms = terms; self.collar = collar
        self.n = int(topology['voxel_count']); self.nc = len(collar['C_nF'])
        p = np.asarray(collar['surface_port_indices']); mapping = np.asarray(voltage_map)
        if (mapping.ndim != 2 or p.shape != (mapping.shape[1],) or p.dtype.kind not in 'iu'
                or np.any(p < 0) or np.any(p >= self.nc) or len(np.unique(p)) != len(p)
                or np.asarray(terms['C_nF']).shape != (self.n,)
                or np.asarray(terms['leak_nS']).shape != (self.n,)
                or any(not np.isfinite(a).all() or np.any(a < 0)
                       for a in [terms['C_nF'], terms['leak_nS'], collar['C_nF']])):
            raise ValueError('Volume, membrane, collar, and ordered surface ports mismatch')
        self.cp = p.copy()
        self.interface = SurfacePortCoupler(self.n, topology['outer_nodes'], voltage_map,
                                            np.full(topology['outer_faces'], 2*terms['g_nS']))
        self.C_nF = np.r_[terms['C_nF'], collar['C_nF']]
        self.source_port_indices = self.n+collar['source_port_indices']

    def action(self, voltage_mV):
        from neck_distributed_conductor import partition_flux
        v = np.asarray(voltage_mV, dtype=float)
        if v.shape != self.C_nF.shape or not np.isfinite(v).all():
            raise ValueError('Complete coupled patch voltage required')
        a = v[:self.n]; b = v[self.n:]
        axial = partition_flux(self.t, self.terms['g_nS'], a)[0]
        face = self.interface.flux(a, b[self.cp])
        out = axial+self.terms['leak_nS']*a+face['cell_outward_pA']
        collar = np.asarray(self.collar['G_nS']@b).ravel()
        np.add.at(collar, self.cp, face['port_outward_pA'])
        return np.r_[out, collar]

    def matrix(self):
        """Explicit assembly for small reference problems, without double clamps."""
        base, (T, g) = self.base_and_congruence()
        return (base+T.T@diags(g)@T).tocsr()

    def base_and_congruence(self):
        """Resistor base and PSD surface certificate for DynamicPortHandoff."""
        from neck_distributed_conductor import assemble
        from scipy.sparse import block_diag
        bulk, _ = assemble(self.t, self.terms,
                           outer_potential_mV=np.zeros(self.t['outer_faces']))
        bulk -= diags(2*self.terms['g_nS']*
                      np.bincount(self.t['outer_nodes'], minlength=self.n))
        base = block_diag([bulk, self.collar['G_nS']], format='csr')
        T, g = self.interface.congruence(); T = T.tocoo()
        mapping = np.r_[np.arange(self.n), self.n+self.cp]
        mapped = coo_matrix((T.data, (T.row, mapping[T.col])),
                            shape=(T.shape[0], base.shape[0])).tocsr()
        return base, (mapped, g)

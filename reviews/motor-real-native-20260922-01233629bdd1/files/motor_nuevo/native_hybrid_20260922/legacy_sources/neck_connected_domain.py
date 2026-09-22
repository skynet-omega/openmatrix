"""Select known continuation of an existing PN domain, without editing labels."""
import numpy as np
from scipy.ndimage import label,generate_binary_structure


def connected_label_domain(labels, seed_mask):
    """Keep every six-neighbour label component intersecting the prior domain.

    Omitted components are outside this experiment, not certified disconnected
    from the whole neuron: a path may leave the known context and return.
    """
    labels=np.asarray(labels);seed=np.asarray(seed_mask)
    if (labels.dtype!=np.bool_ or labels.ndim!=3 or seed.dtype!=np.bool_
            or seed.shape!=labels.shape or not seed.any() or np.any(seed&~labels)):
        raise ValueError('Original Boolean labels and nonempty supported prior domain required')
    components,n=label(labels,generate_binary_structure(3,1))
    keep=np.unique(components[seed]);assert (keep>0).all()
    selected=np.isin(components,keep);sizes=np.bincount(components.ravel())
    excluded=np.setdiff1d(np.arange(1,n+1),keep)
    assert not np.any(seed&~selected) and not np.any(selected&~labels)
    return selected,dict(known_label_components=int(n),retained_component_ids=keep.tolist(),
        seed_cells=int(seed.sum()),selected_cells=int(selected.sum()),
        omitted_component_sizes=sizes[excluded].tolist(),omitted_cells=int(sizes[excluded].sum()),
        six_neighbours_only=True,labels_edited=False,whole_neuron_disconnection_claimed=False)

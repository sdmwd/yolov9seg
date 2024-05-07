import numpy as np


def group_masks_by_overlap(groupes_de_dommages):
    """
    Regroupe les masques qui ont le même type de dommage et au moins un pixel en commun.

    """
    grouped_results = {}

    for damage_type, mask_tuples in groupes_de_dommages.items():
        merged_masks = []
        remaining_masks = list(mask_tuples)

        while remaining_masks:
            first_idx, first_mask = remaining_masks.pop(0)
            combined_group = [first_mask]
            group_indices = [first_idx]

            def mask_overlaps_group(new_mask, group):
                return any(np.any(np.bitwise_and(new_mask, m)) for m in group)

            new_remaining_masks = []
            for idx, mask in remaining_masks:
                if mask_overlaps_group(mask, combined_group):
                    combined_group.append(mask)
                    group_indices.append(idx)
                else:
                    new_remaining_masks.append((idx, mask))

            remaining_masks = new_remaining_masks
            combined_mask = np.bitwise_or.reduce(combined_group)
            merged_masks.append((group_indices[0], combined_mask))

        grouped_results[damage_type] = merged_masks

    return grouped_results


dummy_masks = {
    0: [
        (1, np.array([[1, 1, 0, 0, 0, 0, 1, 1, 0, 0]], dtype=bool)),
        (2, np.array([[0, 1, 1, 0, 0, 0, 1, 0, 1, 0]], dtype=bool))
    ],
    1: [
        (3, np.array([[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=bool)),
        (4, np.array([[1, 0, 1, 1, 0, 0, 0, 0, 1, 1]], dtype=bool))
    ],
    2: [
        (5, np.array([[0, 0, 1, 1, 0, 1, 1, 0, 0, 1]], dtype=bool)),
        (6, np.array([[1, 1, 0, 0, 0, 1, 0, 1, 0, 1]], dtype=bool)),
        (8, np.array([[1, 1, 0, 0, 0, 1, 0, 1, 0, 1]], dtype=bool)),
        (9, np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=bool)),
    ]
}

grouped_masks = group_masks_by_overlap(dummy_masks)

for k, v in grouped_masks.items():
    print('----------------------------------------')
    print(k)
    print(v)
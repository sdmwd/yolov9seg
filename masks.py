def fus_masks(groupes_de_dommages):
    """
    Fusionne les masques qui ont le même type de dommage et au moins un pixel en commun,
    tout en excluant les masques individuels qui ont été utilisés dans une fusion.
    """
    # Define the priority mapping
    damage_priority = {5: 4, 3: 2, 2: 3, 1: 1}  # Priorité des types de dommage
    masks_fus = np.zeros_like(next(iter(groupes_de_dommages.values()))[0][1], dtype=bool)
    
    # Extract the list of all masks with their damage type and index
    all_masks = [
        (dtype, idx, mask) 
        for dtype, masks in groupes_de_dommages.items() 
        for idx, mask in masks
    ]
    all_masks.sort(key=lambda x: damage_priority.get(x[0], 0), reverse=True)

    # Initialize structures to store merged masks and covered pixels
    merged_masks = {}
    covered_pixels = np.zeros_like(masks_fus, dtype=bool)

    # Process and merge masks based on priority
    for dtype, idx, current_tuple in all_masks:
        _, current_mask = current_tuple  # Extract the mask part from the tuple
        current_mask = current_mask.astype(bool)
        effective_mask = current_mask & ~covered_pixels

        if np.any(effective_mask):
            if idx not in merged_masks:
                merged_masks[idx] = {}

            merged_masks[idx][dtype] = effective_mask.astype(float)
            covered_pixels |= effective_mask

    print("Masques fusionnés:", merged_masks)
    return merged_masks

def fus_masks(groupes_de_dommages):
    """
    Fusionne les masques qui ont le même type de dommage et au moins un pixel en commun,
    tout en excluant les masques individuels qui ont été utilisés dans une fusion.
    """

    # Define the priority mapping and set default mask container
    damage_priority = {5: 4, 3: 2, 2: 3, 1: 1}  # Priorité des types de dommage
    masks_fus = np.zeros_like(next(iter(groupes_de_dommages.values()))[0][1], dtype=bool)
    
    # Create a list of all masks with their damage type and index
    all_masks = [(dtype, idx, mask) for dtype, masks in groupes_de_dommages.items() for idx, mask in enumerate(masks)]
    all_masks.sort(key=lambda x: damage_priority.get(x[0], 0), reverse=True)

    # Initialize structures to store merged masks and covered pixels
    merged_masks = {}
    covered_pixels = np.zeros_like(masks_fus, dtype=bool)

    # Process and merge masks based on priority
    for dtype, idx, current_mask in all_masks:
        current_mask = current_mask.astype(bool)
        effective_mask = current_mask & ~covered_pixels

        if np.any(effective_mask):
            if idx not in merged_masks:
                merged_masks[idx] = {}

            merged_masks[idx][dtype] = effective_mask.astype(float)
            covered_pixels |= effective_mask

    print("Masques fusionnés:", merged_masks)
    return merged_masks

# Function definition for 'calculer_overlap' should be added based on requirements

import numpy as np


def grouper_masques_par_dommage(dm, dg):
    """
    Grouper les masques par leurs types de dommage respectifs, 
    en conservant les indices originaux et en excluant les masques vides.

    """
    groupes_de_dommages = {}
    for index, masque in dm.items():
        if np.any(masque):
            type_de_dommage = dg[index]
            if type_de_dommage in groupes_de_dommages:
                groupes_de_dommages[type_de_dommage].append((index, masque))
            else:
                groupes_de_dommages[type_de_dommage] = [(index, masque)]
    return groupes_de_dommages


def calculer_overlap(masques_fusionnes, mp):
    """
    Calcule l'overlap entre les masques fusionnés (organisés par indice puis par type de dommage)
    et un autre dictionnaire de masques (mp) indexé par indice.
    """
    for idx, types in masques_fusionnes.items():
        for dtype in types.keys():
            fusion_mask = types[dtype].astype(bool)
            mp_mask = mp[idx].astype(bool)
            overlap_mask = fusion_mask & mp_mask 
            overlap_count = np.sum(overlap_mask)
            total_mp_pixels = np.sum(mp_mask)
            overlap_percentage = overlap_count / total_mp_pixels if total_mp_pixels > 0 else 0
            masques_fusionnes[idx][dtype] = (types[dtype], overlap_percentage)
        
    return masques_fusionnes


def fusionner_masques(groupes_de_dommages):
    """
    Fusionne les masques qui ont le même type de dommage et au moins un pixel en commun, 
    tout en excluant les masques individuels qui ont été utilisés dans une fusion.
    """
    damage_priority = {1: 3, 2: 1, 3: 2, 4: 4}  # Priorité des types de dommage
    masques_fusionnes = {}
    covered_pixels = np.zeros_like(next(iter(groupes_de_dommages.values()))[0][1], dtype=bool)
    all_masks = [(dtype, idx, mask) for dtype, masks in groupes_de_dommages.items() for idx, mask in masks]
    all_masks.sort(key=lambda x: damage_priority[x[0]], reverse=True)

    for dtype, idx, current_mask in all_masks:
        current_mask = current_mask.astype(bool)
        effective_mask = current_mask & ~covered_pixels

        if np.any(effective_mask):
            if idx not in masques_fusionnes:
                masques_fusionnes[idx] = {}
            masques_fusionnes[idx][dtype] = effective_mask.astype(float)  
            covered_pixels |= effective_mask

    return masques_fusionnes


array_size = 509600
dm = {
    50: np.array([1, 0, 0, 0]),
    51: np.array([0, 1, 0, 0]),
    52: np.array([1, 0, 1, 0]),
    53: np.array([0, 0, 0, 1]),

}
mp = {
    50: np.array([1, 1, 1, 0]),
    51: np.array([0, 1, 0, 0]),
    52: np.array([1, 0, 1, 0]),
    53: np.array([1, 1, 1, 1]),
}

dg = {50: 1,
      51: 1,
      52: 2,
      53: 3}

masques_groupes = grouper_masques_par_dommage(dm, dg)
print("Résultat du regroupement:", masques_groupes)

masques_fusionnes = fusionner_masques(masques_groupes)
print("Résultat de la fusion:", masques_fusionnes)

masques_fusionnes = calculer_overlap(masques_fusionnes, mp)
print("Overlaps:", masques_fusionnes)


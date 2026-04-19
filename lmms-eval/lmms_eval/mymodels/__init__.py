def get_vision_start_end_indices(input_ids, vision_token_ids):
    sections = []
    in_section = False
    current_token_id = None
    start_idx = -1

    for idx, value in enumerate(input_ids):
        if value in vision_token_ids:
            if not in_section:
                start_idx = idx
                in_section = True
                current_token_id = value
            elif value != current_token_id:
                sections.append((start_idx, idx - 1))
                start_idx = idx
                current_token_id = value
        else:
            if in_section:
                sections.append((start_idx, idx - 1))
                in_section = False
                current_token_id = None

    # Handle case where tensor ends with a continuous section
    if in_section:
        sections.append((start_idx, len(input_ids) - 1))

    return tuple(sections)
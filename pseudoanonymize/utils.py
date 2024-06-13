def flatten_replacement_dict(replacement_dict):
    flat_dict = {}
    for key, value in replacement_dict.items():
        if isinstance(key, tuple):
            for sub_key in key:
                flat_dict[sub_key] = value
        else:
            flat_dict[key] = value
    return flat_dict

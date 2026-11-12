def map_category(class_name: str, config: dict):
    if config.get("status") != "CONFIGURED":
        return None, "UNCONFIGURED"
    category = (config.get("mappings") or {}).get(class_name)
    if not category:
        return None, "UNMAPPED"
    return category, "MAPPED"

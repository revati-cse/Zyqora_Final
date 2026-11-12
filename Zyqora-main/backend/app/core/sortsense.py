DEFAULT_ACTIONS = {"Recyclable": "RECYCLE", "Organic": "COMPOST", "Residual": "RESIDUAL"}

def sort_action(category, actions=None):
    if category is None:
        return "CHECK"
    return (actions or DEFAULT_ACTIONS).get(category, "CHECK")

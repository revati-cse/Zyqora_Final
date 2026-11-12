def summarize(items):
    total = len(items)
    accepted = [x for x in items if x.get("trust") == "ACCEPT"]
    recycle = sum(x.get("action") == "RECYCLE" for x in accepted)
    compost = sum(x.get("action") == "COMPOST" for x in accepted)
    residual = sum(x.get("action") == "RESIDUAL" for x in accepted)
    reviews = sum(x.get("trust") == "REVIEW" for x in items)
    return {
        "total": total, "accepted": len(accepted), "review": reviews,
        "recyclable": recycle, "organic": compost, "residual": residual,
        "diversion_rate": ((recycle + compost) / len(accepted)) if accepted else None,
        "review_rate": (reviews / total) if total else 0.0,
    }

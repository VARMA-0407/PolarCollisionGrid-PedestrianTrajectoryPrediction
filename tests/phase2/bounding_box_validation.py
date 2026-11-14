def validate_bbox_xyxy(bbox):
    """Validate an image-space box ordered as x_min, y_min, x_max, y_max."""
    if len(bbox) != 4:
        raise ValueError("bbox_xyxy must contain exactly four coordinates")
    x_min, y_min, x_max, y_max = bbox
    if x_min > x_max:
        raise ValueError("bbox_xyxy has x_min greater than x_max")
    if y_min > y_max:
        raise ValueError("bbox_xyxy has y_min greater than y_max")
    return True

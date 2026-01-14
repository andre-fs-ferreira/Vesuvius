from monai.transforms.transform import MapTransform, Transform

class GetROIMaskdd(MapTransform):
    """
    Create a ROI mask from the ground truth by setting to 0 the regions with ignore_mask_value.
    The ROI mask will have 1 where the ground truth is not equal to ignore_mask_value, and 0 elsewhere.
    """

    def __init__(self, keys, ignore_mask_value=2, new_key_names=None):
        self.keys = keys
        self.ignore_mask_value = ignore_mask_value
        self.new_key_names = new_key_names

    def __call__(self, data):
        for key, new_key in zip(self.keys, self.new_key_names):
            gt = data[key]
            roi_mask = (gt != self.ignore_mask_value).float()
            data[new_key] = roi_mask
        return data

class GetBinaryLabeld(MapTransform):
    def __init__(self, keys, ignore_mask_value=2):
        super().__init__(keys)
        self.ignore_mask_value = ignore_mask_value

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            val = d[key]
            # 1. Use a small epsilon for the ignore value check (safety)
            # This handles values like 1.999 or 2.0000153
            mask = (val > (self.ignore_mask_value - 0.1)) & (val < (self.ignore_mask_value + 0.1))
            val[mask] = 0
            
            # 2. FORCE the remaining values into strictly 0 or 1
            # Anything that isn't background (0) should be 1
            # This fixes the 1.0000153 issue
            val[val > 0.5] = 1.0
            val[val <= 0.5] = 0.0
            
            d[key] = val
        return d

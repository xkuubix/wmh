def test_non_zero_img_unique(bags):
    for bag in bags:
        non_zero_imgs = [img for img, label in zip(bag["image"], bag["label"]) if label == 1]
        assert len(set(non_zero_imgs)) == len(non_zero_imgs), "Non-zero image paths are not unique within a bag"
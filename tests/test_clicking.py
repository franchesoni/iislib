def encode__disk_mask_from_coords():
    import matplotlib.pyplot as plt
    from clicking.encode import disk_mask_from_coords

    points = [(130, 120), (200, 20)]
    out_shape = (256, 256)
    mask = disk_mask_from_coords(points, out_shape)
    plt.imshow(mask)
    plt.savefig("temp.png")

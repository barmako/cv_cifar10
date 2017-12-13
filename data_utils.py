def flat_image(img):
    flat_img = []
    for row in img:
        for pixel in row:
            for channel in pixel:
                flat_img.append(channel)
    return flat_img


def flat_raw(data):
    flatted_data = []
    for img in data:
        flatted_data.append(flat_image(img))
    return flatted_data
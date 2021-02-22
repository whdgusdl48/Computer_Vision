import imutils

# 이미지 슬라이딩
# sliding_window => 매개변수 이미지, 움직일 스텝 수, 박스 크기

def sliding_window(image, step, ws):
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])

def image_pyramid(image,scale=1.5, minSize=(224,224)):

    yield image
    # 이미지 피라미드 형식
    while True:
        w = int(image.shape[1] / scale)
        # 바운딩 박스를 추출할 이미지 재정의
        image = imutils.resize(image,width=w)

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image


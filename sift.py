import cv2


def sift_alignment(image_1: str, image_2: str):

    im1 = cv2.imread(image_1,)
    im2 = cv2.imread(image_2,)

    sift = cv2.xfeatures2d.SIFT_create()
    key_points_1, descriptors_1 = sift.detectAndCompute(im1, None)
    key_points_2, descriptors_2 = sift.detectAndCompute(im2, None)

    bf_matcher = cv2.BFMatcher()  #暴力匹配
    matches = bf_matcher.knnMatch(descriptors_1, descriptors_2, k=2)

    # 应用比率匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # this parameter affects the result filtering
            good_matches.append([m])

    match_img = cv2.drawMatchesKnn(im1, key_points_1, im2, key_points_2,
                                   good_matches, None, flags=2)
    return  match_img


match_img = sift_alignment('01.jpg', '02.jpg')
cv2.imshow('match', match_img)
cv2.waitKey(0)
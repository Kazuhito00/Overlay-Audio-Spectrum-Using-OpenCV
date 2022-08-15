#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import cv2
import numpy as np


def cv_overlay_inset_image(
    background_image,
    overlay_image,
    points,
    mask_offet=2,
):
    assert len(points) == 4, 'Coordinates point must be 4 points'

    temp_background_image = copy.deepcopy(background_image)

    overlay_image_w = overlay_image.shape[1]
    overlay_image_h = overlay_image.shape[0]
    background_image_w = background_image.shape[1]
    background_image_h = background_image.shape[0]

    # 射影変換
    pts1 = np.float32([
        [0, 0],
        [overlay_image_w, 0],
        [overlay_image_w, overlay_image_h],
        [0, overlay_image_h],
    ])
    pts2 = np.float32([
        points[0],
        points[1],
        points[2],
        points[3],
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warp_overlay_image = cv2.warpPerspective(
        overlay_image,
        M,
        (background_image_w, background_image_h),
    )

    # マスク画像生成
    mask_image = np.zeros_like(warp_overlay_image).astype(np.uint8)
    mask_pts = np.float32([
        (points[0][0] + mask_offet, points[0][1] + mask_offet),
        (points[1][0] - mask_offet, points[1][1] + mask_offet),
        (points[2][0] - mask_offet, points[2][1] - mask_offet),
        (points[3][0] + mask_offet, points[3][1] - mask_offet),
    ])
    cv2.drawContours(
        mask_image,
        [np.array(mask_pts).astype(np.int32)],
        -1,
        color=(255, 255, 255),
        thickness=-1,
    )

    # 重畳描画
    result_image = np.where(
        mask_image == 0,
        temp_background_image,
        warp_overlay_image,
    ).astype(np.uint8)

    return result_image

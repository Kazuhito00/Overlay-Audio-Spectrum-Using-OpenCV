#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
from collections import deque

import cv2
import wave
import pyaudio
import numpy as np

from cv_overlay_inset_image import cv_overlay_inset_image

from draw_audio_function import (
    draw_audio_spectrum01,
    draw_audio_spectrum02,
    draw_audio_spectrum03,
    draw_audio_waveform01,
    draw_audio_waveform02,
)

draw_function_list = [
    draw_audio_spectrum01,
    draw_audio_spectrum02,
    draw_audio_spectrum03,
    draw_audio_waveform01,
    draw_audio_waveform02,
]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bg_device", type=int, default=None)
    parser.add_argument("--bg_movie", type=str, default=None)
    parser.add_argument("--bg_image", type=str, default='sample.jpg')

    parser.add_argument("--wave", type=str, default=None)
    parser.add_argument("--audio_device", type=int, default=1)

    parser.add_argument("--frames", type=int, default=2048)
    parser.add_argument("--fft_n", type=int, default=1024)

    parser.add_argument("--draw_type", type=int, default=0)
    parser.add_argument("--color", type=str, default=None)
    parser.add_argument("--border_color", type=str, default='0,0,0')

    args = parser.parse_args()

    return args


class ImageCapture(object):
    _type = None
    TYPE_WEBCAM = 0
    TYPE_MOVIE = 1
    TYPE_IMAGE = 2

    _video_capture = None
    _image = None

    def __init__(self, device_no, movie_path, image_path):
        if movie_path is not None:
            self._type = self.TYPE_MOVIE
            self._video_capture = cv2.VideoCapture(movie_path)
        elif device_no is not None:
            self._type = self.TYPE_WEBCAM
            self._video_capture = cv2.VideoCapture(device_no)
        elif image_path is not None:
            self._type = self.TYPE_IMAGE
            self._video_capture = None
            self._image = cv2.imread(image_path)
        else:
            assert False, 'Be sure to specify device_no or movie_path or image_path.'

    def read(self):
        ret = False
        image = None
        if self._type == self.TYPE_WEBCAM:
            ret, self._image = self._video_capture.read()
            image = copy.deepcopy(self._image)
        elif self._type == self.TYPE_MOVIE:
            ret, self._image = self._video_capture.read()
            image = copy.deepcopy(self._image)
        elif self._type == self.TYPE_IMAGE:
            ret = True
            image = copy.deepcopy(self._image)

        return ret, image

    def release(self):
        if self._type == self.TYPE_WEBCAM:
            self._video_capture.release()
        elif self._type == self.TYPE_MOVIE:
            self._video_capture.release()
        elif self._type == self.TYPE_IMAGE:
            pass

    def get_video_capture_instance(self):
        return self._video_capture


class CvWindow(object):
    _window_name = ''
    _frame = None

    _click_point = None
    _click_point_queue = None

    def __init__(self, window_name='DEBUG', point_history_maxlen=4):
        self._window_name = window_name

        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

        self._click_point_queue = deque(maxlen=point_history_maxlen)

    def imshow(self, image):
        cv2.imshow(self._window_name, image)

    def get_click_point_history(self):
        return self._click_point_queue

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_point = [x, y]
            self._click_point_queue.append(self._click_point)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._click_point = None
            self._click_point_queue.clear()


def extract_warp_erspective_image(
    image,
    points,
    width=640,
    height=360,
):
    assert len(points) == 4, 'Coordinates point must be 4 points'

    # 射影変換
    pts1 = np.float32([
        points[0],
        points[1],
        points[2],
        points[3],
    ])
    pts2 = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result_image = cv2.warpPerspective(
        image,
        M,
        (width, height),
    )

    return result_image


def main():
    # コマンドライン引数
    args = get_args()

    bg_device = args.bg_device
    bg_movie = args.bg_movie
    bg_image = args.bg_image

    filename = args.wave
    audio_device_index = args.audio_device
    frame_n = args.frames
    fft_sample_size = args.fft_n

    draw_type = args.draw_type
    color = args.color
    if color is not None:
        color = tuple(map(int, color.split(',')))
        assert len(
            color) == 3, 'Specify the color in BGR format Example: 255,255,255'
    border_color = args.border_color
    if border_color is not None:
        border_color = tuple(map(int, border_color.split(',')))
        assert len(
            border_color
        ) == 3, 'Specify the border_color in BGR format Example: 255,255,255'

    # 画像準備
    bg_image_capture = ImageCapture(bg_device, bg_movie, bg_image)

    # ウィンドウ準備
    cv_window = CvWindow(window_name='Demo', point_history_maxlen=4)

    # オーディオストリームを開く
    audio = pyaudio.PyAudio()
    if filename is not None:
        # Waveファイル
        wave_file = wave.open(filename, "r")

        format = audio.get_format_from_width(wave_file.getsampwidth())
        nchannels = wave_file.getnchannels()
        framerate = wave_file.getframerate()
        input_mode = False
        input_device_index = audio_device_index
        frames_per_buffer = frame_n
    else:
        # マイクなどのデバイス
        format = pyaudio.paInt16
        nchannels = 1
        framerate = 44100
        input_mode = True
        input_device_index = audio_device_index
        frames_per_buffer = frame_n

    audio_stream = audio.open(
        format=format,
        channels=nchannels,
        rate=framerate,
        input=input_mode,
        input_device_index=input_device_index,
        output=True,
        frames_per_buffer=frames_per_buffer,
    )

    # ハミング窓生成
    hamming_window = np.hamming(fft_sample_size)

    # オーディオ開始フラグ
    audio_started = False
    audio_frame = None

    while True:
        # クリック位置取得
        click_point_history = cv_window.get_click_point_history()
        
        # 4隅が選択されたらオーディオ開始
        if len(click_point_history) == 4 and not audio_started:
            audio_started = True
        
        # オーディオ処理（4隅選択後のみ）
        if audio_started:
            # 音声ファイル読み込み
            if filename is not None:
                # Waveファイル フレーム読み込み
                audio_frame = wave_file.readframes(frame_n)
                if audio_frame == b'':
                    # break
                    wave_file.rewind()
            else:
                # デバイスからの読み込み
                audio_frame = audio_stream.read(frame_n)

            # フレーム再生
            audio_stream.write(audio_frame)

        # 画像読み込み
        ret, bg_image = bg_image_capture.read()
        if not ret:
            break
        debug_image = copy.deepcopy(bg_image)

        # オーディオスペクトラム用データを生成（オーディオ開始後のみ）
        if audio_started and audio_frame is not None:
            # 正規化バッファ取得
            audio_buffer = np.frombuffer(audio_frame, dtype="int16") / 32767
            # 一部分のみ切り出し
            # （正確なスペクトログラムが欲しいわけではないので処理時間短縮のためにシフト省略）
            if audio_buffer.shape[0] > fft_sample_size:
                sampling_data = audio_buffer[audio_buffer.shape[0] -
                                             fft_sample_size:]
            # 窓適応
            sampling_data = hamming_window * sampling_data
            # 周波数解析
            frequency = np.fft.fft(sampling_data)
            amplitude = np.abs(frequency)
            amplitude_spectrum = 20 * np.log(amplitude)
        else:
            # オーディオ未開始時はダミーデータ
            sampling_data = np.zeros(fft_sample_size)
            amplitude_spectrum = np.zeros(fft_sample_size // 2 + 1)

        # デバッグ描画
        if len(click_point_history) < 4:
            for click_point in click_point_history:
                cv2.circle(
                    debug_image,
                    (click_point[0], click_point[1]),
                    4,
                    (255, 255, 255),
                    -1,
                )
                cv2.circle(
                    debug_image,
                    (click_point[0], click_point[1]),
                    2,
                    (0, 0, 0),
                    -1,
                )
        # はめ込み画像生成
        elif len(click_point_history) == 4:
            # クリック位置の画像を抽出
            extract_image = extract_warp_erspective_image(
                bg_image,
                click_point_history,
            )

            # オーディオスペクトラム描画
            if color is not None:
                audio_spectrum_image = draw_function_list[draw_type](
                    amplitude_spectrum,
                    sampling_data,
                    plot_color=color,
                    bg_image=extract_image,
                )
            else:
                audio_spectrum_image = draw_function_list[draw_type](
                    amplitude_spectrum,
                    sampling_data,
                    bg_image=extract_image,
                )

            # はめ込み画像合成
            debug_image = cv_overlay_inset_image(
                bg_image,
                audio_spectrum_image,
                click_point_history,
                border_color=border_color,
            )

        # 画面反映
        cv_window.imshow(debug_image)

        # キー処理(ESC：終了)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    bg_image_capture.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------
if __name__ == "__main__":
    main()

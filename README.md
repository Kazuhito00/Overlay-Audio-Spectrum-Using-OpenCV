# Overlay-Audio-Spectrum-Using-OpenCV
オーディオスペクトラムや波形をOpenCVで、はめ込み描画するサンプルです。<br>
[Draw-Audio-Spectrum-Using-OpenCV](https://github.com/Kazuhito00/Draw-Audio-Spectrum-Using-OpenCV)、[cv_overlay_inset_image](https://github.com/Kazuhito00/cv_overlay_inset_image)を用いています。

https://user-images.githubusercontent.com/37477845/184840565-16602504-0fac-436b-a156-495bc59b7c9a.mp4

# Requirement
```
opencv-python 4.5.5.62 or later
PyAudio 0.2.11         or later
```

# Contents
<table>
    <tr>
        <td width="50">
            No.0
        </td>
        <td width="640">
            <img src="https://user-images.githubusercontent.com/37477845/184486210-a7b4f36e-ebc6-4a3b-99e4-2a94ecc9bb8d.gif" loading="lazy" width="620px">
        </td>
    </tr>
    <tr>
        <td width="50">
            No.1
        </td>
        <td width="640">
            <img src="https://user-images.githubusercontent.com/37477845/184486230-f57a14bd-4616-4c84-93b2-51b66b5d4030.gif" loading="lazy" width="620px">
        </td>
    </tr>
    <tr>
        <td width="50">
            No.2
        </td>
        <td width="640">
            <img src="https://user-images.githubusercontent.com/37477845/184486234-50d0caad-6deb-4c81-9871-f0d3235a95b3.gif" loading="lazy" width="240px">
        </td>
    </tr>
    <tr>
        <td width="50">
            No.3
        </td>
        <td width="640">
            <img src="https://user-images.githubusercontent.com/37477845/184486246-6225aab4-71a6-4d16-9ffe-36f8950cf340.gif" loading="lazy" width="620px">
        </td>
    </tr>
    <tr>
        <td width="50">
            No.4
        </td>
        <td width="640">
            <img src="https://user-images.githubusercontent.com/37477845/184486280-d920223c-678b-4581-ae4e-a47200e46e4e.gif" loading="lazy" width="620px">
        </td>
    </tr>
</table>

# Usage
デモの実行方法は以下です。
```bash
python demo.py --wave=[Your wav file]
```
* --bg_image<br>
背景画像のパス指定<br>
デフォルト：sample.jpg
* --bg_movie<br>
背景動画のパス指定 ※指定時はbg_imageより優先<br>
デフォルト：指定なし
* --bg_device<br>
背景用Webカメラ画像のデバイス指定 ※指定時はbg_imageより優先<br>
デフォルト：指定なし
* --wave<br>
Waveファイルの指定 ※指定時はマイクデバイスより優先<br>
デフォルト：指定なし
* --frames<br>
フレーム読み出し数<br>
デフォルト：2048
* --fft_n<br>
FFTポイント数<br>
デフォルト：1024
* --draw_type<br>
Contentsの描画種別<br>
デフォルト：0
* --color<br>
重畳描画するオーディオスペクトラムや波形の色指定<br>
デフォルト：指定なし ※指定時はBGRで255,255,255のように指定
* --border_color<br>
warpPerspectiveの境界の色<br>
デフォルト：0,0,0

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Overlay-Audio-Spectrum-Using-OpenCV is under [Apache-2.0 license](LICENSE).<br><br>

また、サンプル画像は[フリー素材ぱくたそ](https://www.pakutaso.com)様の<br>
「[ずっと使われていない天井吊りのテレビの写真素材](https://www.pakutaso.com/20180344079post-15604.html)」を利用しています。

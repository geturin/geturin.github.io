# デプス画像から三次元点群を構築の高速化(Python,numpy) <br>


## この記事について
---
初めまして、KEROと申します。<BR>
ロボット工学や画像工学などこと研究してます。<BR>
外人ですか、もっと日本語で技術のことを話しできるため、日本語で技術blogを書き始めた<br>

<br>
今日はPyhonでデプス画像から三次元点群を構築のことついて話します。<br>
<br>

画像は一枚だけで場合なら、コード例が見つけやすい<BR>
例えば[この記事](https://qiita.com/iwatake2222/items/bbd4ccdacfcf8f846a0a)<br>
もしくは[open3d中の関数](http://www.open3d.org/docs/0.7.0/python_api/open3d.geometry.create_point_cloud_from_depth_image.html)<br>

以前Pythonとnumpyでこんな計算行って、点群構築のコード書きましたが、計算めちゃくちゃ遅い。<BR>
自分の目的はロボットのナビゲーションので、連続の画像をリアルタイムで処理なければならない<br>

本来は遅くの理由としては、計算量が多いもしくはPython自体は遅いと思うから、C++で書き直し予定ですが。<br>
<br>

その前計算アルゴリズムを見直しため、もう一回Pythonとnumpyで書き直しました<br>
結果としては処理速度がお幅上げて、Pyhonでもリアルタイム処理できました

テストとしては **960,720** サイズのデプス画像を三次元点群を構築**1000**回ループして、処理時間は**17**秒<br>
```
[Running] python -u "g:\test\test.py"

[Done] exited with code=0 in 17.162 seconds exited with code=0 in 17.162 seconds
```

## コード
---
まずコードを上げて、その下で説明補足します
```
class depth_to_pcd(object):

    def __init__(self, resize_scale):
        self.resize_scale = resize_scale
        self.resize_camera = np.array((910.072 / self.resize_scale, 0, 485.523 / self.resize_scale, 0,
                                       0, 914.094 / self.resize_scale, 336.718 / self.resize_scale, 0,
                                       0, 0, 1, 0), dtype="float").reshape(3, 4)

        self.pixel = np.array([0,0,1]).reshape(3,-1)
        self.resize_camera = np.matrix(self.resize_camera).I
        
        self.pcd_list = np.zeros((720//self.resize_scale,960//self.resize_scale,3))
        self.get_depth_vector()
        return

    def get_depth_vector(self):
        fake_depth =np.zeros((720//self.resize_scale,960//self.resize_scale))
        it = np.nditer(fake_depth, flags=['multi_index'])
        with it:
            while not it.finished:
                self.pixel[0] = it.multi_index[1]
                self.pixel[1] = it.multi_index[0]
                point = np.dot(self.resize_camera, self.pixel)          
                self.pcd_list[it.multi_index] = point[0:3].T[0]
                it.iternext()

    def get_pcd(self, depth):
        self.depth = cv2.resize(depth, (int(960 / self.resize_scale), int(720 / self.resize_scale)),interpolation=cv2.INTER_NEAREST)
        vector_array = self.pcd_list.copy()
        vector_array *= self.depth.reshape((int(720 / self.resize_scale), int(960 / self.resize_scale),1))
        
            
        pointcloud = vector_array.reshape(-1,1,3)
        return pointcloud
```
<br>
テスト用例

```
depth = np.load("ai_depth.npy")
depth2pcd = depth_to_pcd(1)

for i in range(1000):
    pointcloud = depth2pcd.get_pcd(depth)
```
## 説明
---
Pythonで高速計算したいなら<br>
できる限りPythonで計算、ループしなく、numpyで完成して<BR>

そのため、計算手順はできる限り行列式計算でいく<BR>
<BR>
### 初期化
---
<BR>

まず[カメラ座標系](https://mem-archive.com/2018/10/13/post-682/)から画像座標系で変換方程式は

$$
Z_{c}  \begin{bmatrix}u
   \\v\\1    \end{bmatrix}    =\begin{bmatrix}
  f_{x}& 0 & u_{0}&0 \\
  0& f_{y} &v_{0}  & 0\\
  0&  0&  1&0
\end{bmatrix}
\begin{bmatrix}
X_{c} \\Y_{c}\\Z_{c}
\end{bmatrix}
$$

説明のため、以下の形で略書きします
$$
Z\begin{bmatrix}P\end{bmatrix}=\begin{bmatrix}C\end{bmatrix}\begin{bmatrix}D\end{bmatrix}
$$

中では[P]は画像座標系の座標 すなわちpixelの位置　u,v<br>
[C]は[カメラ内部パラメータ](https://mem-archive.com/2018/02/21/post-157/)<br>
[D]がカメラ座標系の座標、すなわち点群座標、今回計算の目的<br>

だから計算のため、先ずは[C] カメラパラメータ行列式を導入
```
 def __init__(self, resize_scale):
        self.resize_scale = resize_scale
        self.resize_camera = np.array((910.072 / self.resize_scale, 0, 485.523 / self.resize_scale, 0,
                                       0, 914.094 / self.resize_scale, 336.718 / self.resize_scale, 0,
                                       0, 0, 1, 0), dtype="float").reshape(3, 4)
```

こちらはのパラメータ行列式は自分ロボット用いたカメラをcalibrationした結果<br>
さらに、処理画像のサイズを調整するため、、resize_scaleパラメータを用意しました<br>
画像サイズが調整する場合、カメラパラメータも合わせて調整が必要

<BR>
[P]　pixel座標行列式を初期化

```
self.pixel = np.array([0,0,1]).reshape(3,-1)
```
<BR>
今回は逆変換、つまり上の方程式では　[D]　は求めている答え

$$
\begin{bmatrix}D\end{bmatrix}=\begin{bmatrix}P\end{bmatrix}\begin{bmatrix}C\end{bmatrix}^{-1} Z
$$

[C]の逆行列を計算

```
self.resize_camera = np.matrix(self.resize_camera).I
```
<BR>

もう一回この方程式を見直しすると
$$
\begin{bmatrix}D\end{bmatrix}=\begin{bmatrix}P\end{bmatrix}\begin{bmatrix}C\end{bmatrix}^{-1} Z
$$
同じカメラの同じサイズの画像が連続入力する場合[P]と[C]は変わらない、Zだけが変化します<br>
すなわち

$$
\begin{bmatrix}D\end{bmatrix}_{t1}=\begin{bmatrix}P\end{bmatrix}\begin{bmatrix}C\end{bmatrix}^{-1} Z_{t1}\\
\begin{bmatrix}D\end{bmatrix}_{t2}=\begin{bmatrix}P\end{bmatrix}\begin{bmatrix}C\end{bmatrix}^{-1} Z_{t2}\\
\vdots \\
\begin{bmatrix}D\end{bmatrix}_{ti}=\begin{bmatrix}P\end{bmatrix}\begin{bmatrix}C\end{bmatrix}^{-1} Z_{ti}\\
\bigtriangledown \\
\begin{bmatrix}P\end{bmatrix}\begin{bmatrix}C\end{bmatrix}^{-1}=\begin{bmatrix}K\end{bmatrix}\\
\begin{bmatrix}D\end{bmatrix}_{ti}= \begin{bmatrix}K\end{bmatrix}Z_{ti}\\
$$

<br>
ここで一気に画像すべてのpixelの[K]を計算します<br>
結果保存するため、先ずはデプス画像と同じサイズ、ひとつpixel中では　1ｘ3大きさがあるの行列を作り<br>

```
self.pcd_list = np.zeros((720//self.resize_scale,960//self.resize_scale,3))
```
[np.nditer](https://snowtree-injune.com/2020/06/29/nditer-z009/) 用いてトラバーサル、画像すべてのpixelの[K]を計算して、結果をpcd_list中で対応した位置に保存します

```
    def get_depth_vector(self):
        fake_depth =np.zeros((720//self.resize_scale,960//self.resize_scale))
        it = np.nditer(fake_depth, flags=['multi_index'])
        with it:
            #loop
            while not it.finished:
                #今の座標対して[P] pixel行列更新
                self.pixel[0] = it.multi_index[1]
                self.pixel[1] = it.multi_index[0]
                #[K]の計算
                point = np.dot(self.resize_camera, self.pixel)
                #結果を保存          
                self.pcd_list[it.multi_index] = point[0:3].T[0]
                it.iternext()
```

これでは初期化の準備が完了<BR>
<BR>

### 変換計算
---
先色々準備した結果、本番の変換計算はとても簡単

```
    def get_pcd(self, depth):
        self.depth = cv2.resize(depth, (int(960 / self.resize_scale), int(720 / self.resize_scale)),interpolation=cv2.INTER_NEAREST)
        vector_array = self.pcd_list.copy()
        vector_array *= self.depth.reshape((int(720 / self.resize_scale), int(960 / self.resize_scale),1))
        
            
        pointcloud = vector_array.reshape(-1,1,3)
        return pointcloud
```

先ずはdepth画像を導入、サイズ調整
```
    def get_pcd(self, depth):
        self.depth = cv2.resize(depth, (int(960 / self.resize_scale), int(720 / self.resize_scale)),interpolation=cv2.INTER_NEAREST)
```

初期化時計算の[K]行列を取る、元行列の数値を変換させないため、copy()が必要

```
vector_array = self.pcd_list.copy()
```

numpy.arrayとnumpy.arrayの　*　掛けでは、**対応位置のデータを掛け**<br>
つまり

```
vector_array *= self.depth.reshape((int(720 / self.resize_scale), int(960 / self.resize_scale),1))
```

ではこんな形で計算する

$$
\begin{bmatrix}
  [K]_{0}&\cdots \\
  \vdots &\vdots     \\
  \cdots&\cdots
\end{bmatrix}*\begin{bmatrix}
  Z_{0}&\cdots \\
  \vdots &\vdots     \\
  \cdots&\cdots
\end{bmatrix}=\begin{bmatrix}
  [K]_{0}Z_{0}&\cdots \\
  \vdots &\vdots     \\
  \cdots&\cdots
\end{bmatrix}=\begin{bmatrix}
  [D]_{0}&\cdots \\
  \vdots &\vdots     \\
  \cdots&\cdots
\end{bmatrix}
$$

そうすると一気にデプスから点群の変化を完成します<br>
<br>
残るのは形を調整して、通用の形になるだけ
```
pointcloud = vector_array.reshape(-1,1,3)
```

## 最後
---
そうのように、計算やループはできるだけnumpyを任せすれば、連続計算の場合計算速度はめちゃくちゃ速くなる<br>
<br>
今一番遅くなる部分は初期化時[K]を計算する時<br>
np.nditer 用いてトラバーサルするちょっと速くなったけど、本質としてはまだPythonでループしているから<br>
<br>
テスト用例1000回の処理で17秒の時間では、約7秒は初期化、ほんっとにデプス画像を処理する時間は十秒だけ<br>
<br>
一応現状してもPythonでロボットためリアルタイム処理が出来ました、でも初期化の部分がもっといい方法があるかもしれない

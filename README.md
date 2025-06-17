# mediapipe_demo
Mediapipeを触ってみたコード

## mediapipe_demoについて

◯実行の仕方（共通）

ターミナルで”source venv/bin/activate”を実行し仮想環境に入る

“python ファイル名”で.pyファイルを実行


◯各ファイルの説明

・Collect_Data.py
グー、チョキ、パーの骨格データを集めてcsvに保存する
Enterで収集開始、qで収集終了。Enterはターミナルに合わせて、qはOpenCVウィンドウに合わせて押す。

・PreprocessData_BuildModel.py
Collect_Data.pyで集めたデータの前処理を行い、それをつかってモデルを作る。

・Discrimination.py
リアルタイム判別機。実行するとOpenCVウィンドウが出てくるので、じゃんけんの手を見せるとそれが何の手か判別してくれる。

・JankenGame.py
カウントダウンの後に人が出した手を認識し、AIとじゃんけんできるゲーム。
AIの手の選択の仕方はランダムなので、連続プレイの場合はプレイヤーの傾向を学習させて次の手に反映させるとかでも良いかも。

・JankenGame_JP.py
説明を日本語にしようとして失敗したやつ。fftというフォントのファイルを渡したいけど日本語対応のやつがないやんってなった。非対応だと豆腐□になる。ちなみにJankenGame.pyで日本語使うと?になる。

参考サイト
https://qiita.com/HayatoF/items/1a660ff7008b47b0a20a


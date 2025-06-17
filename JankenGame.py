import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import random
import time
import os # 画像読み込みのパス調整のため追加

# Mediapipe Handsのセットアップ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# モデルの読み込み
# modelフォルダ内のjanken_model.h5を読み込むようにパスを調整
model_path = os.path.join('model', 'janken_model.h5')
try:
    model = load_model(model_path)
    print(f"モデル {model_path} が正常にロードされました。")
except Exception as e:
    print(f"モデルのロード中にエラーが発生しました: {e}")
    print("model/janken_model.h5 が存在し、PreprocessData_BuildModel.py で作成されていることを確認してください。")
    exit()

# ラベルの定義 (PreprocessData_BuildModel.pyのLabelEncoderの順序と一致させる)
# PreprocessData_BuildModel.py の to_categorical(y_encoded, num_classes=3) の結果から推測すると
# おそらく 'paper':0, 'rock':1, 'scissors':2 の順でエンコードされている可能性が高いです。
# 実際にPreprocessData_BuildModel.pyで fit_transform した際の順序を確認してください。
# ここでは例として以下のように定義します。
# 実際の順序が異なる場合は、この配列の順序を変更してください。
LABELS = ["paper", "rock", "scissors"] # 0:paper, 1:rock, 2:scissors と仮定

# 手のランドマークを抽出する関数
def extract_landmark_data(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

# 手の形を判別する関数
def predict_hand_shape(landmarks):
    if len(landmarks) != 63: # 21点 * 3座標 = 63
        return "無効な手" # 検出点が不足している場合など
    
    # ランドマークデータを正しい形に変換
    landmarks = np.array(landmarks).reshape(1, -1)
    
    # モデルで予測
    prediction = model.predict(landmarks, verbose=0) # verbose=0 で予測時のログ出力を抑制
    predicted_class_index = np.argmax(prediction)

    # ラベルを返す
    if 0 <= predicted_class_index < len(LABELS):
        return LABELS[predicted_class_index]
    else:
        return "不明" # 予測されたインデックスが範囲外の場合

# 勝敗判定関数
def judge_janken(player_hand, ai_hand):
    if player_hand == ai_hand:
        return "Draw!"
    elif (player_hand == 'rock' and ai_hand == 'scissors') or \
         (player_hand == 'scissors' and ai_hand == 'paper') or \
         (player_hand == 'paper' and ai_hand == 'rock'):
        return "You Win!"
    else:
        return "AI Wins!"

# カメラのセットアップ
cap = cv2.VideoCapture(0)

# ゲームの状態変数
game_state = "waiting_for_start" # "waiting_for_start", "countdown", "predict_player", "show_result"
player_predicted_hand = ""
ai_choice = ""
game_result = ""
countdown_start_time = 0
countdown_duration = 3 # カウントダウンの秒数

# AIの手の画像を読み込む
ai_hand_images = {}
image_names = {"rock": "rock.png", "scissors": "scissors.png", "paper": "paper.png"}
for hand_key, file_name in image_names.items():
    img_path = os.path.join('images', file_name)
    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # アルファチャンネルも読み込む
        if img is not None:
            ai_hand_images[hand_key] = img
            print(f"画像 {img_path} をロードしました。")
        else:
            print(f"Warning: 画像 {img_path} のロードに失敗しました。ファイルが存在するか確認してください。")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

# 画像をオーバーレイする関数 (アルファチャンネル対応)
def overlay_image_alpha(background, overlay, x, y, scale=1.0):
    if overlay is None or background is None:
        return background

    h_bg, w_bg, _ = background.shape

    # スケーリング
    scaled_w = int(overlay.shape[1] * scale)
    scaled_h = int(overlay.shape[0] * scale)
    if scaled_w <= 0 or scaled_h <= 0:
        return background
    overlay_resized = cv2.resize(overlay, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    h_ov_res, w_ov_res = overlay_resized.shape[0:2]

    # 貼り付け位置を計算 (はみ出さないように調整)
    y1, y2 = max(0, y), min(h_bg, y + h_ov_res)
    x1, x2 = max(0, x), min(w_bg, x + w_ov_res)

    # 貼り付け領域のオーバーレイ画像のサイズ
    overlay_crop_h = y2 - y1
    overlay_crop_w = x2 - x1

    if overlay_crop_h <=0 or overlay_crop_w <=0:
        return background # 貼り付け領域が0以下なら何もしない

    # オーバーレイ画像を背景のROIにコピー
    if overlay_resized.shape[2] == 4: # RGBAの場合
        alpha_s = overlay_resized[0:overlay_crop_h, 0:overlay_crop_w, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            background[y1:y2, x1:x2, c] = (alpha_s * overlay_resized[0:overlay_crop_h, 0:overlay_crop_w, c] +
                                          alpha_l * background[y1:y2, x1:x2, c])
    else: # RGBの場合 (アルファチャンネルがない場合)
        background[y1:y2, x1:x2] = overlay_resized[0:overlay_crop_h, 0:overlay_crop_w]
    return background


# ゲームループ
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("カメラフレームの取得に失敗しました。")
        continue

    frame = cv2.flip(frame, 1) # 水平反転

    # 画像をRGBに変換してMediaPipeで処理
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # BGRに戻す

    # テキスト表示用の設定
    text_color = (0, 255, 0) # 緑
    font_scale = 1
    thickness = 2
    h, w, _ = frame.shape # フレームの高さと幅

    # ランドマーク描画とプレイヤーの手の予測
    player_current_hand = "手を検出中..."
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks_data_for_predict = extract_landmark_data(hand_landmarks)
            
            if game_state == "predict_player":
                player_current_hand = predict_hand_shape(landmarks_data_for_predict)
                # 予測された有効な手があれば、その手をプレイヤーの最終的な手とする
                if player_current_hand in LABELS:
                    player_predicted_hand = player_current_hand
                else:
                    player_predicted_hand = "検出エラー" # 有効な手が検出されなかった場合

    # --- GUIの表示とゲームロジック ---
    if game_state == "waiting_for_start":
        cv2.putText(frame, "Janken Game", (w // 2 - 150, h // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(frame, "Press 's' to Start", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        cv2.putText(frame, "Press 'q' to Quit", (w // 2 - 100, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    elif game_state == "countdown":
        elapsed_time = time.time() - countdown_start_time
        remaining_time = max(0, countdown_duration - int(elapsed_time))
        
        if remaining_time > 0:
            cv2.putText(frame, f"Janken... {remaining_time}", (w // 2 - 180, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "Pon!", (w // 2 - 80, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            # カウントダウン終了後、AIの手を決定し、プレイヤーの予測フェーズへ移行
            ai_choice = random.choice(LABELS) # AIの手を決定
            game_state = "predict_player"
            countdown_start_time = time.time() # 予測時間の計測開始

    elif game_state == "predict_player":
        cv2.putText(frame, "Show your hand!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        cv2.putText(frame, f"Current Prediction: {player_current_hand}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        
        # プレイヤーが手を見せる時間を設定
        # 例: カウントダウン終了後2秒間予測を続ける
        if time.time() - countdown_start_time > 2.0:
            if player_predicted_hand in LABELS: # 有効な手が予測できていれば
                game_result = judge_janken(player_predicted_hand, ai_choice)
            else:
                game_result = "No valid hand detected."
                player_predicted_hand = "N/A" # 結果画面で表示するための設定
            game_state = "show_result"
        
    elif game_state == "show_result":
        cv2.putText(frame, f"You: {player_predicted_hand}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        cv2.putText(frame, f"AI: {ai_choice}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        cv2.putText(frame, game_result, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, font_scale + 0.5, (255, 255, 0), thickness + 1)
        cv2.putText(frame, "'r' for Replay, 'q' to Quit", (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # AIの手の画像をオーバーレイ表示
        if ai_choice in ai_hand_images:
            overlay_img = ai_hand_images[ai_choice]
            # 画像の表示位置 (右上に表示)
            overlay_x = w - int(overlay_img.shape[1] * 0.7) - 20 # 適宜調整
            overlay_y = 20 # 適宜調整
            frame = overlay_image_alpha(frame, overlay_img, overlay_x, overlay_y, scale=0.7) # 0.7倍に縮小して表示

    cv2.imshow('Janken Game', frame)

    # キー入力処理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # Qキーで終了
        break
    elif key == ord('s') and game_state == "waiting_for_start": # Sキーでゲーム開始
        game_state = "countdown"
        countdown_start_time = time.time()
        # ゲーム開始時に変数をリセット
        player_predicted_hand = ""
        ai_choice = ""
        game_result = ""
    elif key == ord('r') and game_state == "show_result": # Rキーでリプレイ
        game_state = "countdown"
        countdown_start_time = time.time()
        # リプレイ時も変数をリセット
        player_predicted_hand = ""
        ai_choice = ""
        game_result = ""

# リソースの解放
cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import os
import random
from collections import deque, Counter

# Ganti path sesuai lokasi dataset di komputer Anda
DATASET_PATH = r"C:/Users/rizky/Downloads/individual_cards_2"
CAMERA_INDEX = 0

W_REQ, H_REQ = 200, 300
H_CORNER = 85
W_CORNER = 50

recent_labels = deque(maxlen=10)
stable_label = "Unknown"
stability_counter = 0

RANK_MAP = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'jack': 11, 'queen': 12, 'king': 13, 'ace': 14,
    'joker': 15,
    'j': 11, 'q': 12, 'k': 13, 'a': 14,
    'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
}

class WarGameEngine:
    def __init__(self, card_names_list):
        self.full_deck_names = card_names_list
        deck_copy = card_names_list.copy()
        random.shuffle(deck_copy)
        
        self.p_hand = deck_copy[:5] 
        self.c_hand = deck_copy[5:10]
        
        self.score_p = 0
        self.score_c = 0
        self.state = 'CPU_TURN' 
        self.cpu_current_card = None
        self.player_current_card = None
        self.round_result_text = ""

    def get_rank(self, card_name_str):
        name_lower = card_name_str.lower()
        best_match_len = 0
        best_val = 0
        for key, val in RANK_MAP.items():
            if key in name_lower:
                if len(key) > best_match_len:
                    best_match_len = len(key)
                    best_val = val
        return best_val

    def cpu_play(self):
        if len(self.c_hand) > 0:
            card = random.choice(self.c_hand)
            self.c_hand.remove(card)
            self.cpu_current_card = card
            self.state = 'PLAYER_TURN'
        else:
            self.state = 'GAME_OVER'

    def player_play(self, detected_card_name):
        self.player_current_card = detected_card_name
        if detected_card_name in self.p_hand:
            self.p_hand.remove(detected_card_name)

        rank_p = self.get_rank(self.player_current_card)
        rank_c = self.get_rank(self.cpu_current_card)
        
        if rank_p > rank_c:
            self.score_p += 1
            self.round_result_text = "PLAYER WINS!"
        elif rank_c > rank_p:
            self.score_c += 1
            self.round_result_text = "CPU WINS!"
        else:
            self.round_result_text = "DRAW!"
        self.state = 'RESULT'

    def next_round(self):
        if len(self.p_hand) == 0 or len(self.c_hand) == 0:
            self.state = 'GAME_OVER'
        else:
            self.player_current_card = None
            self.cpu_current_card = None
            self.state = 'CPU_TURN'

def load_card_corners():
    dataset_corners = []
    card_names = []
    
    print("[INFO] Loading dataset (Taking Corners)...")
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Path: {DATASET_PATH} not found")
        return [], []

    for file in os.listdir(DATASET_PATH):
        if file.lower().endswith((".jpg", ".png")):
            path = os.path.join(DATASET_PATH, file)
            img = cv2.imread(path)
            if img is None: continue

            img = cv2.resize(img, (W_REQ, H_REQ))
            corner = img[0:H_CORNER, 0:W_CORNER]
            
            dataset_corners.append(corner)
            card_names.append(os.path.splitext(file)[0])

    print(f"[INFO] Loaded {len(dataset_corners)} corners.")
    return dataset_corners, card_names

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] 
    rect[2] = pts[np.argmax(s)] 
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] 
    rect[3] = pts[np.argmax(diff)] 
    return rect

def warp_card(image, pts):
    rect = order_points(pts)
    dst = np.array([
        [0, 0], [W_REQ - 1, 0],
        [W_REQ - 1, H_REQ - 1], [0, H_REQ - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (W_REQ, H_REQ))

def match_corner_robust(cam_corner, dataset_corners, dataset_names):
    best_name = "Unknown"
    best_score = -1.0

    for i, template in enumerate(dataset_corners):
        res = cv2.matchTemplate(cam_corner, template, cv2.TM_CCOEFF_NORMED)
        score = res[0][0]

        if score > best_score:
            best_score = score
            best_name = dataset_names[i]
            
    return best_name, best_score

def process_frame(frame, dataset_corners, dataset_names):
    global stable_label, stability_counter
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 150)
    
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    current_label = "Unknown"
    current_roi_display = None 
    
    highest_score_seen = 0.0
    best_candidate_name = "None"

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000: continue 

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) == 4:
            try:
                warped_color = warp_card(frame, approx.reshape(4, 2))
                cam_corner = warped_color[0:H_CORNER, 0:W_CORNER]
                name, score = match_corner_robust(cam_corner, dataset_corners, dataset_names)
                
                if score > highest_score_seen:
                    highest_score_seen = score
                    best_candidate_name = name

                if score > 0.40: 
                    current_label = name
                    current_roi_display = cam_corner 
                    
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                    
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(frame, f"{name}", (cX - 40, cY), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.putText(frame, f"{int(score*100)}%", (cX - 20, cY + 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    break 
                else:
                    cv2.drawContours(frame, [approx], -1, (0, 255, 255), 2)
            except Exception as e:
                pass 

    if highest_score_seen > 0.2:
        pass

    recent_labels.append(current_label)
    if len(recent_labels) > 0:
        most_common = Counter(recent_labels).most_common(1)[0][0]
        if most_common == stable_label:
            stability_counter += 2 
        else:
            stability_counter -= 1 
        
        stability_counter = max(min(stability_counter, 15), -15)
        if stability_counter > 3: 
            stable_label = most_common

    return frame, current_roi_display

def draw_ui_overlay(frame, game, card_templates, card_names):
    h, w, _ = frame.shape
    
    cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
    cv2.putText(frame, f"YOU: {game.score_p}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"CPU: {game.score_c}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    cv2.putText(frame, "[R] RESET | [Q] QUIT", (w//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    for i in range(len(game.c_hand)):
        x_pos = 250 + (i * 50)
        y_pos = 90
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos+40, y_pos+60), (0, 0, 150), -1)
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos+40, y_pos+60), (255, 255, 255), 1)
        cv2.putText(frame, "?", (x_pos+10, y_pos+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)

    center_x = w // 2
    center_y = h // 2
    
    if game.cpu_current_card:
        try:
            if game.cpu_current_card in card_names:
                idx = card_names.index(game.cpu_current_card)
                card_img = cv2.resize(card_templates[idx], (100, 150))
                
                x_c = center_x - 50
                y_c = center_y - 180 
                
                frame[y_c:y_c+150, x_c:x_c+100] = card_img
                
                cv2.rectangle(frame, (x_c, y_c), (x_c+100, y_c+150), (0, 0, 255), 3)
                cv2.putText(frame, "CPU PLAYS", (x_c, y_c - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                label_clean = game.cpu_current_card.replace("_", " ").upper()
                cv2.putText(frame, label_clean, (x_c, y_c + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            print(f"Error drawing CPU card: {e}")

    if game.player_current_card and game.state == "RESULT":
        try:
            if game.player_current_card in card_names:
                idx = card_names.index(game.player_current_card)
                card_img = cv2.resize(card_templates[idx], (100, 150))
                
                x_p = center_x - 50
                y_p = center_y + 20
                
                frame[y_p:y_p+150, x_p:x_p+100] = card_img
                cv2.rectangle(frame, (x_p, y_p), (x_p+100, y_p+150), (0, 255, 0), 3)
                cv2.putText(frame, "YOU PLAYED", (x_p, y_p + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception:
            pass

    start_x_player = 50
    start_y_player = h - 150
    cv2.putText(frame, "YOUR HAND (Show card to camera):", (start_x_player, start_y_player - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
    
    for i, p_card_name in enumerate(game.p_hand):
        try:
            if p_card_name in card_names:
                idx = card_names.index(p_card_name)
                thumb = cv2.resize(card_templates[idx], (60, 90))
                x_pos = start_x_player + (i * 70)
                y_pos = start_y_player
                if y_pos+90 < h and x_pos+60 < w:
                    frame[y_pos:y_pos+90, x_pos:x_pos+60] = thumb
                    cv2.rectangle(frame, (x_pos, y_pos), (x_pos+60, y_pos+90), (255, 255, 0), 1)
        except Exception:
            pass

    status_text = ""
    instruction = ""
    status_color = (255, 255, 255)

    if game.state == "CPU_TURN":
        status_text = "CPU IS THINKING..."
        status_color = (200, 200, 200)
    elif game.state == "PLAYER_TURN":
        status_text = "YOUR TURN!"
        status_color = (0, 255, 0)
        if stable_label != "Unknown":
            instruction = f"DETECTED: {stable_label}. Press SPACE to Play!"
        else:
            instruction = "Show your card to the camera..."
            
    elif game.state == "RESULT":
        status_text = game.round_result_text
        if "PLAYER WINS" in status_text: status_color = (0, 255, 0)
        elif "CPU WINS" in status_text: status_color = (0, 0, 255)
        else: status_color = (0, 255, 255)
        instruction = "Press ENTER for Next Round"
    elif game.state == "GAME_OVER":
        status_text = "GAME OVER"
        status_color = (0, 0, 255)

    if status_text:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-100), (w, h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, status_text, (text_x, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    
    if instruction:
        inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        inst_x = (w - inst_size[0]) // 2
        cv2.putText(frame, instruction, (inst_x, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return frame

def main():
    imgs, names = load_card_corners()
    if not imgs: return

    game = WarGameEngine(names)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=== PCV WAR GAME (TEMPLATE MATCHING) ===")
    print("Press 'q' to quit, 'r' to reset.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame, debug_roi = process_frame(frame, imgs, names)

        if game.state == "CPU_TURN":
            game.cpu_play() 
            
        elif game.state == "PLAYER_TURN":
            if debug_roi is not None:
                debug_big = cv2.resize(debug_roi, (150, 255))
                cv2.imshow("Corner Debug", debug_big)
                
                cv2.rectangle(frame, (50, 400), (160, 580), (0, 255, 0), 2)
                frame[400:570, 50:150] = cv2.resize(debug_roi, (100, 170))

            key = cv2.waitKey(1) & 0xFF
            if key == 32: 
                if stable_label != "Unknown":
                    if stable_label in game.p_hand:
                        game.player_play(stable_label)
                        print(f"Player played: {stable_label}")
                    else:
                        print(f"Card {stable_label} not in hand!")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'): game = WarGameEngine(names)
        elif key == 13 and game.state == "RESULT": game.next_round() 
        elif key == ord('q'): break

        frame = draw_ui_overlay(frame, game, imgs, names)
        
        cv2.imshow("PCV War Game - Main", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
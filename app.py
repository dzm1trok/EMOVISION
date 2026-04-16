import cv2
import numpy as np
import threading
import queue
import time
import os
from collections import deque
from datetime import datetime


class EmotionVisionUltimate:
    def __init__(self, buffer_size=7, analyze_every=2):
        # Камера
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        # Параметры
        self.analyze_every = analyze_every
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.show_debug = False
        self.recording = False
        self.video_writer = None

        # История эмоций
        self.emotion_history = deque(maxlen=100)
        self.history_timestamps = deque(maxlen=100)

        # Эмоции
        self.emotion_buffer = deque(maxlen=buffer_size)
        self.last_analysis = None
        self.analysis_time = 0
        self.smooth_confidence = 0

        # Цветовая схема
        self.colors = {
            'bg': (15, 15, 20),
            'panel': (30, 30, 40),
            'panel_border': (60, 60, 80),
            'accent': (0, 200, 255),
            'accent_secondary': (255, 100, 200),
            'text': (240, 240, 240),
            'text_dim': (150, 150, 170),
            'success': (100, 255, 100),
            'warning': (255, 200, 100),
            'danger': (255, 100, 100)
        }

        self.emotion_colors = {
            'angry': (80, 80, 255),
            'disgust': (80, 180, 255),
            'fear': (80, 255, 255),
            'happy': (80, 255, 80),
            'sad': (255, 120, 80),
            'surprise': (255, 220, 80),
            'neutral': (160, 160, 180)
        }

        # ТОЛЬКО ASCII-символы для совместимости
        self.emotion_names = {
            'happy': 'HAPPY',
            'sad': 'SAD',
            'angry': 'ANGRY',
            'surprise': 'SURPRISE',
            'fear': 'FEAR',
            'disgust': 'DISGUST',
            'neutral': 'NEUTRAL'
        }

        # Загружаем детектор
        self.detector = None
        self.detector_name = ""
        self._init_detector()

        # Многопоточность
        self.analysis_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=1)
        self.running = True

        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.analysis_thread.start()

        # Окно
        self.window_name = 'Emotion Vision Ultimate'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1600, 900)

        # Шрифты
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_bold = cv2.FONT_HERSHEY_DUPLEX

    def _init_detector(self):
        """Инициализация детектора"""
        print("=" * 60)
        print("  EMOTION VISION ULTIMATE")
        print("=" * 60)

        # Пробуем FER
        try:
            from fer import FER
            self.detector = FER(mtcnn=True)
            self.detector_name = "FER (MTCNN)"
            print(f"[OK] Detector: {self.detector_name}")
            print("     Accuracy: High")
            return
        except ImportError:
            pass

        # DeepFace
        try:
            from deepface import DeepFace
            self.detector = DeepFaceWrapper()
            self.detector_name = "DeepFace"
            print(f"[OK] Detector: {self.detector_name}")
            print("     Accuracy: Medium")
            return
        except ImportError:
            pass

        print("[ERROR] No libraries installed!")
        print("        pip install fer")
        print("        or")
        print("        pip install deepface")
        input("\nPress Enter to exit...")
        exit(1)

    def _analysis_worker(self):
        """Фоновый анализ"""
        while self.running:
            try:
                frame = self.analysis_queue.get(timeout=0.1)
                start = time.time()

                result = self.detector.detect_emotions(frame)

                if result and len(result) > 0:
                    emotions = result[0]['emotions']
                    dominant = max(emotions, key=emotions.get)

                    # Взвешенная стабилизация
                    self.emotion_buffer.append(dominant)

                    weights = {}
                    for i, e in enumerate(self.emotion_buffer):
                        weight = (i + 1) / len(self.emotion_buffer)
                        weights[e] = weights.get(e, 0) + weight

                    emotion = max(weights, key=weights.get)

                    target_confidence = emotions[emotion] * 100

                    # Безопасное получение координат
                    box = result[0].get('box', [0, 0, 100, 100])
                    if isinstance(box, dict):
                        box = [box.get('x', 0), box.get('y', 0),
                               box.get('w', 100), box.get('h', 100)]
                    else:
                        box = [int(v) for v in box]

                    # История
                    self.emotion_history.append(emotion)
                    self.history_timestamps.append(time.time())

                    analysis = {
                        'emotion': emotion,
                        'confidence': target_confidence,
                        'all_emotions': {k: v * 100 for k, v in emotions.items()},
                        'face_location': box,
                        'dominant_raw': dominant
                    }

                    if self.result_queue.full():
                        self.result_queue.get()
                    self.result_queue.put(analysis)
                    self.analysis_time = (time.time() - start) * 1000

            except queue.Empty:
                continue
            except Exception as e:
                if self.show_debug:
                    print(f"Analysis error: {e}")

    def get_analysis(self):
        """Получение результата с интерполяцией"""
        try:
            new_analysis = self.result_queue.get_nowait()

            if self.last_analysis:
                alpha = 0.3
                new_analysis['confidence'] = (
                        alpha * new_analysis['confidence'] +
                        (1 - alpha) * self.last_analysis['confidence']
                )

            self.last_analysis = new_analysis

        except queue.Empty:
            pass

        return self.last_analysis

    def draw_rounded_rect(self, img, pt1, pt2, color, thickness=-1, radius=15):
        """Скругленный прямоугольник с тенью"""
        x1, y1 = pt1
        x2, y2 = pt2

        if thickness == -1:
            # Тень
            shadow_offset = 3
            cv2.rectangle(img, (x1 + radius + shadow_offset, y1 + shadow_offset),
                          (x2 - radius + shadow_offset, y2 + shadow_offset), (0, 0, 0), -1)

            # Основная фигура
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)

            # Углы
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
        else:
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    def draw_glow_text(self, img, text, pos, font, scale, color, glow_color, thickness=2, glow_radius=3):
        """Текст с свечением"""
        x, y = pos

        for offset in range(glow_radius, 0, -1):
            alpha = 0.3 + (glow_radius - offset) * 0.1
            glow = tuple(int(c * alpha) for c in glow_color)
            cv2.putText(img, text, (x, y), font, scale, glow, thickness + offset * 2)

        cv2.putText(img, text, pos, font, scale, color, thickness)

    def draw_history_graph(self, display, x, y, w, h):
        """График истории эмоций"""
        if len(self.emotion_history) < 2:
            return

        self.draw_rounded_rect(display, (x, y), (x + w, y + h), (40, 40, 50), -1, 10)

        cv2.putText(display, "HISTORY (10s)", (x + 10, y - 10),
                    self.font, 0.5, self.colors['text_dim'], 1)

        # Рисуем точки для каждой эмоции
        for emotion, color in self.emotion_colors.items():
            points = []
            for i, hist_emotion in enumerate(self.emotion_history):
                if hist_emotion == emotion:
                    px = x + int((i / len(self.emotion_history)) * w)
                    # Y позиция зависит от эмоции (разные уровни)
                    emotion_idx = list(self.emotion_colors.keys()).index(emotion)
                    py = y + h - 15 - (emotion_idx * (h // 8))
                    points.append((px, py))

            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(display, points[i], points[i + 1], color, 3)

        cv2.rectangle(display, (x, y), (x + w, y + h), self.colors['panel_border'], 2)

    def draw_results(self, frame):
        """Улучшенный интерфейс - ТОЛЬКО ASCII"""
        h, w = frame.shape[:2]
        display_w, display_h = 1280, 720

        scale_x = display_w / w
        scale_y = display_h / h

        if w != display_w or h != display_h:
            display = cv2.resize(frame, (display_w, display_h))
        else:
            display = frame.copy()

        # Легкое размытие фона
        display = cv2.GaussianBlur(display, (0, 0), 0.5)

        # Панель справа
        panel_x = display_w - 400
        panel_w = 380

        # Градиентный фон
        overlay = display.copy()
        for i in range(10):
            alpha = 0.85 - i * 0.02
            color = tuple(int(c * alpha) for c in (20, 20, 30))
            cv2.rectangle(overlay, (panel_x - 10 + i, 10 + i),
                          (display_w - 10 - i, display_h - 10 - i), color, -1)

        cv2.addWeighted(overlay, 0.9, display, 0.1, 0, display)

        # Рамка панели
        cv2.rectangle(display, (panel_x - 10, 10), (display_w - 10, display_h - 10),
                      self.colors['panel_border'], 2)

        analysis = self.last_analysis

        if analysis:
            emotion = analysis['emotion']
            confidence = analysis['confidence']
            color = self.emotion_colors.get(emotion, (128, 128, 128))

            # Большая карточка эмоции
            card_y = 40
            card_h = 160

            # Градиент
            for i in range(card_h):
                alpha = i / card_h
                grad_color = tuple(int(c * (0.7 + 0.3 * alpha)) for c in color)
                cv2.line(display, (panel_x, card_y + i),
                         (display_w - 30, card_y + i), grad_color, 1)

            self.draw_rounded_rect(display, (panel_x, card_y),
                                   (display_w - 30, card_y + card_h), color, 3, 20)

            # Название эмоции
            name = self.emotion_names.get(emotion, emotion.upper())
            self.draw_glow_text(display, name,
                                (panel_x + 20, card_y + 60),
                                self.font_bold, 1.2, (255, 255, 255), color, 2, 4)

            # Процент
            conf_text = f"{confidence:.1f}%"
            self.draw_glow_text(display, conf_text,
                                (panel_x + 20, card_y + 120),
                                self.font, 1.0, (255, 255, 255), color, 2, 3)

            # Круговой индикатор
            center = (display_w - 80, card_y + 80)
            radius = 35

            cv2.circle(display, center, radius, (60, 60, 70), -1)
            cv2.circle(display, center, radius, (100, 100, 110), 2)

            angle = int(360 * confidence / 100)
            cv2.ellipse(display, center, (radius - 5, radius - 5), -90, 0, angle, color, 8)

            # Метрики
            metrics_y = card_y + card_h + 30
            cv2.putText(display, f"Analysis: {self.analysis_time:.0f}ms",
                        (panel_x, metrics_y), self.font, 0.5, self.colors['text_dim'], 1)
            cv2.putText(display, f"Buffer: {len(self.emotion_buffer)}/{self.emotion_buffer.maxlen}",
                        (panel_x, metrics_y + 25), self.font, 0.5, self.colors['text_dim'], 1)

            # График истории
            self.draw_history_graph(display, panel_x, display_h - 200, panel_w - 20, 120)

            # Список всех эмоций
            list_y = card_y + card_h + 80
            emotions = analysis.get('all_emotions', {})

            sorted_emos = sorted(emotions.items(), key=lambda x: -x[1])

            for i, (emo, val) in enumerate(sorted_emos[:5]):
                bar_color = self.emotion_colors.get(emo, (128, 128, 128))
                y_pos = list_y + i * 28

                is_current = emo == emotion
                bg_color = (50, 50, 60) if is_current else (35, 35, 45)

                cv2.rectangle(display, (panel_x, y_pos - 18),
                              (display_w - 30, y_pos + 8), bg_color, -1)

                # Индикатор текущей
                if is_current:
                    cv2.circle(display, (panel_x + 12, y_pos - 5), 4, (255, 255, 255), -1)

                # Имя (ASCII)
                name = self.emotion_names.get(emo, emo.upper())[:10]
                text_color = (255, 255, 255) if is_current else self.colors['text']
                cv2.putText(display, name, (panel_x + 25, y_pos),
                            self.font, 0.5, text_color, 1)

                # Полоса
                bar_x = panel_x + 130
                bar_w = panel_w - 160
                fill_w = int((val / 100) * bar_w)

                cv2.rectangle(display, (bar_x, y_pos - 12), (bar_x + bar_w, y_pos + 2),
                              (50, 50, 60), -1)
                cv2.rectangle(display, (bar_x, y_pos - 12), (bar_x + fill_w, y_pos + 2),
                              bar_color, -1)

                cv2.putText(display, f"{val:.0f}%", (bar_x + bar_w + 10, y_pos),
                            self.font, 0.45, self.colors['text_dim'], 1)

            # Рамка вокруг лица
            face_loc = analysis.get('face_location')
            if face_loc and len(face_loc) >= 4:
                try:
                    x = int(face_loc[0] * scale_x)
                    y = int(face_loc[1] * scale_y)
                    fw = int(face_loc[2] * scale_x)
                    fh = int(face_loc[3] * scale_y)

                    # Пульсирующая рамка
                    pulse = int(3 + 2 * np.sin(time.time() * 4))

                    # Свечение
                    for offset in range(pulse + 5, pulse, -1):
                        alpha = 0.1 + (pulse + 5 - offset) * 0.05
                        glow = tuple(int(c * alpha) for c in color)
                        cv2.rectangle(display, (x - offset, y - offset),
                                      (x + fw + offset, y + fh + offset), glow, 1)

                    cv2.rectangle(display, (x, y), (x + fw, y + fh), color, pulse)

                    # Угловые метки
                    corner_len = 30
                    cv2.line(display, (x, y), (x + corner_len, y), (255, 255, 255), 2)
                    cv2.line(display, (x, y), (x, y + corner_len), (255, 255, 255), 2)
                    cv2.line(display, (x + fw, y), (x + fw - corner_len, y), (255, 255, 255), 2)
                    cv2.line(display, (x + fw, y), (x + fw, y + corner_len), (255, 255, 255), 2)
                    cv2.line(display, (x, y + fh), (x + corner_len, y + fh), (255, 255, 255), 2)
                    cv2.line(display, (x, y + fh), (x, y + fh - corner_len), (255, 255, 255), 2)
                    cv2.line(display, (x + fw, y + fh), (x + fw - corner_len, y + fh), (255, 255, 255), 2)
                    cv2.line(display, (x + fw, y + fh), (x + fw, y + fh - corner_len), (255, 255, 255), 2)

                    # Подпись
                    label = f"  {self.emotion_names.get(emotion, emotion)}  "
                    (tw, th), _ = cv2.getTextSize(label, self.font_bold, 0.6, 2)

                    label_x = x + (fw - tw) // 2
                    label_y = y - 15

                    cv2.rectangle(display, (label_x - 5, label_y - th - 5),
                                  (label_x + tw + 5, label_y + 5), color, -1)
                    cv2.rectangle(display, (label_x - 5, label_y - th - 5),
                                  (label_x + tw + 5, label_y + 5), (255, 255, 255), 2)

                    cv2.putText(display, label, (label_x, label_y),
                                self.font_bold, 0.6, (255, 255, 255), 2)

                except Exception as e:
                    if self.show_debug:
                        print(f"Draw error: {e}")
        else:
            # Нет лица
            pulse = abs(np.sin(time.time() * 2)) * 0.5 + 0.3

            no_face_text = "NO FACE"
            (tw, th), _ = cv2.getTextSize(no_face_text, self.font_bold, 1.0, 2)
            text_x = panel_x + (panel_w - tw) // 2
            text_y = 150

            bg_color = tuple(int(c * pulse) for c in (60, 60, 70))
            cv2.rectangle(display, (panel_x + 20, text_y - th - 10),
                          (display_w - 50, text_y + 20), bg_color, -1)

            cv2.putText(display, no_face_text, (text_x, text_y),
                        self.font_bold, 1.0, self.colors['warning'], 2)

            cv2.putText(display, "Position face in camera", (panel_x + 40, 200),
                        self.font, 0.6, self.colors['text_dim'], 1)

        # Верхняя панель
        info_y = 30

        cv2.putText(display, "EMOTION VISION", (20, info_y),
                    self.font_bold, 0.8, self.colors['accent'], 2)

        fps_color = self.colors['success'] if self.fps > 25 else self.colors['warning'] if self.fps > 15 else \
        self.colors['danger']
        cv2.putText(display, f"FPS: {self.fps:.0f}", (300, info_y),
                    self.font, 0.6, fps_color, 1)

        cv2.putText(display, f"Detector: {self.detector_name}", (420, info_y),
                    self.font, 0.5, self.colors['text_dim'], 1)

        # Индикатор записи
        if self.recording:
            rec_color = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (100, 100, 100)
            cv2.circle(display, (display_w - 100, 30), 8, rec_color, -1)
            cv2.putText(display, "REC", (display_w - 85, 35),
                        self.font, 0.5, rec_color, 1)

        # Нижняя панель управления
        bar_y = display_h - 50
        cv2.rectangle(display, (0, bar_y), (display_w, display_h), (20, 20, 30), -1)
        cv2.line(display, (0, bar_y), (display_w, bar_y), self.colors['panel_border'], 2)

        controls = [
            ("Q", "Exit"),
            ("R", "Record"),
            ("D", "Debug"),
            ("S", "Screenshot")
        ]

        x_offset = 30
        for key, action in controls:
            # Клавиша
            self.draw_rounded_rect(display, (x_offset, bar_y + 10),
                                   (x_offset + 35, bar_y + 40), self.colors['accent'], -1, 5)
            cv2.putText(display, key, (x_offset + 10, bar_y + 32),
                        self.font_bold, 0.5, (0, 0, 0), 1)

            # Действие
            cv2.putText(display, action, (x_offset + 45, bar_y + 32),
                        self.font, 0.5, self.colors['text'], 1)

            x_offset += 140

        return display

    def save_screenshot(self, frame):
        """Сохранение скриншота"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emotion_screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"[OK] Screenshot saved: {filename}")

    def toggle_recording(self, frame):
        """Включение/выключение записи"""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_recording_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            self.recording = True
            print(f"[OK] Recording started: {filename}")
        else:
            self.video_writer.release()
            self.recording = False
            print("[OK] Recording stopped")

    def run(self):
        """Основной цикл"""
        print("\nControls:")
        print("  Q - Exit")
        print("  R - Record video")
        print("  D - Debug mode")
        print("  S - Screenshot")
        print("\nStarting...")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Запись
            if self.recording and self.video_writer:
                self.video_writer.write(frame)

            # FPS
            self.frame_count += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_fps_time = time.time()

            # Анализ
            if self.frame_count % self.analyze_every == 0:
                try:
                    if self.analysis_queue.full():
                        self.analysis_queue.get_nowait()
                    self.analysis_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

            self.get_analysis()
            display = self.draw_results(frame)
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.toggle_recording(frame)
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                print(f"[OK] Debug: {self.show_debug}")
            elif key == ord('s'):
                self.save_screenshot(display)

        self.shutdown()

    def shutdown(self):
        """Завершение"""
        self.running = False
        if self.recording and self.video_writer:
            self.video_writer.release()
        self.analysis_thread.join(timeout=1.0)
        self.cap.release()
        cv2.destroyAllWindows()
        print("[OK] System stopped")


class DeepFaceWrapper:
    """Обертка для DeepFace"""

    def __init__(self):
        from deepface import DeepFace
        self.analyze = DeepFace.analyze

    def detect_emotions(self, frame):
        try:
            result = self.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if not result or len(result) == 0:
                return None

            emotions = result[0]['emotion']
            region = result[0].get('region', {})

            return [{
                'emotions': {k: v / 100 for k, v in emotions.items()},
                'box': [
                    region.get('x', 0),
                    region.get('y', 0),
                    region.get('w', 100),
                    region.get('h', 100)
                ]
            }]
        except:
            return None


if __name__ == "__main__":
    system = EmotionVisionUltimate(buffer_size=7, analyze_every=2)
    system.run()
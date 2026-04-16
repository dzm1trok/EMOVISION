import cv2
import numpy as np
import threading
import queue
import time
import os
from collections import deque


class EmotionVisionPro:
    def __init__(self, buffer_size=5, analyze_every=3):
        # Камера
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Параметры
        self.analyze_every = analyze_every
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.show_debug = False

        # Эмоции
        self.emotion_buffer = deque(maxlen=buffer_size)
        self.last_analysis = None
        self.analysis_time = 0

        # Цвета
        self.colors = {
            'bg': (30, 30, 30),
            'panel': (45, 45, 48),
            'accent': (0, 150, 255),
            'text': (220, 220, 220),
            'text_dim': (150, 150, 150)
        }

        self.emotion_colors = {
            'angry': (60, 60, 255),
            'disgust': (60, 160, 255),
            'fear': (60, 255, 255),
            'happy': (60, 255, 60),
            'sad': (255, 100, 60),
            'surprise': (255, 200, 60),
            'neutral': (180, 180, 180)
        }

        # Загружаем детектор
        self.detector = None
        self._init_detector()

        # Многопоточность
        self.analysis_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=1)
        self.running = True

        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.analysis_thread.start()

        # Окно
        self.window_name = 'Emotion Vision Pro'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1400, 900)

    def _init_detector(self):
        """Инициализация детектора"""
        print("Загрузка модели...")

        # Пробуем FER
        try:
            from fer import FER
            self.detector = FER(mtcnn=False)
            print("✓ FER загружен")
            return
        except ImportError:
            pass

        # Пробуем DeepFace
        try:
            from deepface import DeepFace
            self.detector = DeepFaceWrapper()
            print("✓ DeepFace загружен")
            return
        except ImportError:
            pass

        print("Установите: pip install fer")
        input("Enter для выхода...")
        exit(1)

    def _analysis_worker(self):
        """Фоновый анализ"""
        while self.running:
            try:
                frame = self.analysis_queue.get(timeout=0.1)
                start = time.time()

                result = self.detector.detect_emotions(frame)

                if result:
                    emotions = result[0]['emotions']
                    dominant = max(emotions, key=emotions.get)

                    self.emotion_buffer.append(dominant)

                    # Стабилизация
                    if len(self.emotion_buffer) >= 3:
                        emotion = max(set(self.emotion_buffer),
                                      key=lambda x: list(self.emotion_buffer).count(x))
                    else:
                        emotion = dominant

                    # Безопасное получение координат
                    box = result[0].get('box', [0, 0, 100, 100])
                    # Преобразуем в плоский список чисел
                    if isinstance(box, dict):
                        box = [box.get('x', 0), box.get('y', 0),
                               box.get('w', 100), box.get('h', 100)]
                    else:
                        box = [int(v) for v in box]

                    analysis = {
                        'emotion': emotion,
                        'confidence': emotions[dominant] * 100,
                        'all_emotions': {k: v * 100 for k, v in emotions.items()},
                        'face_location': box
                    }

                    if self.result_queue.full():
                        self.result_queue.get()
                    self.result_queue.put(analysis)
                    self.analysis_time = (time.time() - start) * 1000

            except queue.Empty:
                continue
            except Exception as e:
                if self.show_debug:
                    print(f"Error: {e}")

    def get_analysis(self):
        try:
            self.last_analysis = self.result_queue.get_nowait()
        except queue.Empty:
            pass
        return self.last_analysis

    def draw_results(self, frame):
        h, w = frame.shape[:2]
        display_w, display_h = 1280, 720

        # Масштабирование
        scale_x = display_w / w
        scale_y = display_h / h

        if w != display_w or h != display_h:
            display = cv2.resize(frame, (display_w, display_h))
        else:
            display = frame.copy()

        # Панель справа
        panel_x = display_w - 360

        # Фон
        overlay = display.copy()
        cv2.rectangle(overlay, (panel_x - 10, 10), (display_w - 10, display_h - 10), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)

        # Заголовок
        cv2.putText(display, "EMOTION ANALYSIS", (panel_x, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors['accent'], 2)

        cv2.putText(display, f"FPS: {self.fps:.0f}", (panel_x, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_dim'], 1)

        analysis = self.last_analysis

        if analysis:
            emotion = analysis['emotion']
            confidence = analysis['confidence']
            color = self.emotion_colors.get(emotion, (128, 128, 128))

            # Карточка эмоции
            card_y = 110
            cv2.rectangle(display, (panel_x, card_y), (display_w - 20, card_y + 120), color, -1)
            cv2.rectangle(display, (panel_x, card_y), (display_w - 20, card_y + 120), (255, 255, 255), 2)

            cv2.putText(display, emotion.upper(), (panel_x + 20, card_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(display, f"{confidence:.1f}%", (panel_x + 20, card_y + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Список эмоций
            list_y = card_y + 140
            emotions = analysis.get('all_emotions', {})

            for i, emo in enumerate(['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']):
                val = emotions.get(emo, 0)
                bar_color = self.emotion_colors.get(emo, (128, 128, 128))
                y_pos = list_y + i * 35

                if emo == emotion:
                    cv2.rectangle(display, (panel_x, y_pos - 20), (display_w - 20, y_pos + 10),
                                  (60, 60, 60), -1)

                cv2.putText(display, emo[:8].upper(), (panel_x + 10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255) if emo == emotion else self.colors['text'], 1)

                bar_x = panel_x + 100
                bar_w = 180
                fill_w = int((val / 100) * bar_w)

                cv2.rectangle(display, (bar_x, y_pos - 12), (bar_x + bar_w, y_pos + 2),
                              (60, 60, 60), -1)
                cv2.rectangle(display, (bar_x, y_pos - 12), (bar_x + fill_w, y_pos + 2),
                              bar_color, -1)
                cv2.putText(display, f"{val:.0f}%", (bar_x + bar_w + 10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['text_dim'], 1)

            # Рамка вокруг лица - ИСПРАВЛЕНО
            face_loc = analysis.get('face_location')
            if face_loc and len(face_loc) >= 4:
                try:
                    # Безопасное извлечение и масштабирование
                    x = int(face_loc[0] * scale_x)
                    y = int(face_loc[1] * scale_y)
                    fw = int(face_loc[2] * scale_x)
                    fh = int(face_loc[3] * scale_y)

                    cv2.rectangle(display, (x, y), (x + fw, y + fh), color, 3)

                    # Подпись
                    label = f"{emotion} {confidence:.0f}%"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(display, (x, y - th - 15), (x + tw + 20, y), color, -1)
                    cv2.putText(display, label, (x + 10, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                except Exception as e:
                    if self.show_debug:
                        print(f"Draw error: {e}")
        else:
            # Нет лица
            cv2.putText(display, "NO FACE DETECTED", (panel_x, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.putText(display, "Position face in camera", (panel_x, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_dim'], 1)

        # Подсказки
        hint_y = display_h - 40
        for i, (key, action) in enumerate([("Q", "Quit"), ("D", "Debug")]):
            x = 20 + i * 120
            cv2.rectangle(display, (x, hint_y - 20), (x + 30, hint_y + 5), self.colors['accent'], -1)
            cv2.putText(display, key, (x + 8, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(display, action, (x + 40, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)

        return display

    def run(self):
        print("=" * 60)
        print("  EMOTION VISION PRO")
        print("=" * 60)
        print("Управление: Q - выход, D - отладка")
        print("=" * 60)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            self.frame_count += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_fps_time = time.time()

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
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                print(f"Debug: {self.show_debug}")

        self.shutdown()

    def shutdown(self):
        self.running = False
        self.analysis_thread.join(timeout=1.0)
        self.cap.release()
        cv2.destroyAllWindows()
        print("Система остановлена")


class DeepFaceWrapper:
    """Обертка для DeepFace с единым интерфейсом"""

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

            # Приводим к формату FER
            emotions = result[0]['emotion']
            region = result[0].get('region', {})

            return [{
                'emotions': {k: v / 100 for k, v in emotions.items()},  # Нормализация 0-1
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
    system = EmotionVisionPro(buffer_size=5, analyze_every=3)
    system.run()
    
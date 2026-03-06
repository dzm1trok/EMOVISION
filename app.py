import cv2
from deepface import DeepFace
import time
from collections import deque


class EmotionVision:
    def __init__(self, buffer_size=10, analyze_every=5):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.emotion_buffer = deque(maxlen=buffer_size)
        self.current_emotion = "neutral"
        self.confidence = 0.0
        self.frame_count = 0
        self.analyze_every = analyze_every

        self.emotion_colors = {
            'angry': (0, 0, 255),
            'disgust': (0, 140, 255),
            'fear': (0, 255, 255),
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'surprise': (255, 255, 0),
            'neutral': (128, 128, 128)
        }

        self.last_analysis = None

    def analyze_frame(self, frame):
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if result and len(result) > 0:
                emotions = result[0]['emotion']
                dominant = result[0]['dominant_emotion']
                confidence = emotions[dominant]

                self.emotion_buffer.append(dominant)

                if len(self.emotion_buffer) == self.emotion_buffer.maxlen:
                    self.current_emotion = max(
                        set(self.emotion_buffer),
                        key=self.emotion_buffer.count
                    )

                return {
                    'emotion': self.current_emotion,
                    'confidence': confidence,
                    'all_emotions': emotions,
                    'face_location': result[0].get('region', None)
                }

        except Exception as e:
            print(f"Analysis error: {e}")

        return None

    def draw_results(self, frame, analysis):
        h, w = frame.shape[:2]

        # Фон панели
        cv2.rectangle(frame, (10, 10), (350, 180), (0, 0, 0), -1)

        if analysis:
            emotion = analysis['emotion']
            confidence = analysis['confidence']
            color = self.emotion_colors.get(emotion, (128, 128, 128))

            # Рамка цвета эмоции
            cv2.rectangle(frame, (10, 10), (350, 180), color, 3)

            # Название эмоции
            text = f"{emotion.upper()}: {confidence:.1f}%"
            cv2.putText(frame, text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Топ-3 эмоции с графиками
            emotions = analysis['all_emotions']
            y_offset = 80

            for emo, val in sorted(emotions.items(), key=lambda x: -x[1])[:3]:
                bar_color = self.emotion_colors.get(emo, (128, 128, 128))

                # Фон полосы
                cv2.rectangle(frame, (20, y_offset), (220, y_offset + 20), (50, 50, 50), -1)

                # Заполнение
                bar_width = int(val * 2)
                cv2.rectangle(frame, (20, y_offset), (20 + bar_width, y_offset + 20), bar_color, -1)

                # Текст
                label = f"{emo:8}: {val:.1f}%"
                cv2.putText(frame, label, (230, y_offset + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                y_offset += 30

            # Рамка вокруг лица (исправлено)
            face_loc = analysis.get('face_location')
            if face_loc and isinstance(face_loc, dict):
                # Безопасное извлечение координат
                x = face_loc.get('x', 0)
                y = face_loc.get('y', 0)
                w_face = face_loc.get('w', 0)
                h_face = face_loc.get('h', 0)

                if w_face > 0 and h_face > 0:
                    cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), color, 2)
        else:
            # Нет анализа — ждём
            cv2.putText(frame, "Detecting...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)
            cv2.rectangle(frame, (10, 10), (350, 180), (128, 128, 128), 3)

        return frame

    def run(self):
        print("=" * 50)
        print("Emotion-Vision запущена")
        print(f"Анализ каждые {self.analyze_every} кадров")
        print("Нажмите 'Q' для выхода")
        print("=" * 50)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Ошибка: не удалось получить кадр с камеры")
                break

            # Зеркальное отражение
            frame = cv2.flip(frame, 1)

            # Счётчик кадров
            self.frame_count += 1

            # Анализ каждые N кадров
            if self.frame_count % self.analyze_every == 0:
                print(f"Анализ кадра #{self.frame_count}...")
                self.last_analysis = self.analyze_frame(frame)

                if self.last_analysis:
                    print(f"  -> Обнаружено: {self.last_analysis['emotion']}")

            # Отрисовка
            display = self.draw_results(frame.copy(), self.last_analysis)

            # Показать FPS
            fps_text = f"Frame: {self.frame_count}"
            cv2.putText(display, fps_text, (display.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Emotion-Vision', display)

            # Выход по 'Q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nЗавершение работы...")
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("Система остановлена")


if __name__ == "__main__":
    system = EmotionVision(buffer_size=5, analyze_every=3)
    system.run()
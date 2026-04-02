import cv2
import numpy as np
import threading
import queue
import time
import os
from collections import deque
import urllib.request


class EmotionVisionOptimized:
    def __init__(self, buffer_size=5, analyze_every=2):
        # Камера
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Параметры
        self.analyze_every = analyze_every
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

        # Эмоции и паттерны
        self.emotion_buffer = deque(maxlen=buffer_size)
        self.current_emotion = "neutral"
        self.confidence = 0.0
        self.last_analysis = None
        self.analysis_time = 0

        # Цвета для эмоций
        self.emotion_colors = {
            'angry': (0, 0, 255),  # Красный
            'disgust': (0, 140, 255),  # Оранжевый
            'fear': (0, 255, 255),  # Желтый
            'happy': (0, 255, 0),  # Зеленый
            'sad': (255, 0, 0),  # Синий
            'surprise': (255, 255, 0),  # Голубой
            'neutral': (128, 128, 128)  # Серый
        }

        # ROI индексы для разных частей лица (относительные координаты)
        self.roi_indices = {
            'eyes': (0.2, 0.5, 0.2, 0.8),  # y_start, y_end, x_start, x_end
            'mouth': (0.6, 0.9, 0.3, 0.7),
            'forehead': (0.1, 0.3, 0.3, 0.7),
            'cheeks': (0.4, 0.7, 0.1, 0.9)
        }

        # Многопоточность
        self.analysis_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=1)
        self.running = True

        # Проверяем FER
        self.use_fer = self._check_fer()
        self._init_detector()

        # Запуск потока анализа
        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.analysis_thread.start()

    def _check_fer(self):
        """Проверка установки FER"""
        try:
            from fer import FER
            return True
        except ImportError:
            print("FER не установлен. Используем улучшенный OpenCV анализ")
            print("Для установки FER: pip install fer")
            return False

    def _download_file(self, url, filename):
        """Скачивание файла если не существует"""
        if not os.path.exists(filename):
            print(f"Скачивание {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"Готово: {filename}")
                return True
            except Exception as e:
                print(f"Ошибка загрузки: {e}")
                return False
        return True

    def _init_detector(self):
        """Инициализация детектора"""
        start_time = time.time()

        if self.use_fer:
            from fer import FER
            self.detector = FER(mtcnn=False)
            print(f"FER инициализирован за {time.time() - start_time:.2f}с")
        else:
            # OpenCV DNN
            model_file = "res10_300x300_ssd_iter_140000.caffemodel"
            config_file = "deploy.prototxt"

            base_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"

            success = True
            if not os.path.exists(model_file):
                success = self._download_file(base_url + model_file, model_file)
            if not os.path.exists(config_file):
                success = self._download_file(
                    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/" + config_file,
                    config_file
                ) and success

            if success and os.path.exists(model_file) and os.path.exists(config_file):
                self.face_net = cv2.dnn.readNetFromCaffe(config_file, model_file)
                self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                print(f"OpenCV DNN инициализирован за {time.time() - start_time:.2f}с")
            else:
                print("Используем каскады Haar")
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )

    def _get_roi(self, face_img, roi_name):
        """Получение региона интереса (ROI)"""
        h, w = face_img.shape[:2]
        y_s, y_e, x_s, x_e = self.roi_indices[roi_name]
        y1, y2 = int(h * y_s), int(h * y_e)
        x1, x2 = int(w * x_s), int(w * x_e)
        return face_img[y1:y2, x1:x2]

    def _analyze_face_geometry(self, gray_face):
        """Анализ геометрии лица через ключевые точки"""
        # Используем бинаризацию для выделения контуров
        _, thresh = cv2.threshold(gray_face, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Находим контуры
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features = {}

        # Анализ глаз (верхняя часть)
        eyes_roi = self._get_roi(gray_face, 'eyes')
        eyes_edges = cv2.Canny(eyes_roi, 50, 150)
        eyes_activity = np.sum(eyes_edges > 0) / eyes_edges.size
        features['eyes_activity'] = eyes_activity

        # Анализ рта (нижняя часть)
        mouth_roi = self._get_roi(gray_face, 'mouth')
        mouth_edges = cv2.Canny(mouth_roi, 50, 150)
        mouth_activity = np.sum(mouth_edges > 0) / mouth_edges.size
        features['mouth_activity'] = mouth_activity

        # Соотношение ширины и высоты рта
        mouth_thresh = cv2.adaptiveThreshold(mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
        mouth_contours, _ = cv2.findContours(mouth_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if mouth_contours:
            largest = max(mouth_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            features['mouth_ratio'] = w / max(h, 1)
            features['mouth_area'] = cv2.contourArea(largest) / mouth_roi.size
        else:
            features['mouth_ratio'] = 1.0
            features['mouth_area'] = 0

        # Лоб (морщины/напряжение)
        forehead_roi = self._get_roi(gray_face, 'forehead')
        forehead_std = np.std(forehead_roi)
        features['forehead_tension'] = forehead_std

        # Щеки (припухлость/цвет)
        cheeks_roi = self._get_roi(gray_face, 'cheeks')
        features['cheeks_brightness'] = np.mean(cheeks_roi)

        # Общая яркость и контраст
        features['brightness'] = np.mean(gray_face)
        features['contrast'] = np.std(gray_face)

        # Асимметрия (разница левой и правой половин)
        h, w = gray_face.shape
        left_half = gray_face[:, :w // 2]
        right_half = gray_face[:, w // 2:]
        features['asymmetry'] = np.abs(np.mean(left_half) - np.mean(right_half))

        return features

    def _classify_emotion_advanced(self, features):
        """Расширенная классификация на 7 эмоций"""
        scores = {
            'neutral': 0,
            'happy': 0,
            'sad': 0,
            'angry': 0,
            'surprise': 0,
            'fear': 0,
            'disgust': 0
        }

        # Параметры для анализа
        brightness = features['brightness']
        contrast = features['contrast']
        mouth_ratio = features['mouth_ratio']
        mouth_area = features['mouth_area']
        mouth_activity = features['mouth_activity']
        eyes_activity = features['eyes_activity']
        forehead_tension = features['forehead_tension']
        asymmetry = features['asymmetry']

        # HAPPY: широкий рот, высокая активность рта, высокая яркость
        if mouth_ratio > 2.0 and mouth_area > 0.15 and brightness > 120:
            scores['happy'] += 3
        if mouth_activity > 0.1 and brightness > 110:
            scores['happy'] += 2

        # SAD: низкая яркость, низкая активность рта, "опущенные" уголки (асимметрия)
        if brightness < 100 and mouth_activity < 0.08:
            scores['sad'] += 2
        if contrast < 35 and mouth_ratio < 1.5:
            scores['sad'] += 2
        if asymmetry > 5:
            scores['sad'] += 1

        # ANGRY: высокое напряжение лба, низкая яркость, высокий контраст
        if forehead_tension > 45 and brightness < 110:
            scores['angry'] += 3
        if contrast > 50 and eyes_activity > 0.15:
            scores['angry'] += 2
        if mouth_ratio < 1.3 and mouth_area < 0.1:
            scores['angry'] += 1

        # SURPRISE: высокая активность глаз, овальный рот, высокий контраст
        if eyes_activity > 0.12 and mouth_ratio > 1.3 and mouth_ratio < 2.0:
            scores['surprise'] += 3
        if forehead_tension > 40 and mouth_area > 0.12:
            scores['surprise'] += 2

        # FEAR: высокая активность глаз + напряжение, но не широкий рот
        if eyes_activity > 0.14 and forehead_tension > 35 and mouth_ratio < 1.8:
            scores['fear'] += 3
        if brightness < 105 and contrast > 45:
            scores['fear'] += 2

        # DISGUST: приподнятая верхняя губа (асимметрия), низкая активность рта
        if asymmetry > 8 and mouth_activity < 0.1:
            scores['disgust'] += 3
        if forehead_tension > 30 and brightness > 115:
            scores['disgust'] += 1

        # NEUTRAL: средние значения по всем параметрам
        neutral_indicators = [
            100 < brightness < 130,
            30 < contrast < 50,
            1.4 < mouth_ratio < 1.9,
            0.05 < mouth_activity < 0.12,
            forehead_tension < 40
        ]
        scores['neutral'] = sum(neutral_indicators)

        # Находим победителя
        dominant = max(scores, key=scores.get)
        max_score = scores[dominant]

        # Нормализуем в вероятности (softmax-подобное)
        exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
        sum_exp = sum(exp_scores.values())
        probabilities = {k: (v / sum_exp) * 100 for k, v in exp_scores.items()}

        # Уверенность - разница между топ-1 и топ-2
        sorted_scores = sorted(probabilities.values(), reverse=True)
        confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 50

        return dominant, min(confidence * 2, 100), probabilities

    def _analyze_with_opencv(self, frame):
        """Улучшенный анализ через OpenCV"""
        h, w = frame.shape[:2]

        # Детекция лица
        if hasattr(self, 'face_net'):
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104.0, 177.0, 123.0])
            self.face_net.setInput(blob)
            detections = self.face_net.forward()

            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    faces.append((x1, y1, x2 - x1, y2 - y1))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None

        # Берем первое лицо
        x, y, w_face, h_face = faces[0]

        # Вырезаем и нормализуем
        face_roi = frame[y:y + h_face, x:x + w_face]
        if face_roi.size == 0:
            return None

        # Приводим к стандартному размеру
        face_normalized = cv2.resize(face_roi, (128, 128))
        gray_face = cv2.cvtColor(face_normalized, cv2.COLOR_BGR2GRAY)

        # Анализируем
        features = self._analyze_face_geometry(gray_face)
        emotion, confidence, all_emotions = self._classify_emotion_advanced(features)

        return {
            'box': [x, y, w_face, h_face],
            'emotions': all_emotions,
            'dominant': emotion,
            'confidence': confidence,
            'features': features  # Для отладки
        }

    def _analysis_worker(self):
        """Фоновый поток анализа"""
        while self.running:
            try:
                frame = self.analysis_queue.get(timeout=0.1)
                start = time.time()

                if self.use_fer:
                    result = self.detector.detect_emotions(frame)
                    if result:
                        emotions = result[0]['emotions']
                        dominant = max(emotions, key=emotions.get)

                        self.emotion_buffer.append(dominant)
                        emotion = max(set(self.emotion_buffer), key=self.emotion_buffer.count) \
                            if len(self.emotion_buffer) == self.emotion_buffer.maxlen else dominant

                        analysis = {
                            'emotion': emotion,
                            'confidence': emotions[dominant] * 100,
                            'all_emotions': {k: v * 100 for k, v in emotions.items()},
                            'face_location': result[0]['box']
                        }

                        if self.result_queue.full():
                            self.result_queue.get()
                        self.result_queue.put(analysis)
                        self.analysis_time = (time.time() - start) * 1000

                else:
                    result = self._analyze_with_opencv(frame)
                    if result:
                        dominant = result['dominant']
                        self.emotion_buffer.append(dominant)
                        emotion = max(set(self.emotion_buffer), key=self.emotion_buffer.count) \
                            if len(self.emotion_buffer) == self.emotion_buffer.maxlen else dominant

                        analysis = {
                            'emotion': emotion,
                            'confidence': result['confidence'],
                            'all_emotions': result['emotions'],
                            'face_location': result['box']
                        }

                        if self.result_queue.full():
                            self.result_queue.get()
                        self.result_queue.put(analysis)
                        self.analysis_time = (time.time() - start) * 1000

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis error: {e}")

    def get_analysis(self):
        """Получить последний результат"""
        try:
            self.last_analysis = self.result_queue.get_nowait()
        except queue.Empty:
            pass
        return self.last_analysis

    def draw_results(self, frame):
        """Отрисовка с расширенной информацией"""
        h, w = frame.shape[:2]

        # Панель побольше для 7 эмоций
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        analysis = self.last_analysis

        if analysis:
            emotion = analysis['emotion']
            confidence = analysis['confidence']
            color = self.emotion_colors.get(emotion, (128, 128, 128))

            # Режим
            mode = "FER" if self.use_fer else "OpenCV-Advanced"
            cv2.putText(frame, f"[{mode}]", (390, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.rectangle(frame, (10, 10), (380, 280), color, 3)

            # Основная эмоция крупно
            cv2.putText(frame, f"{emotion.upper()}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, f"{confidence:.1f}%", (200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Все 7 эмоций с графиками
            emotions = analysis.get('all_emotions', {})
            y_offset = 80

            for emo in ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']:
                val = emotions.get(emo, 0)
                bar_color = self.emotion_colors.get(emo, (128, 128, 128))
                bar_width = int(val * 2.5)  # Масштаб для наглядности

                # Индикатор текущей эмоции
                marker = "▶" if emo == emotion else " "

                cv2.rectangle(frame, (30, y_offset), (280, y_offset + 18), (50, 50, 50), -1)
                cv2.rectangle(frame, (30, y_offset), (30 + bar_width, y_offset + 18), bar_color, -1)
                cv2.putText(frame, f"{marker} {emo[:8]:8}:{val:5.1f}%", (290, y_offset + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y_offset += 22

            # Рамка вокруг лица
            face_loc = analysis.get('face_location')
            if face_loc and len(face_loc) == 4:
                x, y, w_face, h_face = face_loc
                cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), color, 2)

                # Подпись над рамкой
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "Detecting face...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)
            cv2.rectangle(frame, (10, 10), (380, 280), (128, 128, 128), 3)

        # FPS
        fps_text = f"FPS: {self.fps:.1f} | Analysis: {self.analysis_time:.0f}ms"
        cv2.putText(frame, fps_text, (w - 400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Подсказка
        hint = "Show: happy, sad, angry, surprise, fear, disgust, neutral"
        cv2.putText(frame, hint, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    def run(self):
        """Основной цикл"""
        print("=" * 60)
        print("Emotion-Vision ADVANCED")
        mode = "FER (нейросеть)" if self.use_fer else "OpenCV (геометрия + паттерны)"
        print(f"Режим: {mode}")
        print("Поддерживаемые эмоции: happy, sad, angry, surprise, fear, disgust, neutral")
        print(f"Анализ каждые {self.analyze_every} кадров")
        print("Нажмите 'Q' для выхода")
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
            cv2.imshow('Emotion-Vision ADVANCED', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.shutdown()

    def shutdown(self):
        """Завершение"""
        self.running = False
        self.analysis_thread.join(timeout=1.0)
        self.cap.release()
        cv2.destroyAllWindows()
        print("Система остановлена")


if __name__ == "__main__":
    system = EmotionVisionOptimized(buffer_size=3, analyze_every=2)
    system.run()

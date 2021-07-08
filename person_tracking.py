import cv2
import face_recognition
from numpy import result_type

from gaze.gaze_tracking import GazeTracking
from utils import video_reader


class PersonDetector:
    def __init__(self, config):
        self.known_names = []
        self.known_encodings = []

        self.gaze_detector = GazeTracking()

        self.width = config["width"]
        self.height = config["height"]

        self.gaze_hlimits = config.get("gaze_hlimits")
        if self.gaze_hlimits is None:
            self.gaze_hlimits = (0.2, 0.8)

        self.gaze_vlimits = config.get("gaze_vlimits")
        if self.gaze_vlimits is None:
            self.gaze_vlimits = (0.2, 0.8)

        person_image = config.get("person_image")
        person_name = config.get("person_name")

        if person_name is None:
            person_name = "student"

        if person_image is not None:
            person_encoding = face_recognition.face_encodings(person_image)[0]
            self.known_names += [person_name]
            self.known_encodings += [person_encoding]

    def detect_persons(self, frame):
        found_persons = []
        scale = 4

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(
            rgb_frame, (0, 0), fx=1 / scale, fy=1 / scale
        )  # resize for optimization
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for location, encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_encodings, encoding)
            top, right, bottom, left = location
            person_face_bb = [
                int(left * scale),
                int(top * scale),
                int((right - left) * scale),
                int((bottom - top) * scale),
            ]

            if True in matches:
                person_name = self.known_names[matches.index(True)]
            else:
                person_name = "person#{}".format(len(self.known_names))
                self.known_names.append(person_name)
                self.known_encodings.append(encoding)

            found_persons.append((person_name, person_face_bb))

        return found_persons

    def detect_gaze(self, frame):
        # `GazeTracking` lib is detecting gaze from first found face
        # and we can't guarantee that if there are more than 2 face
        # which one will be detected, probably there is need
        # to develop our own solution based on lib?
        self.gaze_detector.refresh(frame)
        left_eye = self.gaze_detector.pupil_left_coords() or (-1, -1)
        right_eye = self.gaze_detector.pupil_right_coords() or (-1, -1)
        hratio = self.gaze_detector.horizontal_ratio() or -1
        vratio = self.gaze_detector.vertical_ratio() or -1

        return left_eye, right_eye, hratio, vratio

    def detect(self, frame):
        work_frame = frame.copy()
        anomaly_markup = {}
        check_gaze = False
        search_name = "student"

        detected_persons = self.detect_persons(work_frame)

        student_data = [p for p in detected_persons if p[0] == search_name]
        student_detected = len(student_data) == 1
        anomaly_markup["student_not_detected"] = not student_detected
        anomaly_markup["unknown_persons"] = (
            len(detected_persons) > 1 and student_detected
        ) or (len(detected_persons) > 0 and not student_detected)

        left_eye, right_eye, hratio, vratio = (-1, -1), (-1, -1), -1, 1
        if student_detected and not anomaly_markup["unknown_persons"]:
            # TODO: right now we dont detect gazes when there are unknown persons, and
            #  this affects 'student_not_looking_on_monitor' anomaly stats,
            #  but we can track gazes even in this situation: crop frame by face coords
            #  for every face we have and pass to detect_gaze(), therefore even when there are many persons,
            #  we can still detect 'student_not_looking_on_monitor' anomaly
            _, student_face = student_data[0]
            x, y, w, h = student_face
            left_eye, right_eye, hratio, vratio = self.detect_gaze(
                work_frame[y : y + h, x : x + w]
            )

        is_looking = (
            self.gaze_hlimits[0] < hratio < self.gaze_hlimits[1]
            and self.gaze_vlimits[0] < vratio < self.gaze_vlimits[1]
        )

        anomaly_markup["student_not_looking_on_monitor"] = not is_looking

        raw_markup = {
            "faces": detected_persons,
            "leye_x": int(left_eye[0]),
            "leye_y": int(left_eye[1]),
            "reye_x": int(right_eye[0]),
            "reye_y": int(right_eye[1]),
            "hratio": float(hratio),
            "vratio": float(vratio),
            "is_looking": is_looking,
        }

        return raw_markup, anomaly_markup

    def __call__(self, frame):
        return self.detect(frame)


def detect_webcam_anomalies(source, config):
    person_detector = PersonDetector(config)

    skip_frames = config["skip_frames"]
    prev_timestamp = 0

    for frame, frame_timestamp, frame_number in video_reader(
        source, sample_rate=skip_frames
    ):
        try:
            _, anomaly_markup = person_detector(frame)
        except Exception as e:
            raise Exception(
                f"Frame #{frame_number}, time: {frame_timestamp} failed, cause: {str(e)}"
            )

        yield {
            "start_time": prev_timestamp / 1000,
            "end_time": frame_timestamp / 1000,
            "anomalies": anomaly_markup,
        }

        prev_timestamp = frame_timestamp


if __name__ == "__main__":
    import time
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-s", "--source", type=str, default="video", help="type of source: video|image"
    )
    parser.add_argument(
        "-p", "--path", type=str, required=True, help="path to input source"
    )
    parser.add_argument(
        "-i", "--image", type=str, required=True, help="path to person image"
    )
    args = vars(parser.parse_args())

    source_path = args["path"]
    person_image = face_recognition.load_image_file(args["image"])

    if args["source"] == "video":
        # if cam -> cast to int
        try:
            source_path = int(source_path)
        except:
            pass

        video = cv2.VideoCapture(source_path)
        print('Video loaded')

        processing_config = {
            "person_image": person_image,
            "skip_frames": 4,
            "frames_num": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "gaze_hlimits": (0.2, 0.8),
            "gaze_vlimits": (0.2, 0.8),
        }

        print('Processing started')
        t = time.time()
        for result in detect_webcam_anomalies(video, processing_config):
            print('Frame result: ',result)
        
        video.release()
        duration = time.time() - t
        print(f"Processing of {args['path']} ended, duration - {duration}")

    elif args['source'] == "image":
        source_img = face_recognition.load_image_file(source_path)
        print('Image loaded')

        h, w = source_img.shape[:2]

        processing_config = {
            "person_image": person_image,
            "width": w,
            "height": h,
            "gaze_hlimits": (0.2, 0.8),
            "gaze_vlimits": (0.2, 0.8),
        }

        person_detector = PersonDetector(processing_config)

        t = time.time()
        print('Processing started')
        _, result = person_detector(source_img)
        duration = time.time() - t
        print(f"Processing of {args['path']} ended, duration - {duration}")
        print('Result: ', result)
    else:
        print(f'Wrong source type, got {args["source"]}')

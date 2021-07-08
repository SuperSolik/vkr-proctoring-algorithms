import functools
import json
from collections import defaultdict
from multiprocessing import cpu_count

import ahocorasick
import cv2
import numpy as np
import pytesseract

from utils import ThreadPool, crop_image, map_by_workers, video_reader


def simple_rect_detector(img, x_k=2, y_k=2):
    """Generates overlapping rectangles to cover whole image
    x_k - how many overlapping rectangles should fit in img width
    y_k - how many overlapping rectangles should fit in img height
    returns a generator with rectangles
    """
    x_k = x_k if x_k > 1 else 2
    y_k = y_k if y_k > 1 else 2

    h, w = img.shape[:2]
    x_d = w / (x_k + 1)
    y_d = h / (y_k + 1)
    return (
        (int(x), int(y), int(x + 2 * x_d), int(y + 2 * y_d))
        for y in np.linspace(0, h - h / (y_k - 1), y_k)
        for x in np.linspace(0, w - w / (x_k - 1), x_k)
    )


def aho_create_automaton(keywords_dct):
    """Inserts all keys from dict into a Aho Corasik trie and creates state machine"""
    aho_automaton = ahocorasick.Automaton()

    process_app_keyword = defaultdict(
        lambda: lambda a, b: (a, b)
    )  # will return app name for all apps except browser
    process_app_keyword["browser"] = lambda a, b: (
        a,
        a,
    )  # will return keyword(link) for browser

    for app_name, app_data in keywords_dct.items():
        for key_word in app_data["keywords"]:
            aho_automaton.add_word(
                key_word, process_app_keyword[app_name](key_word, app_name)
            )

    aho_automaton.make_automaton()
    return aho_automaton


def detect_anomalies_in_text(text, aho_automaton):
    """Matches text to key words from dict"""
    found_apps = set()

    for end_entry_idx, (found_word, value) in aho_automaton.iter(text.lower()):
        found_apps.add(value)

    return list(found_apps)


def recognize_text_in_image(image, try_inverted=False):
    """Recognizes text on given image"""
    # -- psm 7 - assume text on image as signle line, just as EAST provides
    # from https://medium.com/@jaafarbenabderrazak.info/ocr-with-tesseract-opencv-and-python-d2c4ec097866
    # when testing on apps rects, change to 6 or 11 or leave without (3 by default)
    config = "-l eng --oem 1 --psm 11"

    detected_text = pytesseract.image_to_string(image, config=config)

    if try_inverted:
        inverted_text = pytesseract.image_to_string(255 - image, config=config)
        detected_text += inverted_text

    return detected_text


def detect_screenshot_anomalies(img, keywords_dct, rect_detector, parallel=True):
    """Detects anomalies in text on screenshot"""
    rects = rect_detector(img)
    text_images = tuple(crop_image(img, r) for r in rects)

    if parallel:
        texts = map_by_workers(
            functools.partial(recognize_text_in_image, try_inverted=True), text_images
        )
    else:
        texts = tuple(
            recognize_text_in_image(area, try_inverted=True) for area in text_images
        )
    screenshot_text = "".join(set(texts))

    warn_dct = keywords_dct["warn"]
    ok_dct = keywords_dct["ok"]

    warn_apps = detect_anomalies_in_text(
        screenshot_text, aho_create_automaton(warn_dct)
    )
    ok_apps = detect_anomalies_in_text(screenshot_text, aho_create_automaton(ok_dct))

    return {"warn": warn_apps, "ok": ok_apps}


def generate_anomalies_markup(
    frame, frame_timestamp, frame_number, prev_frame_timestamp, keywords_dct
):
    """Creates dictionary containing frame anomalies, defined in keywords_dct"""
    try:
        frame_markup = detect_screenshot_anomalies(
            frame, keywords_dct, simple_rect_detector, parallel=True
        )
        timestamps = {
            "start_time": prev_frame_timestamp / 1000,
            "end_time": frame_timestamp / 1000,
        }
    except Exception as e:
        raise Exception(
            f"Frame #{frame_number}, time: {frame_timestamp} failed, cause: {str(e)}"
        )
    else:
        return {**timestamps, **frame_markup}


def detect_screencast_anomalies(video, keywords_dct, parallel=False):
    """Generates anomalies markup for screencast video"""
    if not isinstance(video, cv2.VideoCapture):  # assuming video is a string
        video = cv2.VideoCapture(video)

    fps = video.get(cv2.CAP_PROP_FPS)
    markup = []

    with ThreadPool(cpu_count()) as pool:
        prev_timestamp = 0
        wait_results = []
        for frame, timestamp, frame_number in video_reader(video, sample_rate=fps // 2):
            if parallel:
                wait_results.append(
                    pool.apply_async(
                        generate_anomalies_markup,
                        (
                            frame.copy(),
                            timestamp,
                            frame_number,
                            prev_timestamp,
                            keywords_dct,
                        ),
                    )
                )
            else:
                markup.append(
                    pool.apply(
                        generate_anomalies_markup,
                        (
                            frame.copy(),
                            timestamp,
                            frame_number,
                            prev_timestamp,
                            keywords_dct,
                        ),
                    )
                )
            prev_timestamp = timestamp

        if parallel:
            markup = tuple(
                r.get() for r in wait_results
            )  # wait for results if parallel

    return markup


if __name__ == "__main__":
    """usage of 'detect_screenshot_anomalies' example"""
    import time
    from argparse import ArgumentParser
    from pprint import pprint

    parser = ArgumentParser()
    parser.add_argument(
        "-s", "--source", type=str, default="video", help="type of source: video|image"
    )
    parser.add_argument("-p", "--path", type=str, help="path to input source")
    parser.add_argument(
        "-t",
        "--parallel",
        action="store_true",
        help="amount of padding to add to each border of ROI",
    )

    args = vars(parser.parse_args())

    f = open("./dictionary.json")
    key_words = json.load(f)
    f.close()

    print("Processing started")
    t = time.time()
    if args["source"] == "video":
        anomalies_markup = detect_screencast_anomalies(
            args["path"], key_words, args["parallel"]
        )
    elif args["source"] == "image":
        anomalies_markup = detect_screenshot_anomalies(
            cv2.imread(args["path"]), key_words, simple_rect_detector, args["parallel"]
        )
    else:
        print(f'Wrong source type, got {args["source"]}')

    duration = time.time() - t
    print(f"Processing of {args['path']} ended, duration - {duration}")

    pprint(anomalies_markup)

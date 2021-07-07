import os
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool

import cv2

os.environ["OMP_THREAD_LIMIT"] = "1"


class cache_result:
    """class for storing function result to avoid
    multiple heavy calls with same args"""

    __slots__ = ["function", "result", "args", "kwargs"]

    def __init__(self, function):
        self.function = function
        self.result = None
        self.args = []
        self.kwargs = {}

    def __call__(self, *args, **kwargs):
        if self.result is None or self.args != args or self.kwargs != kwargs:
            self.args = args
            self.kwargs = kwargs
            self.result = self.function(*self.args, **self.kwargs)

        return self.result


def map_by_workers(handler, tasks):
    """maps tasks to handler using worker processes"""
    with ThreadPool(cpu_count()) as workers_pool:
        results = workers_pool.map(handler, tasks)
    return results


def video_reader(video, sample_rate=0):
    """video iterator, yielding frame and frame end timestamp in ms"""
    if isinstance(video, str):
        source = cv2.VideoCapture(video)
    elif isinstance(video, cv2.VideoCapture):
        source = video
    else:
        raise ValueError(
            f'"video" argument should be str-path (link) to video or cv2.VideoCapture, got {type(video)}'
        )

    success = source.grab()
    frame_number = 0
    while success:
        if frame_number % (sample_rate + 1) == 0:
            _, frame = source.retrieve()
            yield frame, source.get(cv2.CAP_PROP_POS_MSEC), frame_number

        frame_number += 1
        success = source.grab()


def crop_image(image, rect, padding=0.0):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = rect
    dx = int((x2 - x1) * padding)
    dy = int((y2 - y1) * padding)
    if x1 - dx > w or x2 + dx > w:
        x1 = w - (x2 - x1 - 2 * dx)
        x2 = w
    elif x1 - dx < 0 or x2 + dx < 0:
        x1 = 0
        x2 = abs(x2 - x1) + 2 * dx
    if y1 - dy > h or y2 + dy > h:
        y1 = h - (y2 - y1 - 2 * dy)
        y2 = h
    elif y1 - dy < 0 or y2 + dy < 0:
        y1 = 0
        y2 = abs(y2 - y1) + 2 * dy
    return image[y1:y2, x1:x2]

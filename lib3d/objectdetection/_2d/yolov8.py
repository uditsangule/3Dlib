# Author :udit
# Created on : 30/03/24
# Features :
import numpy as np
import os
from ultralytics import YOLO
from utility._opencv import uniquecols

class yolov8():
    def __init__(self , modelsize='m' , acceptclasses=None):
        modelpath = os.path.join('..','..','..','data','models',f'yolov8{modelsize}-seg.pt')
        self.model = YOLO(modelpath)
        self.accpted = np.array([id for id , name in self.model.names.items() if name in acceptclasses]) if acceptclasses is not None else None
        self.class_colors = uniquecols(n_cols=len(self.model.names))


    def detect(self , frame , min_confidence = .2):
        tracker = self.model.track(source=frame ,show=True, persist=True , verbose=False , retina_masks=True,
                                 tracker='botsort.yaml',conf=min_confidence)
        if tracker[0].boxes.shape[0] < 1:
            return
        k=1
        return
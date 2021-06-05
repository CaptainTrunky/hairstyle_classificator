import logging
from pathlib import Path

import cv2
import csv
import copy
import numpy as np
import torch as T
import tqdm

from dataset import build_augmentations


logging.basicConfig(level=logging.INFO)


class Evaluator:
    def __init__(self):
        self.face_detector = None

    def _get_classifier(self, weights):
        from model import get_model

        classifier = get_model()

        classifier.load_state_dict(T.load(weights))

        quantized = T.quantization.quantize_dynamic(
            classifier.to("cpu"), {T.nn.Linear, T.nn.Conv2d}, dtype=T.qint8
        )

        #quantized.to("cuda")

        quantized.eval()

        return quantized

    def _get_face_detector(self):
        from facenet_pytorch import MTCNN

        self.face_detector = MTCNN(keep_all=True, device=T.device("cuda:0"))
    
    def _detect_faces(self, image):
        boxes, _ = self.face_detector.detect(np.expand_dims(image, 0))

        for idx, box in enumerate(boxes[0]):
            h = abs(box[0] - box[2])
            w = abs(box[1] - box[3])

            dw = 0.5 * w
            dh = 0.5 * h

            new_box = copy.deepcopy(box)

            new_box[0] -= dw
            new_box[0] = 0 if new_box[0] < 0 else new_box[0]

            new_box[2] += dw
            new_box[2] = image.shape[1] if new_box[2] > image.shape[1] else new_box[2]

            new_box[1] -= dh
            new_box[1] = 0 if new_box[1] < 0 else new_box[1]

            new_box[3] += dh
            new_box[3] = image.shape[0] if new_box[3] > image.shape[0] else new_box[3]

            boxes[0][idx] = new_box

        return boxes[0].astype(np.uint32)

    def run(self, dataset_root, weights):
        classifier = self._get_classifier(weights)
        self._get_face_detector()

        if not isinstance(dataset_root, Path):
            dataset_root = Path(dataset_root)

        paths = sorted(dataset_root.glob("*.jpg"))

        preproccesing = build_augmentations()["eval"]

        with open("./results.csv", "w", newline="") as out:
            writer = csv.writer(out, delimiter=",")
            
            times_per_frame = []

            for idx, path in tqdm.tqdm(enumerate(paths)):
                img = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                boxes = self._detect_faces(img)

                if boxes.size == 0:
                    writer.writerow([path.as_posix(), "-1"])
                    continue

                # since we don't care about processing all faces, select random box
                box = boxes[0]

                crop = img[box[1]:box[3], box[0]:box[2], :]

                if False:
                    cv2.imwrite(
                        f"{path.name}", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                        [cv2.IMREAD_UNCHANGED]
                    )

                crop = preproccesing(crop)

                crop = T.unsqueeze(crop, 0)
                
                #crop = T.Tensor(crop).to("cuda")

                start = T.cuda.Event(enable_timing=True)
                end = T.cuda.Event(enable_timing=True)

                start.record()

                predict = classifier(crop).squeeze(-1)
               
                label = T.round(T.sigmoid(predict))

                end.record()

                T.cuda.synchronize()

                times_per_frame.append(start.elapsed_time(end))

                writer.writerow([path.as_posix(), 1 - label.item()])

        logging.info(f"DONE, time per call: {np.mean(times_per_frame)}")


if __name__ == "__main__":
    import fire

    fire.Fire(Evaluator)



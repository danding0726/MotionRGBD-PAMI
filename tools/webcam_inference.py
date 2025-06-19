import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from config import Config
from lib.model.build import build_model
from lib.datasets.base import Normaliztion
from utils.utils import load_pretrained_checkpoint


def preprocess_frame(frame, args):
    resize = eval(args.resize)
    crop_size = args.crop_size
    sample_size = args.sample_size

    left = (resize[0] - crop_size) // 2
    top = (resize[1] - crop_size) // 2

    transform = transforms.Compose([
        Normaliztion(),
        transforms.ToTensor()
    ])

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.resize(resize)
    img = img.crop((left, top, left + crop_size, top + crop_size))
    img = img.resize((sample_size, sample_size))
    img = np.array(img)
    return transform(img).view(3, sample_size, sample_size, 1)


def main():
    parser = argparse.ArgumentParser(description='Webcam inference demo')
    parser.add_argument('--config', required=True, help='path to yaml config')
    parser.add_argument('--checkpoint', required=True, help='model checkpoint')
    parser.add_argument('--sample-duration', type=int, default=32, help='frames per clip')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--display', action='store_true', help='show webcam')
    args = parser.parse_args()
    args = Config(args)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = build_model(args)
    load_pretrained_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame, args))
        if len(frames) == args.sample_duration:
            clip = torch.cat(frames, dim=3).permute(0, 3, 1, 2).unsqueeze(0)
            clip = clip.to(device)
            with torch.no_grad():
                (logits, _, _, _), _ = model(clip)
                pred = logits.argmax(dim=1).item()
            print('Prediction:', pred)
            frames = []
        if args.display:
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

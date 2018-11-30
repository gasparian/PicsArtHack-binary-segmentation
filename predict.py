import sys
from train import *

trainer = Trainer(path="./data/resnet101_05BCE_no_CLR_50e_ADAM_no_weight")
trainer.load_state(mode="metric")


while True:
    file_path = sys.stdin.readline()[:-1]
    if not os.path.isfile(file_path):
        print(file_path)
        print("file not found")
        sys.exit(-1)

    imgs = cv2.imread(file_path)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    imgs = np.array(imgs, dtype=np.uint8)
    out = trainer.predict_crop(imgs)

    out.save("/output.png")

    print("done")
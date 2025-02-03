import json
import kagglehub

dataset_path = "/home/bulat/.cache/kagglehub/datasets/timoboz/clevr-dataset/versions/2"

def get_scenes():
    with open(dataset_path + '/CLEVR_v1.0/scenes/CLEVR_train_scenes.json', 'r') as f:
        scenes = json.load(f)['scenes']
    return scenes


if __name__ == '__main__':
    # Download latest version
    path = kagglehub.dataset_download("timoboz/clevr-dataset")
    print("Path to dataset files:", path)

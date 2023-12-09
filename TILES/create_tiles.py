from pathlib import Path


def load_annotation(path: str) -> list:
    with open(path, 'r') as f:
        # [Class, cx, xy, w, h] 
        ann = [list(map(float, line.replace('\n', '').split())) for line in f]

    return ann


if __name__ == '__main__':
    name = 'himanshu'

    data_path = str(Path.cwd()) + r'\TILES\InputData'
    image_path = data_path + rf'\{name}.jpeg'
    annotation_path = data_path + rf'\{name}\obj_train_data\{name}.txt'

    annotation_list = load_annotation(annotation_path)
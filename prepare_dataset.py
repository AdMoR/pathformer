import os
import pickle
from pathformer.datasets.path_dataset_v2 import parse_document


if __name__ == "__main__":
    dir_ = "/media/amor/data/svg_data"
    files = os.listdir(dir_)
    memory = "./memory.txt"
    if os.path.exists(memory):
        with open(memory, "r") as f:
            last_i = int(f.read())
    else:
        last_i = 0
    for i, f in enumerate(files):
        print(i, f)
        if i < last_i:
            continue
        try:
            rez = parse_document(os.path.join(dir_, f))
            last_i = i
            with open(memory, "w") as f:
                f.write(str(i))
        except ValueError as e:
            print(f"{f} failed with : {e}")
        #pickle.dump(rez, open(f, "wb"))


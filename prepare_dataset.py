import os
import pickle
from pathformer.datasets.path_dataset_v2 import parse_document


if __name__ == "__main__":
    dir_ = "/home/amor/Downloads/svg_data/svg_processed"
    out_dir = "/media/amor/data/svg_parsed_processed"
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
            outfile = os.path.join(out_dir, f"{f.split(".")[0]}.pkl")
            pickle.dump(rez, open(outfile, "wb"))
            print("=>> ", len(rez))
            last_i = i
            with open(memory, "w") as f:
                f.write(f"{f},{i}")
        except (ValueError, KeyError, TypeError, AssertionError) as e:
            print(f"{f} failed with : {e}")



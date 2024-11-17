import os
import sys
import subprocess
import shutil


def main_processing(input_svg, outdir):
    """
    Original code in shell.

    for f in /media/amor/data/svg_data/svg/*.svg; do
        echo $f
        export r=$(echo $f | sed -r "s/.+\/(.+)\..+/\1/")
        inkscape --actions="select-all;object-to-path;export-type:svg;export-filename:./$r.svg;export-do;" $f
        inkscape ./$r.svg --export-plain-svg --export-type=svg --export-filename=$r_out
        python3 /usr/share/inkscape/extensions/ungroup_deep.py ./`echo $r`_out.svg > ./`echo $r`_test.svg
        python3 /usr/share/inkscape/extensions/applytransform.py ./`echo $r`_test.svg > ./`echo $r`_finish.svg
        rm  ./`echo $r`_out.svg  ./`echo $r`_test.svg ./`echo $r`.svg
    done

    The issue is that the behaviour of inkscape extension depends on the content
    of the svg file (presence of <g> or transform)
    """
    # Step 1
    filename = os.path.basename(input_svg).split(".")[0]
    outfile1 = f"{os.path.join(outdir, filename)}_copy.svg"
    cmd = f'inkscape --actions="select-all;object-to-path;export-type:svg;export-filename:{outfile1};export-do;" {input_svg}'
    output, error = subprocess.Popen(cmd, shell=True, text=True).communicate()
    if error:
        raise Exception("Failed on copy 1")

    #Step 2
    outfile2 = f"{os.path.join(outdir, filename)}_out.svg"
    rez = subprocess.run(["inkscape", outfile1,
                          "--export-plain-svg", "--export-type=svg", f"--export-filename={outfile2}"])
    if rez.returncode == 1:
        raise Exception("Failed on simplify 2")

    # Step 3
    outfile3 = f"{os.path.join(outdir, filename)}_ungrouped.svg"
    f = open(outfile3, "w")
    rez = subprocess.call(["python3", "/usr/share/inkscape/extensions/applytransform.py", outfile2],
                          stdout=f)
    f.close()
    if rez == 1:
        raise Exception("Failed on ungroup 3")
    with open(outfile3) as f:
        if len(f.readlines()) == 0:
            outfile3 = outfile2

    # Step 4
    outfile4 = f"{os.path.join(outdir, filename)}_transformed.svg"
    f = open(outfile4, "w")
    rez = subprocess.call(["python3", "/usr/share/inkscape/extensions/ungroup_deep.py", outfile3], stdout=f)
    f.close()
    if rez == 1:
        raise Exception("Failed on ungroup 3")
    with open(outfile4) as f:
        if len(f.readlines()) == 0:
            outfile4 = outfile3

    # Step 5 - Clean up
    final_file = f"{os.path.join(outdir, filename)}_final.svg"
    shutil.copyfile(outfile4, final_file)
    for ext in ["copy", "out", "transformed", "ungrouped"]:
        to_del = f"{os.path.join(outdir, filename)}_{ext}.svg"
        if os.path.exists(to_del):
            os.remove(to_del)

    return final_file


def register_in_file(str_, file_name):
    with open(file_name, "a") as g:
        g.write(str_)


def load_register_file(fname, index):
    """
    Expectation ;-separated file
    """
    if os.path.exists(fname):
        index.update(set(l.split(";")[0] for l in open(fname).readlines()))


if __name__ == "__main__":
    source_folder_path = sys.argv[1]
    output_folder_path = sys.argv[2]

    index_file = "./processing_index.txt"
    error_file = "./processing_errors.txt"

    done = set()
    load_register_file(index_file, done)
    load_register_file(error_file, done)

    for f in os.listdir(source_folder_path):
        input_svg = os.path.join(source_folder_path, f)
        print(">> ", input_svg)
        if input_svg in done:
            continue
        if os.stat(input_svg).st_size == 0:
            register_in_file(f"{input_svg};EmptyFile\n", error_file)
            continue
        try:
            result = main_processing(input_svg, output_folder_path)
            register_in_file(f"{input_svg};{result}\n", index_file)
        except Exception as e:
            print("Error : ", input_svg, " ", str(e))
            register_in_file(f"{input_svg};{e}\n", error_file)

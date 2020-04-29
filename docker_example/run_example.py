import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


if __name__ == "__main__":

    import scripts.evalresults as evalresults

    print("Starting MOOD example...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        type=str,
        help="Input dir requires a subfolder 'toy' and 'toy_label' i.e. input_dir/toy, input_dir/toy_label",
    )
    parser.add_argument(
        "--no_gpu",
        required=False,
        default=False,
        type=bool,
        help="If you have not installed the nvidia docker toolkit, set this arg to False",
    )

    args = parser.parse_args()

    data_dir = args.input_dir
    no_gpu = args.no_gpu

    tmp_dir = tempfile.TemporaryDirectory()
    output_dir = tmp_dir.name

    toy_input_dir = os.path.join(data_dir, "toy")
    toy_label_sample_dir = os.path.join(data_dir, "toy_label", "sample")
    toy_label_pixel_dir = os.path.join(data_dir, "toy_label", "pixel")

    output_sample_dir = os.path.join(data_dir, "pixel")
    output_pixel_dir = os.path.join(data_dir, "sample")

    example_dir = pathlib.Path(__file__).parent.absolute()

    print("Building docker...")

    try:
        # "docker build ${DOCKER_FILE_PATH} -t mood_name"
        ret = subprocess.run(["docker", "build", example_dir, "-t", "mood_example"], check=True)
    except Exception:
        print("Building Docker failed:")
        print(ret)
        exit(1)

    print("Docker build.")
    print("\nPredicting pixel-level anomalies.")

    gpu_str = ""
    if not no_gpu:
        gpu_str = "--gpus all "

    try:
        docker_str = (
            f"sudo docker run {gpu_str}-v {toy_input_dir}:/mnt/data "
            f"-v {output_dir}:/mnt/pred mood_example sh /workspace/run_pixel_brain.sh /mnt/data /mnt/pred"
        )
        ret = subprocess.run(docker_str.split(" "), check=True,)
    except Exception:
        print("Running Docker pixel-script failed:")
        print(ret)
        exit(1)

    print("\nPredicting sample-level anomalies.")

    try:
        docker_str = (
            f"sudo docker run {gpu_str}-v {toy_input_dir}:/mnt/data "
            f"-v {output_dir}:/mnt/pred mood_example sh /workspace/run_sample_brain.sh /mnt/data /mnt/pred"
        )
        ret = subprocess.run(docker_str.split(" "), check=True,)
    except Exception:
        print("Running Docker sample-script failed:")
        print(ret)
        exit(1)

    print("\nEvaluating predictions...")

    res_pixel = evalresults.eval_dir(output_dir, toy_label_pixel_dir, mode="pixel",)
    print("Pixel-level score:", res_pixel)

    res_sample = evalresults.eval_dir(output_dir, toy_label_sample_dir, mode="sample",)
    print("Sample-level scores:", res_sample)

    tmp_dir.cleanup()

    print("Done.")

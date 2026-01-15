import yaml
import itertools
import copy
import sys
import csv
import traceback
import subprocess
import time
from extract_prototype2 import extract_metrics
from typing import Dict, Any, List
from pathlib import Path
from learn import automate_train
from hw_stats import get_pc_stats
from realTimeHardwareLogger import get_RAM_numbers

###NOTE: STILL NEEDS DEBUGGING AND FINISHING UP, PUSHED JUST TO GIVE AN IDEA + FOR ADDITIONAL WORK

#FUNCTION: opens a yaml file with hyperparameter (hp) ranges, loads as dict (settings)
def yaml_to_dict(file_path: str) -> Dict[str,Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            settings = yaml.safe_load(file)
        print("Loaded yaml to settings dict.")
        return settings
    except Exception:
        print("Couldn't load yaml to settings.")
        traceback.print_exc()
        raise



def csv_to_dict_list(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = []

            for row in reader:
                # Convert numeric strings to int/float when possible
                parsed_row = {}
                for key, value in row.items():
                    if value is None:
                        parsed_row[key] = None
                    else:
                        value = value.strip()
                        if value == "":
                            parsed_row[key] = None
                        else:
                            # Try int → float → fallback to string
                            try:
                                parsed_row[key] = int(value)
                            except ValueError:
                                try:
                                    parsed_row[key] = float(value)
                                except ValueError:
                                    parsed_row[key] = value

                rows.append(parsed_row)

        print("Loaded CSV to list-of-dicts.")
        return rows

    except Exception:
        print("Couldn't load CSV to dict.")
        traceback.print_exc()
        raise


def find_step_at_or_closest_to_benchmark(
    rows,
    benchmark_reward,
    mean_key="mean_reward",
    step_key="step",
):
    """
    Returns:
      (step, mean_reward, reached_threshold: bool)
    """

    best_row = None
    best_diff = float("inf")

    for row in rows:
        mean = row.get(mean_key)
        step = row.get(step_key)

        if mean is None or step is None:
            continue

        # mean threshold reached
        if mean >= benchmark_reward:
            return step, mean, True

        # track below threshold
        diff = benchmark_reward - mean
        if diff >= 0 and diff < best_diff:
            best_diff = diff
            best_row = row

    if best_row is not None:
        return best_row[step_key], best_row[mean_key], False

    # No usable metrics at all
    return -1, None, False



#FUNCTION: extracts the hyperparameter subdict (lists of hyperparams) from settings
def extract_hp(settings: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return next(iter(settings["behaviors"].values()))["hyperparameters"]
    except Exception:
        print("Couldn't extract hyperparameters from settings.")
        print("Full traceback:")
        traceback.print_exc()
        raise

#FUNCTION: gets the keys (attributes - hp names) from hyperparameter dictionary ~ used as columns in csv
def get_ordered_keys(multi_yaml_path: Path) -> List[str]:
    settings = yaml_to_dict(str(multi_yaml_path))
    hp = extract_hp(settings)
    return list(hp.keys())


#FUNCTION: generates all possible hp combos -> dictionary of lists (one list - one hp combo)
def generate_hp_combos(hp: Dict[str, List[Any]]) -> Dict[int, List[Any]]:
    try:
        print("Generating hp combinations...")
        value_lists = []
        #extract the lists of each hyperparameter values (e.g. list for alpha, beta, etc)
        for key in hp:
            value_lists.append(hp[key])

        #compute cartesian product of all the values in hp lists
        value_combos = itertools.product(*value_lists)

        #fill a dict of all hp_combos -> one list = one testable yaml file
        hp_combos = {}
        i = 1
        for c in value_combos:
            hp_combos[i] = list(c)
            i += 1
        print("Printing the resulting combos:")
        print(hp_combos)
        return hp_combos
    except Exception as e:
        print("Couldn't generate combinations.")
        print("Exception type:", type(e).__name__)
        print("Exception message:", e)
        print("Full traceback:")
        traceback.print_exc()
        raise

#FUNCTION: init_csv() -> initializes an empty csv file with columns
def init_csv(csv_doc: Path, hp_keys: List[str]):

    columns = [
        "ID",
        "cpu_type",
        "cpu_cores",
        "ram_gb",
        "has_nvidia",
        "nvidia_gpu_name",
        "os",
        *hp_keys,
        "ram_mb_used",
        "ram_usage_percent",
        "ram_avg_percent",
        "benchmark_steps",
        "benchmark_mean",
        "benchmark_reached"
    ]

    with open(csv_doc, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)

#FUNCTION: append a row representing data from single run of training
def append_data(
        writer: csv.writer,
        run_id: str,
        hw_info: Dict[str,Any],
        hp_values: List[Any],
        ram_mb_used: Any = 0,
        ram_usage_percent: Any = 0,
        ram_avg_percent: Any = 0,
        benchmark_step: Any = 0,
        benchmark_mean: Any = 0,
        benchmark_reached: bool = False,

) -> None:
    row = [
        run_id,
        hw_info["cpu_type"],
        hw_info["cpu_cores"],
        hw_info["ram_gb"],
        hw_info["has_nvidia"],
        hw_info["nvidia_gpu_name"],
        hw_info["os_name"],
        *hp_values,
        ram_mb_used,
        ram_usage_percent,
        ram_avg_percent,
        benchmark_step,
        benchmark_mean,
        benchmark_reached,
    ]
    writer.writerow(row)

#FUNCTION: runs all trainings
def run_trainings(
        folder_path: Path,
        run_id: str,
        flags: List[str],
        hp_keys: List[str],
        csv_path: Path,
        build_exe: str,
        benchmark_reward: float

):

    hw_info = get_pc_stats()
    print("Get hw specs for run")

    #Prep yaml files + get counts
    yaml_files = sorted(folder_path.glob(f"{run_id}.*.yaml"))
    num_files = len(yaml_files)
    finished_files = 0

    if(num_files == 0):
        print("No yaml files to train.")
        return

    with csv_path.open("a", newline="", encoding="utf-8") as results:
        writer = csv.writer(results)

        for file in yaml_files:
            if not file.is_file():
                continue

            #Create new run id for indiviual yaml
            path_to_file = file.resolve()
            print("Path to file: ", path_to_file)
            index = file.stem.split(".")[-1]
            print("Index: ", index)
            new_run_id = f"{run_id}.{index}"
            print("Run id: ", new_run_id)

            print("="*90)
            print("AUTOMATE STEP: Initiating training of ", path_to_file, " yaml", str(index))
            print("="*90)


            #Train using learn.py
            train_args = [str(path_to_file), "--run-id", new_run_id, "--env", build_exe, *flags]
            print(train_args)
            #Start the ram usage log

            ram_csv_name = (
                Path(r"/Users/Sebastian/PycharmProjects/ml-agents/ml-agents/mlagents/trainers/hw")
                / f"{new_run_id}.hardware_log.csv"
            )

            logger_script = Path(__file__).resolve().parent / "realTimeHardwareLogger.py"

            ram_log = subprocess.Popen(
                [sys.executable, str(logger_script), str(ram_csv_name)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text = True
            )


            automate_train(train_args)
            ram_log.terminate()

            max_ram_mb, ram_usage_percent, ram_avg_percent = get_RAM_numbers(ram_csv_name)

            metrics_csv = extract_metrics(
                run_id=new_run_id,
                output_csv="prototype.csv"
            )

            if metrics_csv is None:
                print(f"[WARN] No metrics for run {new_run_id}, skipping benchmark step")
                step_at_benchmark = -1
                mean_at_step = None
                reached = False
            else:
                metrics_rows = csv_to_dict_list(str(metrics_csv))
                step_at_benchmark, mean_at_step, reached = find_step_at_or_closest_to_benchmark(
                    metrics_rows,
                    benchmark_reward=benchmark_reward
                )

                if reached:
                    print(
                        f"[OK] {new_run_id} reached benchmark "
                        f"({mean_at_step} ≥ {benchmark_reward}) at step {step_at_benchmark}"
                    )
                else:
                    print(
                        f"[WARN] {new_run_id} did NOT reach benchmark. "
                        f"Closest mean = {mean_at_step} at step {step_at_benchmark}"
                    )

            #Get the hp values for this training
            run_config = yaml_to_dict(str(path_to_file))
            run_hp = extract_hp(run_config)

            hp_values = []
            for key in hp_keys:
                hp_values.append(run_hp[key])

            #Write results into csv
            append_data(
                writer,
                new_run_id,
                hw_info,
                hp_values,
                max_ram_mb,
                ram_usage_percent,
                ram_avg_percent,
                benchmark_step=step_at_benchmark,
                benchmark_mean=mean_at_step,
                benchmark_reached=reached,
            )

            finished_files += 1

        print("="*90)
        print("AUTOMATE STEP: Training of ", path_to_file, " finished.")
        print("="*90)




#FUNCTION: parses multi-hp yaml file -> generates x new .yaml files ~ all combos of hp's to be tested & returns path to yaml files folder
def parse_multi_yaml(path: str, run_id: str):
    #extract multi-hp yaml configs
    settings = yaml_to_dict(path)
    #extract hp from settings
    hp = extract_hp(settings)
    #generate all possible hp combos
    hp_combos = generate_hp_combos(hp)
    #create a folder for generated .yaml files -> folder in config folder (separate from poca, ppo etc.)
    print("Creating path...")
    path_to_result = Path(__file__).resolve().parents[2] / "config" / "multi" / run_id
    path_to_result.mkdir(parents=True, exist_ok=True)
    print("The path to folder: ", path_to_result)

    print("Generating .yaml files....")
    n = len(hp_combos)
    try:
        for i in range(1, n + 1):
            #copy the original configuration & create separate hp dict
            temp = copy.deepcopy(settings)
            temp_hp = extract_hp(temp)
            #get the current hp value combo
            values = hp_combos[i]

            #over-write the hp ranges with single values of the combo i
            k = 0
            for parameter in temp_hp:
                temp_hp[parameter] = values[k]
                k += 1

            #dump the new temp configs into yaml file in original yaml syntax
            with open(path_to_result / f"{run_id}.{i}.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(temp, f, sort_keys=False)

        return path_to_result
    except Exception as e:
        print("Couldn't generate combinations.")
        print("Exception type:", type(e).__name__)
        print("Exception message:", e)
        print("Full traceback:")
        traceback.print_exc()
        raise


#FUNCTION: main() -> entry point to the script, parses the multi-hp yaml file + runs learn.py on each combo
def main():
    #TODO: add logger to accommodate pause/resume of training scenario

    if len(sys.argv) < 3:
        print("Provide arguments in the form: <path_to_multi_hp_yaml> <run_id> <benchmark performance> [flags]")
        sys.exit(2)

    try:
        #Get CL args
        yaml_path = sys.argv[1]
        run_id = sys.argv[2]
        bench_reward = sys.argv[3]
        build_exe = sys.argv[4]
        flags = sys.argv[5:]

        #Parse the multi-hp yaml, create combination yamls and get the path to their folder
        folder_path = parse_multi_yaml(yaml_path, run_id)

        #Get ordered hp keys
        hp_keys = get_ordered_keys(Path(yaml_path))

        #Create csv for result data, initialize with columns
        result_path = folder_path / f"{run_id}.csv"
        init_csv(result_path, hp_keys)

        #Train
        print("Beginning automated training...")
        run_trainings(folder_path, run_id, flags, hp_keys, result_path,build_exe,benchmark_reward=float(bench_reward))
        print("Finish all trainings.")

    except Exception as e:
        print("Exception type:", type(e).__name__)
        print("Exception message:", e)
        print("Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()



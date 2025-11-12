"""
MASTER TRAINING PIPELINE - Phase 2
Complete automation: hardware capture + training + monitoring + extraction

This script does EVERYTHING you need for Phase 2 data collection.
"""

import os
import sys
import json
import time
import yaml
import glob
import psutil
import platform
import threading
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator


class MasterTrainingPipeline:
    """Complete training pipeline with all data collection."""

    def __init__(self, config_path, run_id, results_dir="results"):
        self.config_path = config_path
        self.run_id = run_id
        self.results_dir = results_dir
        self.run_dir = Path(results_dir) / run_id

        # Data storage
        self.hardware_info = {}
        self.hyperparameters = {}
        self.resource_samples = {
            'timestamps': [],
            'ram_gb': [],
            'cpu_percent': []
        }
        self.training_start_time = None
        self.training_end_time = None
        self.monitoring = False

    def step1_capture_hardware(self):
        """Step 1: Capture hardware specifications (using your hardwareInfo.py logic)."""
        print(f"\n{'=' * 70}")
        print("STEP 1: Capturing Hardware Information")
        print(f"{'=' * 70}\n")

        try:
            import socket
            import re
            import uuid

            # Using your comprehensive hardware capture logic
            self.hardware_info = {
                'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'ram_total_gb': round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
                'cpu_count': os.cpu_count(),
                'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S"),
                'python_version': platform.python_version(),
                'capture_timestamp': datetime.now().isoformat()
            }

            # Network information (with error handling)
            try:
                self.hardware_info['hostname'] = socket.gethostname()
            except Exception:
                self.hardware_info['hostname'] = 'unknown'

            try:
                self.hardware_info['ip_address'] = socket.gethostbyname(socket.gethostname())
            except Exception:
                # Fallback: try to get IP another way
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    self.hardware_info['ip_address'] = s.getsockname()[0]
                    s.close()
                except Exception:
                    self.hardware_info['ip_address'] = '127.0.0.1'

            try:
                self.hardware_info['mac_address'] = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
            except Exception:
                self.hardware_info['mac_address'] = 'unknown'

            # Add disk information
            disk = psutil.disk_usage('/')
            self.hardware_info['disk_total_gb'] = round(disk.total / (1024 ** 3), 2)
            self.hardware_info['disk_used_gb'] = round(disk.used / (1024 ** 3), 2)
            self.hardware_info['disk_free_gb'] = round(disk.free / (1024 ** 3), 2)
            self.hardware_info['disk_percent'] = round(disk.percent, 1)

            # Initial CPU usage
            self.hardware_info['cpu_usage_at_start'] = round(psutil.cpu_percent(interval=1), 1)

            print("✓ Hardware Information:")
            print(f"  Platform: {self.hardware_info['platform']}")
            print(f"  Hostname: {self.hardware_info['hostname']}")
            print(f"  Processor: {self.hardware_info['processor']}")
            print(f"  CPU Cores: {self.hardware_info['cpu_count']}")
            print(f"  Total RAM: {self.hardware_info['ram_total_gb']} GB")
            print(f"  Disk Free: {self.hardware_info['disk_free_gb']} GB / {self.hardware_info['disk_total_gb']} GB")

            return True

        except Exception as e:
            print(f"❌ Error capturing hardware: {e}")
            return False

    def step2_extract_hyperparameters(self):
        """Step 2: Extract hyperparameters from config file."""
        print(f"\n{'='*70}")
        print("STEP 2: Extracting Hyperparameters")
        print(f"{'='*70}\n")

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            behaviors = config.get('behaviors', {})
            if not behaviors:
                print("❌ No behaviors found in config")
                return False

            behavior_name = list(behaviors.keys())[0]
            behavior = behaviors[behavior_name]

            hyperparams = behavior.get('hyperparameters', {})
            network = behavior.get('network_settings', {})

            self.hyperparameters = {
                'environment': behavior_name,
                'trainer_type': behavior.get('trainer_type', 'ppo'),
                'max_steps': behavior.get('max_steps', 0),
                'time_horizon': behavior.get('time_horizon', 0),
                'summary_freq': behavior.get('summary_freq', 0),

                'learning_rate': hyperparams.get('learning_rate', 0),
                'batch_size': hyperparams.get('batch_size', 0),
                'buffer_size': hyperparams.get('buffer_size', 0),
                'num_epoch': hyperparams.get('num_epoch', 0),
                'beta': hyperparams.get('beta', 0),
                'epsilon': hyperparams.get('epsilon', 0),
                'lambd': hyperparams.get('lambd', 0),

                'hidden_units': network.get('hidden_units', 0),
                'num_layers': network.get('num_layers', 0),
                'normalize': network.get('normalize', False),
            }

            print("✓ Configuration Details:")
            print(f"  Environment: {self.hyperparameters['environment']}")
            print(f"  Algorithm: {self.hyperparameters['trainer_type'].upper()}")
            print(f"  Learning Rate: {self.hyperparameters['learning_rate']}")
            print(f"  Batch Size: {self.hyperparameters['batch_size']}")
            print(f"  Network Size: {self.hyperparameters['hidden_units']} units × {self.hyperparameters['num_layers']} layers")
            print(f"  Max Steps: {self.hyperparameters['max_steps']:,}")

            return True

        except Exception as e:
            print(f"❌ Error extracting hyperparameters: {e}")
            return False

    def _monitor_resources(self):
        """Background thread for monitoring resources."""
        while self.monitoring:
            try:
                ram_gb = psutil.virtual_memory().used / (1024**3)
                cpu_percent = psutil.cpu_percent(interval=1)

                self.resource_samples['timestamps'].append(time.time())
                self.resource_samples['ram_gb'].append(ram_gb)
                self.resource_samples['cpu_percent'].append(cpu_percent)

                time.sleep(9)  # Total 10 seconds (1 s for cpu_percent + 9 s sleep)

            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")

    def step3_run_training(self):
        """Step 3: Run training with resource monitoring."""
        print(f"\n{'='*70}")
        print("STEP 3: Running Training with Monitoring")
        print(f"{'='*70}\n")

        # Create run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Start resource monitoring
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        monitor_thread.start()
        print("✓ Resource monitoring started (sampling every 10s)")

        # Build training command
        cmd = [
            "mlagents-learn",
            self.config_path,
            f"--run-id={self.run_id}",
            "--force"  # ← Add this line
        ]

        print(f"\nCommand: {' '.join(cmd)}")
        print(f"\n⏳ Waiting for Unity connection...")
        print(f"👉 GO TO UNITY AND PRESS PLAY NOW!")
        print(f"\n{'='*70}\n")

        # Record start time
        self.training_start_time = time.time()

        try:
            # Run training
            result = subprocess.run(cmd, check=True)
            training_success = True
            print(f"\n✓ Training completed successfully!")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with error code {e.returncode}")
            training_success = False

        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted by user")
            training_success = False

        finally:
            # Stop monitoring
            self.training_end_time = time.time()
            self.monitoring = False
            monitor_thread.join(timeout=5)
            print("✓ Resource monitoring stopped")

        return training_success

    def step4_extract_tensorboard(self):
        """Step 4: Extract TensorBoard metrics."""
        print(f"\n{'='*70}")
        print("STEP 4: Extracting TensorBoard Metrics")
        print(f"{'='*70}\n")

        try:
            # Find event files
            event_files = glob.glob(
                str(self.run_dir / "**" / "events.out.tfevents.*"),
                recursive=True
            )

            if not event_files:
                print(f"❌ No TensorBoard files found in {self.run_dir}")
                return None

            print(f"✓ Found event file: {event_files[0]}")

            # Load TensorBoard data
            ea = event_accumulator.EventAccumulator(event_files[0])
            ea.Reload()

            available_keys = ea.Tags()["scalars"]
            print(f"✓ Available metrics: {len(available_keys)}")

            # Extract metrics
            metrics_map = {
                "Environment/Cumulative Reward": "cumulative_reward",
                "Environment/Episode Length": "episode_length",
                "Policy/Entropy": "entropy",
                "Losses/Policy Loss": "policy_loss",
                "Losses/Value Loss": "value_loss",
            }

            data = {"step": [], "wall_time": []}

            # Get primary metric
            primary = "Environment/Cumulative Reward"
            if primary in available_keys:
                for event in ea.Scalars(primary):
                    data["step"].append(event.step)
                    data["wall_time"].append(event.wall_time)
                    data[metrics_map[primary]] = data.get(metrics_map[primary], [])
                    data[metrics_map[primary]].append(event.value)

            num_steps = len(data["step"])

            # Extract other metrics
            for tb_key, col_name in metrics_map.items():
                if tb_key == primary or tb_key not in available_keys:
                    continue

                values = [e.value for e in ea.Scalars(tb_key)]

                # Handle length mismatch
                if len(values) < num_steps:
                    values.extend([None] * (num_steps - len(values)))
                elif len(values) > num_steps:
                    values = values[:num_steps]

                data[col_name] = values

            df = pd.DataFrame(data)
            df["time_elapsed"] = df["wall_time"] - df["wall_time"].iloc[0]
            df = df.drop("wall_time", axis=1)

            print(f"✓ Extracted {len(df)} training steps")

            return df

        except Exception as e:
            print(f"❌ Error extracting TensorBoard: {e}")
            return None

    def step5_create_summary(self, df_metrics):
        """Step 5: Create final summary combining all data."""
        print(f"\n{'='*70}")
        print("STEP 5: Creating Final Summary")
        print(f"{'='*70}\n")

        summary = {
            'run_id': self.run_id,
            'collection_timestamp': datetime.now().isoformat(),
        }

        # Add hardware info
        summary.update(self.hardware_info)

        # Add hyperparameters
        summary.update(self.hyperparameters)

        # Add training duration
        if self.training_start_time and self.training_end_time:
            duration_seconds = self.training_end_time - self.training_start_time
            summary['training_duration_seconds'] = round(duration_seconds, 2)
            summary['training_duration_minutes'] = round(duration_seconds / 60, 2)
            summary['training_duration_hours'] = round(duration_seconds / 3600, 2)

        # Add resource usage stats
        if self.resource_samples['ram_gb']:
            summary['peak_ram_gb'] = round(max(self.resource_samples['ram_gb']), 2)
            summary['avg_ram_gb'] = round(sum(self.resource_samples['ram_gb']) / len(self.resource_samples['ram_gb']), 2)
            summary['min_ram_gb'] = round(min(self.resource_samples['ram_gb']), 2)

        if self.resource_samples['cpu_percent']:
            summary['peak_cpu_percent'] = round(max(self.resource_samples['cpu_percent']), 1)
            summary['avg_cpu_percent'] = round(sum(self.resource_samples['cpu_percent']) / len(self.resource_samples['cpu_percent']), 1)

        # Add training performance metrics
        if df_metrics is not None and not df_metrics.empty:
            summary['total_training_steps'] = int(df_metrics['step'].max())
            summary['final_cumulative_reward'] = float(df_metrics['cumulative_reward'].iloc[-1])
            summary['max_cumulative_reward'] = float(df_metrics['cumulative_reward'].max())
            summary['mean_cumulative_reward'] = float(df_metrics['cumulative_reward'].mean())
            summary['final_episode_length'] = float(df_metrics['episode_length'].iloc[-1])
            summary['mean_episode_length'] = float(df_metrics['episode_length'].mean())

            if 'entropy' in df_metrics.columns:
                entropy_values = df_metrics['entropy'].dropna()
                if len(entropy_values) > 0:
                    summary['initial_entropy'] = float(entropy_values.iloc[0])
                    summary['final_entropy'] = float(entropy_values.iloc[-1])

        print("✓ Summary Statistics:")
        print(f"  Duration: {summary.get('training_duration_minutes', 0):.1f} minutes")
        print(f"  Steps: {summary.get('total_training_steps', 0):,}")
        print(f"  Final Reward: {summary.get('final_cumulative_reward', 0):.2f}")
        print(f"  Peak RAM: {summary.get('peak_ram_gb', 0):.2f} GB")
        print(f"  Avg CPU: {summary.get('avg_cpu_percent', 0):.1f}%")

        return summary

    def step6_save_everything(self, df_metrics, summary):
        """Step 6: Save all data to files."""
        print(f"\n{'='*70}")
        print("STEP 6: Saving All Data")
        print(f"{'='*70}\n")

        # Save detailed TensorBoard metrics
        if df_metrics is not None:
            metrics_file = self.run_dir / "tensorboard_metrics.csv"
            df_metrics.to_csv(metrics_file, index=False)
            print(f"✓ TensorBoard metrics: {metrics_file}")

        # Save resource samples
        if self.resource_samples['timestamps']:
            resource_file = self.run_dir / "resource_samples.json"
            with open(resource_file, 'w') as f:
                json.dump(self.resource_samples, f, indent=2)
            print(f"✓ Resource samples: {resource_file}")

        # Save summary (single row CSV for Phase 2)
        summary_file = self.run_dir / "training_summary.csv"
        df_summary = pd.DataFrame([summary])
        df_summary.to_csv(summary_file, index=False)
        print(f"✓ Training summary: {summary_file}")

        # Also save as JSON for readability
        json_file = self.run_dir / "training_summary.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Training summary (JSON): {json_file}")

        print(f"\n✓ All data saved to: {self.run_dir}/")

        return summary_file

    def run_complete_pipeline(self):
        """Execute the complete training pipeline."""
        print(f"\n{'#'*70}")
        print(f"# MASTER TRAINING PIPELINE - Phase 2")
        print(f"# Run ID: {self.run_id}")
        print(f"# Config: {self.config_path}")
        print(f"{'#'*70}")

        # Step 1: Hardware
        if not self.step1_capture_hardware():
            return False

        # Step 2: Hyperparameters
        if not self.step2_extract_hyperparameters():
            return False

        # Step 3: Training
        training_success = self.step3_run_training()

        # Step 4: Extract metrics
        df_metrics = self.step4_extract_tensorboard()

        # Step 5: Create summary
        summary = self.step5_create_summary(df_metrics)

        # Step 6: Save everything
        summary_file = self.step6_save_everything(df_metrics, summary)

        # Final message
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE!")
        print(f"{'='*70}")
        print(f"\n✓ Training: {'SUCCESS' if training_success else 'FAILED/INTERRUPTED'}")
        print(f"✓ Summary file: {summary_file}")
        print(f"\nNext steps:")
        print(f"  1. Review the data in: {self.run_dir}/")
        print(f"  2. Run more experiments with different configs")
        print(f"  3. Combine all training_summary.csv files for ML analysis")
        print(f"\n{'='*70}\n")

        return True


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Master training pipeline with complete data collection"
    )
    parser.add_argument(
        "config_path",
        help="Path to training configuration YAML"
    )
    parser.add_argument(
        "run_id",
        help="Unique identifier for this training run"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Base directory for results (default: results)"
    )

    args = parser.parse_args()

    # Validate config exists
    if not os.path.exists(args.config_path):
        print(f"❌ Config file not found: {args.config_path}")
        sys.exit(1)

    # Run pipeline
    pipeline = MasterTrainingPipeline(
        config_path=args.config_path,
        run_id=args.run_id,
        results_dir=args.results_dir
    )

    success = pipeline.run_complete_pipeline()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

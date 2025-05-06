import os
import subprocess
import sys
import click
import questionary

import signal

ASSETS_DIR = "assets"
IGNORE = {"domelights"}
SKIP_QUESTIONS = False


def list_assets():
    return sorted(
        [
            d
            for d in os.listdir(ASSETS_DIR)
            if os.path.isdir(os.path.join(ASSETS_DIR, d)) and d not in IGNORE
        ]
    )


def run_script(script_path, asset_name, **kwargs):
    # Flatten kwargs into CLI args
    cmd = [sys.executable, script_path, "--asset-name", asset_name]
    for key, value in kwargs.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:  # only include flag if True
                cmd.append(flag)
        elif value is not None:
            cmd.extend([flag, str(value)])

    click.secho(" ".join(cmd), fg="cyan")

    # Start the subprocess in its own process group
    process = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid  # create new session for the child
    )

    try:
        # Wait for the subprocess to finish
        process.wait()
    except KeyboardInterrupt:
        click.secho("[main.py] CTRL+C detected. Sending SIGINT to subprocess...", fg="red")
        os.killpg(process.pid, signal.SIGINT)
        process.wait()  # Optionally wait again until subprocess exits

        raise KeyboardInterrupt

    return process.returncode == 0


@click.command()
def main():
    click.echo("Scanning for available assets...")
    assets = list_assets()
    if not assets:
        click.secho("No valid assets found.", fg="red")
        return

    default_asset = "japanesePlaneToy"

    while True:
        try:
            process_choices = ["Train NeRF", "Evaluate NeRF", "Process NeRF Input Data"]

            process_choice = questionary.select(
                "Choose a desired process:", choices=process_choices + ["EXIT->"]
            ).ask()

            if process_choice == "EXIT->" or process_choice is None:
                break

            asset_choice = questionary.select(
                "Select an asset to process:", choices=[f"{default_asset} (CURRENT DEFAULT ASSET)"] + assets
            ).ask()

            if asset_choice == f"{default_asset} (CURRENT DEFAULT ASSET)":
                asset_choice = default_asset
            if asset_choice is None:
                break

            if process_choice == "Train NeRF":
                click.secho(f"\nTraining NeRF model for '{asset_choice}'", fg="yellow")

                show_figures = (
                    questionary.confirm("Show figures generated during training?", default=False)
                    .skip_if(SKIP_QUESTIONS, default=False)
                    .ask()
                )

                if show_figures is None:
                    break

                run_script("src/nerf/nerf.py", asset_choice, show_figures=show_figures)

            elif process_choice == "Evaluate NeRF":
                click.secho(f"\nEvaluating NeRF model for '{asset_choice}'", fg="yellow")
                run_script("src/nerf/eval_nerf.py", asset_choice)

            elif process_choice == "Process NeRF Input Data":
                click.secho("\nProcessing input data to fit NeRF model for '{asset_choice}'", fg="yellow")
                click.secho("Step 1: Resize to 100px by 100px.", fg="cyan")
                run_script("src/nerf/resize_data.py", asset_choice)

                click.secho("Step 2: Convert data from JSON to NPZ format.", fg="cyan")
                run_script("src/nerf/convert_json_to_npz.py", asset_choice)

            default_asset = asset_choice
            click.secho("Process finished. Select new process or exit.", fg="green")

        except KeyboardInterrupt:
            break

    click.secho("Thanks for using Neural-for-USD!", fg="green")


if __name__ == "__main__":
    main()

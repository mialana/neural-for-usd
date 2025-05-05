import os
import subprocess
import sys
import click
import questionary

ASSETS_DIR = "assets"
IGNORE = {"domelights"}
SKIP_QUESTIONS = False

def list_assets():
    return sorted([
        d for d in os.listdir(ASSETS_DIR)
        if os.path.isdir(os.path.join(ASSETS_DIR, d)) and d not in IGNORE
    ])

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

    result = subprocess.run(
        cmd,
        check=True
    )
    return result.returncode == 0

@click.command()
def main():
    click.echo("Scanning for available assets...")
    assets = list_assets()
    if not assets:
        click.secho("No valid assets found.", fg="red")
        return

    asset_name = questionary.select(
        "Select an asset to process:",
        choices=assets + ["Skip->"]
    ).ask()

    try:
        if asset_name != "Skip->":
            click.secho(f"\nStep 1: resize_data.py for '{asset_name}'", fg="yellow")
            run_script("src/nerf/resize_data.py", asset_name)

            click.secho(f"\nStep 2: convert_json_to_npz.py for '{asset_name}'", fg="yellow")
            run_script("src/nerf/convert_json_to_npz.py", asset_name)

            assets = ["Continue->"] + assets

        process_choices = ["Train NeRF", "Evaluate NeRF"]
        
        next_step = questionary.select(
            "Which process next?",
            choices=process_choices
        ).ask()

        saved_asset_name = asset_name

        asset_name = questionary.select(
            "Reselect asset?",
            choices=assets
        ).ask()

        if asset_name == "Continue->":
            asset_name = saved_asset_name

        if next_step == "Train NeRF":
            click.secho(f"\nStep 3: nerf.py for '{asset_name}'", fg="yellow")

            show_figures = questionary.confirm("Show figures generated during training?").skip_if(SKIP_QUESTIONS, default=False).ask()
            run_script("src/nerf/nerf.py", asset_name, show_figures=show_figures)
        elif next_step == "Evaluate NeRF":
            click.secho(f"\nStep 3: eval_nerf.py for '{asset_name}'", fg="yellow")
            run_script("src/nerf/eval_nerf.py", asset_name)

    except subprocess.CalledProcessError as e:
        click.secho(f"Error: Script failed with exit code {e.returncode}", fg="red")

if __name__ == "__main__":
    main()

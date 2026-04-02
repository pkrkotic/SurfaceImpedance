# SurfaceImpedance

Command-line tool for computing and plotting surface impedance versus frequency
with several selectable physics models.

## Features

- Frequency sweeps over a logarithmic grid
- Multiple impedance models selectable from the CLI
- Multi-case comparison with repeated `--case` arguments
- Terminal summary of the chosen configuration
- CSV or JSON data export
- Optional plot display or PNG export

## Models

- `normal-skin`
  Uses the classical good-conductor skin-effect approximation.
- `half-space`
  Uses a generalized half-space wave-impedance approximation and accepts
  `tau = 0` for frequency-independent conductivity.
- `multi-layer`
  Uses a multilayer surface impedance recursion with a JSON-defined layer stack.
- `rough-single`
  Uses the Helmreich-Gold surface roughness gradient model for a single rough transition.

## Quick start

Create a virtual environment and install the package:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -e .[plot]
```

## Desktop launcher

If you want a double-clickable launcher on Windows, this repository includes:

- `Launch SurfaceImpedance.cmd`
- `launch-surface-impedance.ps1`

Running the `.cmd` file will:

- create `.venv` on first launch if needed,
- install the project into that environment,
- open a new PowerShell window in the project folder with the environment activated.

You can create a desktop shortcut that points to `Launch SurfaceImpedance.cmd`, and
optionally change the shortcut icon in the shortcut properties.

List available models:

```powershell
py -m surface_impedance.cli --list-models
```

Run a classical skin-effect sweep and save both data and a plot:

```powershell
py -m surface_impedance.cli `
  --model normal-skin `
  --f-min 1e3 `
  --f-max 1e9 `
  --points 300 `
  --sigma 5.8e7 `
  --export results\copper.csv `
  --plot results\copper.png
```

Compare multiple cases on one shared plot with different conductivities:

```powershell
py -m surface_impedance.cli `
  --f-min 1e3 `
  --f-max 1e9 `
  --points 300 `
  --case "label=copper-classic,model=normal-skin,sigma=5.8e7" `
  --case "label=copper-half-space,model=half-space,sigma=4.7e7,tau=0,epsr_real=1.0,epsr_imag=0.0,mur_real=1.0,mur_imag=0.0" `
  --case "label=steel-classic,model=normal-skin,sigma=1.4e6,mu_r=80" `
  --export results\comparison.csv `
  --plot results\comparison.png
```

Run the half-space model and display the plot interactively:

```powershell
py -m surface_impedance.cli `
  --model half-space `
  --tau 0 `
  --sigma 5.8e7 `
  --epsr-real 1.0 `
  --epsr-imag 0.0 `
  --mur-real 1.0 `
  --mur-imag 0.0 `
  --show-plot
```

Run the multilayer model using the example stack file in the project root:

```powershell
py -m surface_impedance.cli `
  --model multi-layer `
  --layers-file multi-layer-smooth.json `
  --f-min 1e3 `
  --f-max 1e9 `
  --points 300 `
  --show-plot
```

Run the single-layer roughness model:

```powershell
py -m surface_impedance.cli `
  --model rough-single `
  --sigma-metal 5.8e7 `
  --rq 0.5e-6 `
  --xmin-factor -5 `
  --show-plot
```

## Notes

- Surface impedance is reported in ohms.
- When `--case` is used, each case shares the same frequency grid but can
  override its own model parameters.
- Multi-layer plots automatically include a representative layer schematic
  below the impedance plot. The base layer is the reference medium at `x = 0`,
  and each JSON layer extends to the right by its thickness.
- The multilayer JSON file now defines both the `base_layer` and the finite
  `layers` on top of it.
- For the smooth multilayer JSON file, `Layer 1` is the finite layer closest
  to the base layer, and the last layer in the list is the one closest to air.
- Plotting requires the optional `plot` dependency group.

## Tests

```powershell
py -m unittest discover -s tests
```

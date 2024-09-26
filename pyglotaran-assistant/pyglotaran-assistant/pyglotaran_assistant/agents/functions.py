from typing import Optional
from typing_extensions import Annotated
import os


import nbformat
from nbformat.v4 import new_notebook
from pyparamgui import (
    IRF,
    KineticParameters,
    Settings,
    SimulationConfig,
    Coordinates,
    SpectralParameters,
    generate_model_parameter_and_data_files,
)


def read_file(
    file_path: Annotated[str, "The relative file path of the file to read "]
) -> Annotated[
    str | OSError,
    "Returns the contents of the file as a string, otherwise an OSError if it can't read it",
]:
    """Reads a file and returns its contents.

    Args:
        file_path (str): The path to the file to be read.

    Returns:
        str: The contents of the file.
    """
    with open(file_path, "r") as file:
        return file.read()


def read_notebook_content(
    notebook_path: Annotated[str, "The relative path to the notebook file"],
    max_cells: Annotated[
        Optional[int],
        "Maximum number of cells to include (inclusive), if None, include all cells",
    ] = None,
) -> Annotated[
    str | OSError | Exception,
    "Returns the contents of the notebook as a string, otherwise an OSError or some generic Exception, if it can't read it or if it faces some other issue",
]:
    """
    Retrieves the content of Jupyter notebook cells (only code) and their respective outputs as a multiline string.

    Parameters:
    - notebook_path (str): The path to the notebook file.
    - max_cells (Optional[int]): "Maximum number of cells to include (inclusive), if None, include all cells"

    Returns:
    - str: A multiline string containing the content of the notebook cells and their outputs.
    """
    with open(notebook_path, "r") as f:
        notebook = nbformat.read(f, as_version=4)

    total_cells = len(notebook.cells)

    cells_to_include = min(total_cells, max_cells) if max_cells else total_cells

    output_lines = []

    for index, cell in enumerate(notebook.cells[:cells_to_include]):
        output_lines.append("_____________________________________________")

        if cell.cell_type == "code":
            output_lines.append(f"CELL INPUT {index + 1}:")
            output_lines.append(cell.source.strip())

            if "outputs" in cell:
                output_lines.append(f"CELL OUTPUT {index + 1}:")
                for output in cell.outputs:
                    if output.output_type == "stream":
                        output_lines.append(output.text.strip())
                    elif output.output_type == "execute_result":
                        output_lines.append(output.data["text/plain"].strip())
                    elif output.output_type == "error":
                        output_lines.append("\n".join(output.traceback).strip())

        output_lines.append("_____________________________________________")
        output_lines.append("")

    return "\n".join(output_lines)


def generate_pyglotaran_model_parameter_data_files(
    kinetic_parameters: Annotated[
        KineticParameters, "Kinetic parameters for the simulation."
    ],
    spectral_parameters: Annotated[
        SpectralParameters, "Spectral parameters for the simulation."
    ],
    coordinates: Annotated[
        Coordinates, "Combined time and spectral coordinates for the simulation."
    ],
    settings: Annotated[Settings, "Other settings for the simulation."],
    irf: Annotated[
        IRF, "Instrument Response Function (IRF) settings for the simulation."
    ],
    model_file_name: Annotated[
        str, "The name of the file to save the model."
    ] = "model.yaml",
    parameter_file_name: Annotated[
        str, "The name of the file to save the parameters."
    ] = "parameter.csv",
    data_file_name: Annotated[
        str, "The name of the file to save the data."
    ] = "dataset.nc",
) -> Annotated[
    str | RuntimeError,
    "Returns a completion string message, otherwise raises a RuntimeError if it faced any issue while creating files",
]:
    """
    Generates Pyglotaran model, parameter, and data files based on the input from the widget instance and file names.

    Parameters:
    - kinetic_parameters (KineticParameters): Kinetic parameters for the simulation.
    - spectral_parameters (SpectralParameters): Spectral parameters for the simulation.
    - coordinates (Coordinates): Combined time and spectral coordinates for the simulation.
    - settings (Settings): Other settings for the simulation.
    - irf (IRF): Instrument Response Function (IRF) settings for the simulation.
    - model_file_name (str): The name of the file to save the model.
    - parameter_file_name (str): The name of the file to save the parameters.
    - data_file_name (str): The name of the file to save the data.

    Returns:
    None, but writes files to the disk.
    """
    simulation_config = SimulationConfig(
        kinetic_parameters=kinetic_parameters,
        spectral_parameters=spectral_parameters,
        coordinates=coordinates,
        settings=settings,
        irf=irf,
    )
    try:
        generate_model_parameter_and_data_files(
            simulation_config=simulation_config,
            model_file_name=model_file_name,
            parameter_file_name=parameter_file_name,
            data_file_name=data_file_name,
        )
    except Exception as e:
        raise RuntimeError("Files could not be created.") from e


def create_empty_jupyter_notebook(
    notebook_name: Annotated[
        str, "The name of the notebook without the .ipynb extension."
    ]
) -> Annotated[
    None | str,
    "None if the notebook was created successfully, otherwise returns a write error message.",
]:
    """
    Creates an empty Jupyter notebook with the given name.

    Parameters:
    - notebook_name (str): The name of the notebook without the .ipynb extension.

    Returns:
    None if the notebook was created successfully, otherwise returns a write error message.
    """

    if not notebook_name.endswith(".ipynb"):
        notebook_name += ".ipynb"

    if os.path.exists(notebook_name):
        return "A notebook with that name already exists."

    nb = new_notebook()

    try:
        with open(notebook_name, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        return None
    except IOError as e:
        return f"An error occurred while writing the notebook: {e}"

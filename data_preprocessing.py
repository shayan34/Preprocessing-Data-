import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


def preprocess_data(input_file, output_file):
    """
    Preprocess the data from input CSV file and save it to output CSV file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.

    Returns:
        pd.DataFrame: Processed data as a DataFrame.
    """
    # Read the input CSV file, skipping unnecessary header rows
    df = pd.read_csv(input_file, sep=";", skiprows=6, header=None)

    # Extract sensor names and units from the headers
    headers = df.iloc[0]
    units = df.iloc[1]

    # Combine sensor names and units into a new column header format
    new_columns = [f'{name}_{unit.strip("<>")}' for name, unit in zip(headers, units)]

    # Set the new column names and remove unnecessary rows
    df.columns = new_columns
    df = df.iloc[2:]
    df = df.dropna()

    # Save the reformatted data to output.csv
    df.to_csv(output_file, sep=";", index=False)
    print("Preprocess data done")

    return df


def create_subplots(data):
    """
    Create a figure with three subplots showing time series data for the first three sensors.

    Args:
        data (pd.DataFrame): Processed data as a DataFrame.
    """
    # Create a figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    # Get the first three sensor names from the DataFrame columns
    sensors = data.columns[1:4]  # Assuming the time column is at index 0

    for i, sensor in enumerate(sensors):
        ax = axes[i]
        ax.plot(data["t_(s)"], data[sensor], label=sensor)
        ax.set_title(f"Time Series for {sensor}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        # Customize y-axis tick formatting (e.g., reduce the number of ticks)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        ax.legend(loc="upper right")  # Adjust the legend placement
        # Customize x-axis tick labels
        x_ticks = np.arange(
            0, len(data), step=len(data) // 10
        )  # Adjust the step value as needed
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            data["t_(s)"].iloc[x_ticks], rotation=45
        )  # Rotate x-axis labels for readability

    # Adjust subplot layout
    plt.tight_layout()

    # Save the figure as *.eps
    plt.savefig("output.eps", format="eps")
    print("Figure saved as output.eps")


def main():
    input_file = "input.csv"
    output_file = "output.csv"

    # Preprocess the data
    df = preprocess_data(input_file, output_file)

    # Create subplots and save the figure
    create_subplots(df)


if __name__ == "__main__":
    main()

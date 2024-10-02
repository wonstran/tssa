# tssa - Transportation Surrogate Safety Analysis Tools

TSSA is a set of tools designed for conducting transportation sorrogate safety studies, specifically focusing on pedestrian and bicycle safety. It leverages AI and machine learning to analyze and interpret transportation data, providing insights and recommendations for improving safety in urban environments.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Modeling and Analysis](#modeling-and-analysis)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Collection**: Collect data from various sources like traffic cameras, sensors, and public datasets.
- **Data Analysis**: Use AI/ML algorithms to analyze trends and identify safety risks for pedestrians and cyclists.
- **Visualization**: Generate heatmaps and graphs for easy interpretation of data.
- **Predictive Modeling**: Predict future accident hotspots and suggest preventive measures.
- **Recommendations**: Provide actionable recommendations based on data analysis to improve pedestrian and bicycle safety.

## Installation

To get started with TSSA, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/tssa.git
    cd tssa
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment variables (e.g., API keys, database credentials) in a `.env` file.

4. Run the application:
    ```bash
    python app.py
    ```

## Usage

After installation, you can run the application locally to:

1. Collect data from transportation systems and traffic sensors.
2. Run AI models to analyze transportation safety data.
3. Visualize results in a web interface or save them for further reporting.

Sample usage:

```bash
python analysis.py --data=data/traffic_data.csv --output=output/safety_report.pdf

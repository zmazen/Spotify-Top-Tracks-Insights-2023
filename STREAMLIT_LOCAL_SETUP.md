# Full Tutorial: Running Streamlit Projects Locally

This guide explains step-by-step how to run the Streamlit projects **locally** on your computer. Follow carefully to get the projects running smoothly.


## 1. Prerequisites

Before running the projects, make sure you have:

- **Python 3.9 or higher** installed.
  - Download from: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- **Git** installed (optional but recommended to clone repositories)
  - Download from: [https://git-scm.com/downloads](https://git-scm.com/downloads)

## 2. Clone the Project Repository

Open a terminal or command prompt and run:

```bash
git clone <REPOSITORY_URL>
cd <PROJECT_FOLDER_NAME>
```

- Replace `<REPOSITORY_URL>` with the URL of the GitHub repository.  
- Replace `<PROJECT_FOLDER_NAME>` with the folder name of the project.

Example:

```bash
git clone https://github.com/yourusername/world-happiness-report-2023-analysis.git
cd world-happiness-report-2023-analysis
```

## 3. Install Dependencies

All necessary Python libraries are listed in the `requirements.txt` file. To install them, run:

```bash
pip install -r requirements.txt
```

### **Requirements List Example**

A typical `requirements.txt` may include:

```
streamlit
pandas
plotly
matplotlib
numpy
scikit-learn
statsmodels
```

> Make sure `requirements.txt` is in the root directory of the project.

## 4. Running the Streamlit Application

Once dependencies are installed, you can launch the app with:

```bash
streamlit run app.py
```

- `app.py` should be replaced with the main Python file of your project if named differently.  
- Streamlit will automatically open your default web browser.  

Default URL:

```
http://localhost:8501
```

You can interact with all charts, models, and visualizations from there.

## 6. Optional: Running Without Git

If you prefer not to use Git:

1. Download the project as a ZIP from GitHub.  
2. Extract the ZIP file.  
3. Navigate to the folder in terminal.  
4. Run the same steps: install dependencies and launch `streamlit run app.py`.

## 8. Summary

1. Install Python 3.9+ and optionally Git.  
2. Clone or download the project.  
3. Install all dependencies from `requirements.txt`.  
4. Run `streamlit run app.py`.  
5. Open the browser to interact with the app.  

Following these steps ensures the project works **locally** exactly as intended, with all interactive charts and ML models functional.
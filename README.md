# CS 7643 - Deep Learning Final Project: Deep Learning for Waste Classification and Recycling

## Description
With the increasing emphasis on sustainability and waste management, automating the classification of waste materials plays a crucial role in improving recycling efficiency. This project aims to develop a deep learning model capable of automatically identifying various types of waste, such as paper, plastic, metal, glass, and cardboard, from image data. The project will leverage a pre-existing dataset, containing labeled images of different types of waste. The goal is to explore various deep learning architectures, including custom CNNs and transfer learning techniques, to achieve a high level of accuracy in waste classification and object detection. This project could have far-reaching applications in waste management systems, landfill monitoring, and environmental education.

## 🏛 Repository Structure

```
DEEPLEARNING_PROJECT
├── data/                            --> Raw and processed data
│   ├── raw/
│   └── processed/
│
├── notebooks/                       --> Jupyter notebooks
│   ├── exploration.ipynb            --> Dataset understanding and visualization
│   └── final_project.ipynb          --> Final executable notebook (calls src/main.py)
│
├── src/                             --> Source code
│   ├── config/
│   │   └── config_file.py           --> Project-level configuration (e.g., class names, device, paths)
│   ├── evaluate/
│   │   └── evaluate.py              --> Evaluation logic for trained models
│   ├── models/
│   │   └── model_template.py        --> Model definition (e.g., CNN or custom architecture)
│   ├── predict/
│   │   └── predict.py               --> Inference on new data
│   ├── preprocessing/
│   │   ├── helpers/
│   │   │   └── data_helpers.py      --> Preprocessing utilities (e.g., data split, augmentation)
│   │   └── preprocessing.py         --> Data preparation workflow (transforms, loaders, etc.)
│   ├── train/
│   │   └── train.py                 --> Training logic (training loop, optimizer, loss)
│   ├── utils/
│   │   ├── common.py                --> General utilities
│   │   ├── logger.py                --> Logging configuration
│   │   └── plots.py                 --> Plotting functions for results or metrics
│   └── main.py                      --> Project orchestration: from preprocessing to training and evaluation
│
├── outputs/                         --> Saved models, logs, predictions
│   ├── models/
│   ├── logs/
│   └── results/
│
├── requirements.txt                 --> Project dependencies
├── .gitignore                       --> Files and folders to ignore in git
└── README.md                        --> Project overview (this file)
```

## 🚀 How to run the project

### 1. Environment Setup

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Final Notebook

All training, evaluation and analysis are integrated in the `final_project.ipynb` notebook.

Open it in Jupyter or VSCode and run each section interactively.


## 🤝 Contributing

To collaborate efficiently on this class project, we followed a simple and effective Git workflow using the `main` branch and feature branches:

1. We each created feature branches for specific tasks:
   ```bash
   git checkout -b feature/task_name
   ```

2. After completing our changes, we staged and committed them:
   ```bash
   git add .
   git commit -m "Describe your changes here"
   ```

3. We pushed our branches to the remote repository:
   ```bash
   git push origin feature/task_name
   ```

4. Pull Requests were created and reviewed by teammates before merging into `main`.

5. We regularly pulled the latest changes from `main` to stay updated:
   ```bash
   git pull origin main
   ```

This approach allowed us to work in parallel, avoid merge conflicts, and maintain a clean and organized project structure.

## 👥 Team

- [Lourdes Artacho Sierras](mailto:lsierras3@gatech.edu)
- [Jorge Calvar Seco](mailto:jseco3@gatech.edu)
- [Carlota López Argote](mailto:cargote3@gatech.edu)

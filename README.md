# 📊 data2model

**data2model** is a multi-page interactive Streamlit application that lets you upload any dataset, handle missing values, train multiple machine learning models (classification or regression), and visualize results — all without writing a single line of code.

> 🔥 Perfect for students, data scientists, and ML engineers who want an end-to-end playground for rapid experimentation.

Live sreamlit app [click here] (https://data2model-08.streamlit.app/Visualization)
---

## 🚀 Features

- 📂 Upload CSV, Excel, or JSON files
- 🧹 Clean data with built-in missing value handlers (drop, fill, interpolate)
- 🧠 Choose task: **Classification** or **Regression**
- 🎯 Select multiple ML models to train
- ⚖️ Automatic feature scaling for Classification and SVR
- 📊 Visualize:
  - Confusion Matrix
  - Predicted vs Actual
  - Residual Distribution
- 🧪 Evaluate with Accuracy, MSE, R²
- 📈 Visualize dataset relationships (correlation matrix)
- 💾 Modular codebase ready for:
  - Model saving with `joblib`

---

## 🖥️ App Pages

| Page                | Description                                  |
|---------------------|----------------------------------------------|
| 📂 Data Handling     | Upload, inspect, and clean your dataset      |
| 🧠 Model Training     | Select and train ML models                  |
| 📈 Visualization      | View performance and prediction insights    |

---

## 📂 Project Structure

```
data2model/
│
├── 🏠_Home.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── data/                  # Uploaded/test datasets
├── models/                # Saved models 
├── media/                 # Images/plots 
│
├── src/
│   ├── data_loader.py
│   ├── dfvisualization.py
│   ├── preprocess.py
│   ├── model_saver.py
│   ├── model_selector.py
│   ├── preprocessor.py
│   ├── report_generator.py
│   └── visualizer.py
│
└── pages/
    ├── 1_Data_Handeling.py
    ├── 2_Model_Training.py
    └── 3_Visualization.py
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/TejaDurgam08/data2model.git
cd data2model
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
# or
source .venv/bin/activate       # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run 🏠_Home.py
```

It will open in your browser at [http://localhost:8501](http://localhost:8501)

---

## 🌍 Deploy on Streamlit Cloud

1. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Click **New App**
3. Choose this GitHub repo
4. Set `app.py` as the main file
5. Click **Deploy**

Your app will be live.


---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

Built with ❤️ by [Teja Durgam](https://github.com/TejaDurgam08)

If you like this project, consider giving it a ⭐ on GitHub!

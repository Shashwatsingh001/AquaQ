# AquaQ - Water Quality Prediction System

## 📌 Project Overview
AquaQ is an advanced water quality prediction system designed to analyze and forecast critical water parameters such as **dissolved oxygen, turbidity, temperature, and pH levels**. The project leverages **Google Earth Engine API**, **machine learning models**, and an interactive dashboard to provide real-time insights into water quality.

## 🚀 Features
- **Data Collection**: Utilizes **Google Earth Engine API** to fetch standardized satellite and historical water quality data.
- **Machine Learning Models**: Implements **Random Forest** and **Deep Learning** techniques to predict water parameters.
- **Interactive Dashboard**: Built using **Streamlit** for real-time visualization of water quality trends.
- **Preprocessing & Optimization**: Cleans and preprocesses data with **feature engineering** and missing value handling for improved accuracy.

## 🛠️ Tech Stack
- **Programming Language**: Python
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, TensorFlow
- **Visualization**: Matplotlib, Seaborn
- **API & Deployment**: Google Earth Engine API, Streamlit

## 📂 Project Structure
```
AquaQ/src
│-- data/                   # Raw and processed datasets
│-- savedmodels/                 # Trained ML models
│-- satellitefunction/              # Jupyter notebooks for data analysis
│-- models/                # Scripts for data preprocessing and model training
│-- deployement/main.py                  # Streamlit dashboard for visualization
│-- frontend/                       
│-- README.md               # Project documentation
```

## 🔧 Installation & Usage
### Prerequisites
- Python 3.8+
- Google Earth Engine API setup

### Installation
```sh
# Clone the repository
git clone https://github.com/YeliwZinn/AquaQ.git
cd AquaQ

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```sh
streamlit run main.py
```

## 📊 Model Training
To train the machine learning models, run the following command:
```sh
python models/train_model.py
```

## 📈 Results
AquaQ achieves high accuracy in predicting dissolved oxygen levels and other water parameters, demonstrating how AI can assist in sustainable environmental monitoring.

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests.

## 📜 License
This project is licensed under the MIT License.

## 📞 Contact
For any inquiries, reach out via [LinkedIn](https://www.linkedin.com/in/shubham-shankar-a7b1b2285/) or [GitHub](https://github.com/YeliwZinn).


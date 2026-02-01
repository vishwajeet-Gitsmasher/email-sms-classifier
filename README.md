# 📧📱 Email & SMS Spam Classifier

A Machine Learning web application that classifies **Email or SMS messages** as **Spam** or **Not Spam (Ham)** using **Natural Language Processing (NLP)** and a **Multinomial Naive Bayes** model.  
The app is built with **Python & Streamlit** and deployed on **Render**.

---

## 🚀 Live Demo
👉 *Deployed on Render*  
https://sms-email-spam-classifier-6tyk.onrender.com

---

## 📌 Features
- Classifies **SMS / Email text** as Spam or Not Spam
- Real-time prediction via web interface
- Uses **TF-IDF Vectorization**
- Lightweight & fast **Multinomial Naive Bayes model**
- Clean and simple **Streamlit UI**
- Fully deployed on **Render**

---

## 🧠 Machine Learning Workflow
1. Text Cleaning & Preprocessing
2. Tokenization using **NLTK**
3. Stopword Removal & Stemming
4. Feature Extraction using **TF-IDF Vectorizer**
5. Classification using **Multinomial Naive Bayes**

---

## 🛠 Tech Stack
- **Python**
- **Streamlit**
- **Scikit-learn**
- **NLTK**
- **Pandas & NumPy**
- **Render (Deployment)**

---

## 📂 Project Structure
```

email-sms-classifier/
│
├── app.py                     # Streamlit application
├── model.pkl                  # Trained Multinomial NB model
├── vectorizer.pkl             # TF-IDF Vectorizer
├── requirements.txt           # Python dependencies
├── setup.sh                   # Render setup script
├── nltk.txt                   # NLTK resource requirements
├── spam.csv                   # Dataset
├── sms-spam-detection.ipynb   # Training & experimentation notebook
├── README.md                  # Project documentation

````

---

## ⚙️ Installation & Setup (Local)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/vishwajeet-Gitsmasher/email-sms-classifier.git
cd email-sms-classifier
````

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download NLTK resources

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### 5️⃣ Run the app

```bash
streamlit run app.py
```

---

## 🌐 Deployment on Render

* Web Service type: **Python**
* Build Command:

```bash
pip install -r requirements.txt
```

* Start Command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 📊 Dataset

* Source: SMS Spam Collection Dataset
* Contains labeled SMS messages (`spam` / `ham`)
* Used for training and evaluation

---

## 🧪 Model Performance

* Algorithm: **Multinomial Naive Bayes**
* High precision on spam detection
* Lightweight & efficient for real-time inference

---

## 🔮 Future Improvements

* Add Email file upload support
* Improve UI with charts & confidence scores
* Try advanced models (Logistic Regression, SVM)
* Deploy using Docker

---

## 👨‍💻 Author

**Vishwajeet**
📌 Machine Learning Engineer/ Data Scientist

Feel free to ⭐ the repository if you found this useful!

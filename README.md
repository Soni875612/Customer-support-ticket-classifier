# 🎫 Customer Support Ticket Classifier

> An end-to-end ML pipeline that automatically classifies customer support tickets by **category** and **urgency level**, and recommends the appropriate action — built with TF-IDF + Logistic Regression.

---

## 📌 Overview

Customer support teams deal with hundreds of tickets daily. Manually triaging them wastes time and delays resolution. This project automates that triage using a trained machine learning model that:

- **Classifies** the ticket into one of 4 categories
- **Detects** urgency level (critical → low)
- **Recommends** a specific action based on category × urgency matrix
- **Exposes** a Flask REST API + interactive web dashboard

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📂 **4 Categories** | `billing`, `technical`, `complaint`, `general` |
| 🚨 **4 Urgency Levels** | `critical`, `high`, `medium`, `low` |
| 🤖 **ML Model** | TF-IDF Vectorizer + Logistic Regression (scikit-learn) |
| 🧹 **Text Preprocessing** | Lowercasing, URL/mention removal, regex cleaning |
| 📋 **SLA Mapping** | Auto-assigns SLA based on urgency |
| ⚡ **Recommended Actions** | 16 pre-defined actions (4 categories × 4 urgency levels) |
| 🌐 **REST API** | Flask-based `/api/classify` endpoint |
| 📊 **Dashboard** | Interactive HTML dashboard (`SupportAI_Dashboard.html`) |
| 📈 **Metrics Endpoint** | `/api/metrics` returns live model accuracy |

---

## 📁 Project Structure

```
ticket_classifier/
│
├── train_model.py          # Train and save both ML models
├── classifier.py           # Core classification logic (used by Flask)
├── app.py                  # Flask web server + REST API
│
├── category_model.pkl      # Saved category classifier
├── urgency_model.pkl       # Saved urgency classifier
├── model_metrics.json      # Accuracy & F1 scores
├── requirements.txt        # Python dependencies
│
├── SupportAI_Dashboard.html  # Interactive web dashboard
└── .vscode/
    └── settings.json
```

---


## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-support-ticket-classifier.git
cd customer-support-ticket-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
```
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
nltk>=3.8.0
flask>=2.0.0
flask-cors>=3.0.0
```

### 3. Train the Models *(one-time setup)*

```bash
python train_model.py
```

This will:
- Train both category and urgency models on synthetic data
- Save `category_model.pkl` and `urgency_model.pkl`
- Save `model_metrics.json` with accuracy scores

### 4. Run the Flask App

```bash
python app.py
```

App will start at: `http://localhost:5000`

---

## 🚀 Usage

### Option A: Python (Direct)

```python
from classifier import classify

result = classify("I was charged twice and my account is frozen!")

print(result['category']['label'])       # billing
print(result['category']['confidence'])  # 87.3
print(result['urgency']['label'])        # high
print(result['urgency']['confidence'])   # 76.1
print(result['recommended_action'])      # Route to billing specialist within 4 hours.
```

### Option B: REST API

**Classify a ticket:**

```bash
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "App keeps crashing every time I open it"}'
```

**Sample Response:**

```json
{
  "input": "App keeps crashing every time I open it",
  "category": {
    "label": "technical",
    "confidence": 91.2,
    "icon": "⚙️",
    "color": "#3b82f6",
    "desc": "Bugs, errors, crashes, features",
    "breakdown": {
      "billing": 2.1,
      "technical": 91.2,
      "complaint": 4.5,
      "general": 2.2
    }
  },
  "urgency": {
    "label": "medium",
    "confidence": 68.4,
    "icon": "🟡",
    "sla": "< 24 hours",
    "priority": 2
  },
  "recommended_action": "Add to tech support queue with standard SLA."
}
```

**Get model metrics:**

```bash
curl http://localhost:5000/api/metrics
```

### Option C: Web Dashboard

Open `http://localhost:5000` in your browser to use the interactive dashboard.

---

## 🏷️ Category Labels

| Category | Description | SLA (varies by urgency) |
|---|---|---|
| 💳 `billing` | Payment, invoices, refunds, charges | 1 hr – 72 hrs |
| ⚙️ `technical` | Bugs, errors, crashes, features | 1 hr – 72 hrs |
| 😤 `complaint` | Dissatisfaction, negative experience | 1 hr – 72 hrs |
| 💬 `general` | Questions, info, how-to queries | 1 hr – 72 hrs |

## 🚨 Urgency Levels

| Urgency | Icon | SLA | Example |
|---|---|---|---|
| `critical` | 🔴 | < 1 hour | Production down, data breach |
| `high` | 🟠 | < 4 hours | Team blocked, client demo tomorrow |
| `medium` | 🟡 | < 24 hours | Feature not working as expected |
| `low` | 🟢 | < 72 hours | General questions, suggestions |

---

## 🧠 Model Details

### Architecture

Both models use the same scikit-learn Pipeline:

```
Raw Text → clean_text() → TF-IDF Vectorizer → Logistic Regression → Label
```

| Model | TF-IDF n-grams | Max Features | LR C value |
|---|---|---|---|
| Category | (1, 2) | 5000 | 5.0 |
| Urgency | (1, 3) | 3000 | 3.0 |

### Text Preprocessing (`clean_text`)

```python
text = text.lower()                          # lowercase
text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
text = re.sub(r'@\w+', '', text)            # remove mentions
text = re.sub(r'[^a-zA-Z\s!?]', ' ', text) # keep only letters + !?
text = re.sub(r'\s+', ' ', text).strip()    # normalize spaces
```

### Training Data

Synthetic dataset inspired by the [Kaggle Customer Support on Twitter dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter):

| Class | Training Samples |
|---|---|
| Billing | 60 |
| Technical | 60 |
| Complaint | 60 |
| General | 40 |
| **Total** | **220** |

Urgency model: 60 samples (15 per level)

### Model Performance *(on synthetic test set)*

| Metric | Score |
|---|---|
| Category Accuracy | ~70.5% |
| Urgency Accuracy | ~58.3% |
| Category F1 (macro) | ~69.5% |
| Urgency F1 (macro) | ~59.2% |

> ⚠️ **Note:** These scores are on a synthetic test set. Performance improves significantly when trained on real labeled data (e.g., the Kaggle Twitter Support dataset).

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive dashboard |
| `POST` | `/api/classify` | Classify a support ticket |
| `GET` | `/api/metrics` | Returns model accuracy metrics |

**POST `/api/classify` — Request Body:**

```json
{
  "text": "Your ticket text here"
}
```

---

## 🛠️ Improving Model Accuracy

The current model is trained on synthetic data. To improve performance for production use:

1. **Use real data** — Replace synthetic samples with actual labeled support tickets
2. **More samples** — 1000+ samples per category is ideal
3. **Better features** — Try adding features like ticket source, customer tier, time of day
4. **Advanced models** — Swap Logistic Regression for XGBoost, or fine-tune a BERT model
5. **Active learning** — Label misclassified tickets and retrain iteratively

---

## 🧪 Running Tests

```bash
python classifier.py
```

This runs 5 built-in test cases and prints category, urgency, and recommended action for each.

---

## 📦 Dependencies

```
scikit-learn   — ML models and pipelines
pandas         — Data handling during training
numpy          — Numerical operations
nltk           — NLP utilities
flask          — Web server and REST API
flask-cors     — Cross-Origin Resource Sharing
```

---

## 🎬Project Screenshots
<img width="1918" height="926" alt="image" src="https://github.com/user-attachments/assets/b433fcc0-2336-4358-a485-69ddd50ae6ed" />


<img width="1919" height="906" alt="image" src="https://github.com/user-attachments/assets/2137cd59-d7b3-4b37-a418-8f4791b68fc1" />


<img width="1919" height="922" alt="image" src="https://github.com/user-attachments/assets/774d759e-6594-4719-a893-01c2508a7c05" />


<img width="1909" height="913" alt="image" src="https://github.com/user-attachments/assets/b6c92c83-af5d-41b3-bdcc-cc18132f5327" />


<img width="1914" height="909" alt="image" src="https://github.com/user-attachments/assets/1a2dccbd-7536-4825-aed5-d43c32460f9d" />


<img width="1710" height="836" alt="image" src="https://github.com/user-attachments/assets/173a72ba-e1ce-4a36-bd1b-029ad1c9fc17" />


<img width="1917" height="776" alt="image" src="https://github.com/user-attachments/assets/c12292b9-cb24-4f00-bcf3-9ef8e53d0593" />


<img width="1919" height="629" alt="image" src="https://github.com/user-attachments/assets/6dd358f5-1526-4372-bafc-94a54be62629" />


<img width="1919" height="880" alt="image" src="https://github.com/user-attachments/assets/e8e170ee-87ba-491a-855b-161280c780b0" />


<img width="1666" height="853" alt="image" src="https://github.com/user-attachments/assets/84ec36ad-807e-4c25-a879-7497a4538a15" />


<img width="1658" height="730" alt="image" src="https://github.com/user-attachments/assets/ee442b78-267e-40ff-aeca-58f59f7ad8f7" />




---

## 👩‍💻 Author

**Soni**  
Project — Customer Support Ticket Classifier

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/soni-devi-131a9938b/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Soni875612)
[![LeetCode](https://img.shields.io/badge/LeetCode-FFA116?style=for-the-badge&logo=leetcode&logoColor=black)](https://leetcode.com/u/soni_2007/)
https://soni875612.github.io/

---

## 📄 License

This project is for educational purposes. Feel free to fork, modify, and build on top of it.

---

## 🙏 Acknowledgements

- Training data inspired by the [Customer Support on Twitter dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) on Kaggle
- Built with [scikit-learn](https://scikit-learn.org/) and [Flask](https://flask.palletsprojects.com/)

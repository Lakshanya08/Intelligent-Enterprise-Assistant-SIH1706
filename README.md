# 🤖 Intelligent Enterprise Assistant

**AI-Powered Enterprise Chatbot using Python, PyTorch and Natural Language Processing**

---

## 📖 Overview

The **Intelligent Enterprise Assistant** is an AI-driven chatbot developed to enhance organizational efficiency by automating employee support services. The system understands user queries written in natural language and provides intelligent responses related to HR policies, IT helpdesk support, and enterprise information.

The chatbot uses **Natural Language Processing (NLP)** and **Deep Learning** techniques to identify user intent and generate appropriate responses through an interactive interface.

---

## 🎯 Objectives

- **Automate enterprise support queries**
- **Reduce manual workload for HR and IT teams**
- **Provide instant responses to employees**
- **Improve organizational productivity**
- **Demonstrate AI chatbot integration in enterprises**

---

## ✨ Features

- **Natural Language Understanding (NLU)**
- **Deep Learning-based Intent Detection**
- **HR Query Support**
- **IT Helpdesk Automation**
- **Context-aware Responses**
- **Flask Web Interface**
- **Custom Knowledge Base**
- **Easy Deployment**

---

## 🧠 System Workflow

1. **User enters a query**
2. Text is processed using **NLTK**
3. Query converted into **Bag-of-Words vector**
4. **PyTorch neural network** predicts intent
5. Matching response is selected
6. Response displayed via **Flask web application**

---

## 🛠️ Technology Stack

- **Python** – Core programming language  
- **PyTorch** – Deep learning framework  
- **NLTK** – Natural language processing  
- **Flask** – Backend web framework  
- **Scikit-learn** – Data processing utilities  
- **HTML, CSS, JavaScript** – Frontend interface  

---

## 📂 Project Structure

Intelligent-Enterprise-Assistant/
│
├── app.py
├── chatbot.py
├── train.py
├── intents.json
├── model.pth
├── requirements.txt
│
├── templates/
├── static/
└── README.md


---

## ⚙️ Installation

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/intelligent-enterprise-assistant.git
cd intelligent-enterprise-assistant

```
### 2. Install Dependencies
pip install torch nltk scikit-learn flask
### 3. Download NLTK Resources
Run once in Python:
import nltk
nltk.download('punkt')

## 🚀 Running the Project
Step 1 — Train the Model
python train.py
This will train the neural network and create the trained model file model.pth.
Step 2 — Run Chatbot
python chatbot.py
Step 3 — Run Web Application
python app.py

## Open your browser and visit:

http://127.0.0.1:5000
### Example Queries

1.What is the leave policy?
2.When is salary credited?
3.I need IT support
4.Reset my password
5.Hello

### 🧩 Model Details

1.Model Type: Feedforward Neural Network
2.Framework: PyTorch
3.Input: Bag-of-Words Vector
4.Output: Intent Classification
5.Loss Function: CrossEntropyLoss
6.Optimizer: Adam Optimizer

### Future Enhancements

1.Voice-enabled chatbot
2.Multi-language support
3.Database integration
4.Cloud deployment
5.Transformer-based NLP models (BERT)
6.Enterprise system integration

### Author
LAKSHANYA N
212224230136



# Federated Meta-Learning Framework for Personalized AI Assistants Across Edge Devices

## About the Project
This project aims to develop an AI assistant that learns individual user preferences while maintaining data privacy. It uses Federated Learning and Meta-Learning to provide personalized responses across multiple edge devices.

## Objectives
- Develop an intelligent AI assistant.
- Personalize responses according to user preferences.
- Use Meta-Learning (MAML) for fast adaptation to new users.
- Apply Federated Learning to support collaborative model training.
- Reduce the need to share private user data.

## Technologies Used
- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Meta-Learning:** Model-Agnostic Meta-Learning (MAML)
- **Backend:** FastAPI
- **Machine Learning:** Scikit-learn
- **Federated Learning:** Flower
- **Version Control:** Git and GitHub

## Project Modules
1. **AI Assistant:** Handles user messages and generates responses.
2. **Meta-Learning & Personalization:** Learns user preferences and adapts the model.
3. **Federated Learning:** Supports collaborative learning across devices.
4. **Backend:** Provides API endpoints for communication.
5. **Frontend:** Provides the user interface.

## Installation

Clone the repository:

```bash
git clone https://github.com/arjunthorat433-netizen/Federated-Meta-Learning-AI-Assistant.git
cd Federated-Meta-Learning-AI-Assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Backend

```bash
uvicorn backend.main:app --reload
```

Open the API documentation:

http://127.0.0.1:8000/docs

## Current Development Status
The project is under development. The MAML-based personalization module has been implemented with synthetic user preference data and connected to a FastAPI endpoint. Federated integration and real-user preference learning are ongoing development tasks.

## Team Project
**Project Title:** Federated Meta-Learning Framework for Personalized AI Assistants Across Edge Devices

**Repository:** [GitHub Repository](https://github.com/arjunthorat433-netizen/Federated-Meta-Learning-AI-Assistant)


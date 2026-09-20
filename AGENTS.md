# AI Detector Project Rules


## Project Goal

Transform this school AI detector into a professional educational platform for teachers.

The system should:
- analyze student texts
- estimate AI-generated probability
- provide explanations
- keep results understandable for teachers


# Development Rules

Before changing code:

1. Analyze existing implementation
2. Explain planned changes
3. Identify affected files
4. Wait for approval before major rewrites


# Architecture Rules

Do not destroy working functionality.

Prefer:
- modular architecture
- clean separation of components
- documented changes


# Machine Learning Rules

Current ML model is a baseline.

Future improvements should consider:

- Transformer models
- better datasets
- evaluation metrics

Always track:

- accuracy
- precision
- recall
- F1-score
- ROC-AUC


# Dataset Rules

Never evaluate a model on training data.

Check:
- data quality
- duplicates
- class balance


# Frontend Rules

The interface is designed for teachers.

Priorities:

1. Simplicity
2. Clear explanations
3. Modern UI
4. Responsive design


# Documentation

After significant changes update:

- CHANGELOG.md
- PROJECT_STATUS.md
- README.md
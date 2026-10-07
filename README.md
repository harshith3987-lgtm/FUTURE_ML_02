# Support Ticket Classification & Prioritization

A machine learning system for automatically classifying IT support tickets and assigning priority levels using NLP and supervised machine learning.

## Overview

IT support teams receive large numbers of tickets that need to be routed to the appropriate category and prioritized for efficient resolution.

This project develops an NLP-based machine learning pipeline that:

- Classifies support tickets into relevant service categories.
- Assigns High, Medium, or Low priority.
- Compares multiple machine learning algorithms.
- Evaluates model performance using standard classification metrics.
- Provides a reusable prediction pipeline for new support tickets.

## Dataset

The project uses the **IT Service Ticket Classification Dataset** containing **47,837 support tickets** across 8 service categories.

### Categories

- Access
- Administrative rights
- Hardware
- HR Support
- Internal Project
- Miscellaneous
- Purchase
- Storage

The original dataset contains ticket text and category labels.

> **Note:** The original dataset does not contain a priority column. Priority labels were therefore generated using predefined business rules based on ticket content. Consequently, priority-model performance measures agreement with these derived labels rather than independently human-annotated priority.

## Project Workflow

```text
Support Ticket
      ↓
Text Cleaning
      ↓
TF-IDF Feature Extraction
      ↓
Machine Learning Models
      ↓
Category Prediction
      +
Priority Prediction
      ↓
Final Ticket Classification
````

## NLP Preprocessing

The ticket text was prepared using:

* Lowercase conversion
* Punctuation and special-character removal
* Whitespace normalization
* TF-IDF feature extraction
* Unigram and bigram features

## Machine Learning Models

Three classification algorithms were compared:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (Linear SVM)

The best-performing model was selected separately for category and priority classification.

## Category Classification Results

| Model               |   Accuracy |  Precision |     Recall |   F1-Score |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| **Linear SVM**      | **86.30%** | **86.39%** | **86.30%** | **86.30%** |
| Logistic Regression |     85.43% |     86.14% |     85.43% |     85.40% |
| Naive Bayes         |     69.84% |     78.50% |     69.84% |     68.24% |

### Best Model

**Linear SVM**

Linear SVM achieved the highest overall performance for support ticket category classification.

## Priority Classification Results

| Model               |   Accuracy |  Precision |     Recall |   F1-Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| **Linear SVM**      | **97.18%** | **97.21%** | **97.18%** | **97.10%** |
| Logistic Regression |     94.21% |     94.32% |     94.21% |     93.84% |
| Naive Bayes         |     83.93% |     84.87% |     83.93% |     81.65% |

### Best Model

**Linear SVM**

The model achieved strong performance against the derived priority labels. High-priority tickets had lower recall than Medium and Low priority tickets, indicating that some urgent tickets may still be under-predicted.

## Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification reports
* Confusion matrices

The project also includes model comparison visualizations and class-wise performance analysis.

## Example Prediction

### Input

> My laptop is not working and I urgently need it for today's work.

### Prediction

```text
Category: Hardware
Priority: High
```

The saved models can also process multiple new support tickets through the reusable prediction function.

## Project Structure

```text
FUTURE_ML_02/
│
├── data/
│   ├── raw/
│   │   └── all_tickets_processed_improved_v3.csv
│   │
│   └── processed/
│       └── processed_tickets.csv
│
├── models/
│   ├── category_linear_svm.joblib
│   ├── category_tfidf_vectorizer.joblib
│   ├── priority_linear_svm.joblib
│   └── priority_tfidf_vectorizer.joblib
│
├── notebooks/
│   └── support_ticket_classification.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── category_distribution.png
│   │   ├── model_comparison.png
│   │   ├── category_confusion_matrix.png
│   │   ├── priority_model_comparison.png
│   │   └── priority_confusion_matrix.png
│   │
│   ├── predictions/
│   │   ├── sample_predictions.csv
│   │   └── final_prediction_demo.csv
│   │
│   ├── category_model_evaluation.csv
│   ├── category_classification_report.csv
│   ├── priority_model_evaluation.csv
│   ├── priority_classification_report.csv
│   ├── final_model_summary.csv
│   ├── final_results_report.csv
│   └── project_insights.csv
│
├── src/
│
└── README.md
```

## Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **NLTK**
* **Matplotlib**
* **Seaborn**
* **Joblib**
* **Jupyter Notebook**

## Key Insights

* Linear SVM performed best for both classification tasks.
* TF-IDF with unigram and bigram features provided an effective representation of support ticket text.
* Category classification achieved **86.30% accuracy**.
* Priority classification achieved **97.18% accuracy against derived priority labels**.
* Administrative rights was the most challenging category based on recall.
* High-priority tickets showed comparatively lower recall and represent an area for further improvement.
* The prediction pipeline can classify new support tickets using the saved models.

## Limitations

The main limitation of the project is the absence of original priority annotations in the dataset.

The priority labels were generated using predefined keyword-based business rules. Therefore, the priority classification results should not be interpreted as performance against real human-assigned priorities.

Future improvements could include:

* Using a dataset with human-annotated priority labels.
* Improving priority rules using domain-specific information.
* Hyperparameter tuning.
* Class-weight optimization for high-priority tickets.
* Testing transformer-based NLP models.
* Deploying the model as an API or web application.

## Future Scope

The system can be extended into a complete support-ticket automation platform with:

* Automatic ticket routing
* Priority-based queue management
* Real-time ticket classification
* Confidence scoring
* Support-agent dashboards
* REST API deployment
* Integration with help-desk platforms

## Conclusion

This project demonstrates an end-to-end NLP and machine learning workflow for IT support ticket classification and prioritization.

After comparing Logistic Regression, Multinomial Naive Bayes, and Linear SVM, **Linear SVM was selected as the final model for both tasks** based on its superior evaluation performance.

The project provides trained models, TF-IDF vectorizers, evaluation reports, visualizations, prediction examples, and a reusable prediction pipeline suitable for further development and deployment.



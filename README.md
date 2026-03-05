![Image](https://github.com/user-attachments/assets/e4a1d970-cae3-4a86-b4b1-a0f9fdc00edf)
# Welcome to the Ramadan League Data Science Competition!

We are excited to have you participate in this challenge. The competition is divided into four problems, each designed to test your skills in different aspects of data science.

## Learning First

This competition is designed first and foremost as a learning experience.
Please avoid using AI tools, especially coding agents, to solve the problems.
Take time to deeply research each topic, understand the methods you apply, and build your solutions through your own reasoning and experimentation.

## Competition Structure

### Problems:
1. [**1.The Hidden Leak** (20%)](./1.The%20Hidden%20Leak/)  
   Short description: Identify and prevent data leakage in a time-series forecasting pipeline, then propose a leak-free evaluation protocol.  
   Difficulty: Medium  
   Submission format: `report.md` or `report.txt` (required), optional `analysis.ipynb` for pseudo-code/backtesting sketch.  
2. [**2. Sentiment Sleuth** (30%)](./2.%20Sentiment%20Sleuth/)
   Short description: Build a multiclass tweet sentiment classifier (negative, neutral, positive) from labeled text data.  
   Difficulty: Medium  
   Submission format: `solution.ipynb` (required) and optional `train.py` for reproducible training script.  
3. [**3. Buggy Logistic Regression** (20%)](./3.%20Buggy%20Logistic%20Regression/)  
   Short description: Debug and correct a broken machine learning implementation, then verify expected model behavior.  
   Difficulty: Easy-Medium  
   Submission format: fixed `problem.py` (required) and a short `notes.md` explaining identified bugs and fixes.  
4. [**4. Taxi Time Challenge** (30%)](./4.%20Taxi%20Time%20Challenge/)
   Short description: Predict taxi trip duration using temporal and geospatial trip features with strong feature engineering.  
   Difficulty: Hard  
   Submission format: `solution.ipynb` (required), optional `train.py` + `inference.py` for clean pipeline separation.  


### Scoring System

Your score will be calculated as follows:

```math
\text{score} = \text{weight}_{\text{INE}} \times \sum (\text{weight}_{\text{problem}} \times \frac{\text{score}_{\text{problem}}}{100})
```

Where:
- **$\text{score}_{\text{problem}}$**: The score you earn for each problem, ranging from 0 to 100.
- **$\text{weight}_{\text{problem}}$**: The weight assigned to each problem as listed above.
- **$\text{weight}_{\text{INE}}$**: A factor based on your INE year:
  - **INE3**: 0.7
  - **INE2**: 0.8
  - **INE1**: 1
  - For teams with members from multiple years, the $\text{weight}_{\text{INE}}$ will be the average.

## Code Documentation Requirements

Please ensure that every line of code is well documented and explained. Failure to do so may result in a loss of points.

## GitHub Submission Guide (Beginner Friendly)

Follow these steps to submit your work from your own GitHub account.

1. *If you don't have git:*Install Git (one-time setup)
   - Download and install Git: https://git-scm.com/downloads
   - After installation, open a terminal and verify:

```bash
git --version
```

2. Create a local folder on your computer
   - Choose a location (for example, Desktop) and create a folder for the competition.


3. Create your own repository on GitHub
   - Go to GitHub and create a new empty repository (example: `ramadan-league-submission-teamXX`).
   - Do not initialize it with a README if possible.

4. Clone this competition repository to your computer
   - Run this inside the local folder you just created:
   - Run:

```bash
git clone https://github.com/<competition-owner>/<competition-repo>.git
cd <competition-repo>
```

5. Connect your local copy to your personal repository
   - Replace `<your-username>` and `<your-repo>`:

```bash
git remote rename origin upstream
git remote add origin https://github.com/<your-username>/<your-repo>.git
git remote -v
```

6. Add your solutions
   - Work inside each problem folder and add your required files (`.md`, `.txt`, `.ipynb`, `.py`).
   - Save all changes.

7. Commit your work
   - Run:

```bash
git add .
git commit -m "Add solutions for Ramadan League problems"
```

8. Push to your GitHub repository
   - If your main branch is `main`:

```bash
git push -u origin main
```

9. Submit your repository link
   - Open your personal GitHub repository in browser.
   - Copy the URL and submit it in the Google Form below.

## Submit Your Repository

Please refer to the following form to submit your repositories:
[google form](https://docs.google.com/forms/d/e/1FAIpQLSceSTzbDgOv_c-mDuGWFyJOco8_GXw8w4IJPfDxCBK4zb9TsA/viewform?usp=header)

<br>
<br>

---

<p align="center"> <strong>Good luck, and may the best data scientists win! 👾</strong> </p>

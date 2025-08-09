import pandas as pd
import os
from collections import defaultdict

def recommend_courses(data):
    try:
        subjects_path = os.path.join('data', 'work1.csv')
        df_subjects = pd.read_csv(subjects_path, encoding='latin1')
        df_subjects.columns = ['SUBJECT NAME', 'SUBJECT CATEGORY']
        df_subjects['SUBJECT NAME'] = df_subjects['SUBJECT NAME'].str.strip()
        df_subjects['SUBJECT CATEGORY'] = df_subjects['SUBJECT CATEGORY'].str.strip().str.upper()

        electives_path = os.path.join('data', 'sheet3.csv')
        df_electives = pd.read_csv(electives_path, encoding='latin1')
        df_electives.columns = ['ELECTIVE COURSES', 'CATEGORIES']
        df_electives['ELECTIVE COURSES'] = df_electives['ELECTIVE COURSES'].fillna(method='ffill').str.strip()
        df_electives['CATEGORIES'] = df_electives['CATEGORIES'].str.replace('•', '').str.strip().str.upper()

        category_scores = defaultdict(list)
        for subject_key, mark_str in data.items():
            try:
                mark = float(mark_str)
                subject_clean = str(subject_key).strip().upper()
                
                for _, row in df_subjects.iterrows():
                    if row['SUBJECT NAME'].upper() in subject_clean:
                        category_scores[row['SUBJECT CATEGORY']].append(mark)
                        break
            except ValueError:
                continue

        if not category_scores:
            return ["No matching subjects found in our database"]

        avg_scores = {cat: sum(marks)/len(marks) for cat, marks in category_scores.items()}
        top_categories = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        elective_scores = defaultdict(float)
        current_course = ""
        
        for _, row in df_electives.iterrows():
            if pd.notna(row['ELECTIVE COURSES']):
                current_course = row['ELECTIVE COURSES']
            
            if pd.notna(row['CATEGORIES']) and current_course:
                for category, score in top_categories:
                    if category in row['CATEGORIES'].upper():
                        elective_scores[current_course] += score

        if not elective_scores:
            return ["No specific electives found for your top categories"]
        
        top_electives = sorted(elective_scores.items(), key=lambda x: x[1], reverse=True)[:4]
        return [elective[0] for elective in top_electives]

    except Exception as e:
        print(f"Error: {str(e)}")
        return ["Error processing your request"]

import os
import pandas as pd
import numpy as np
import rootutils

# Initialize rootutils to load env variables from .env
root_dir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.database_service import DatabaseService

def main():
    db = DatabaseService()
    if not db.connect():
        raise RuntimeError("Failed to connect to the database.")
        
    try:
        print("Extracting survey responses and answers from database...")
        survey_responses = db.extract_from_database("survey_response")
        answers = db.extract_from_database("answer")
    finally:
        db.disconnect()
        
    print(f"Total survey responses found: {len(survey_responses)}")
    print(f"Total answers found: {len(answers)}")
    
    # Clean timestamps and extract dates
    survey_responses["timestamp"] = pd.to_datetime(survey_responses["timestamp"]).astype("datetime64[ns]")
    survey_responses["date"] = survey_responses["timestamp"].dt.date
    
    # Define cohort filters (matching CohortBuilder behavior)
    drop_users = [1, 2, 3, 10, 21, 22, 43, 44]
    
    # Analyze both unfiltered and filtered cohorts
    for name, df_sr in [("Unfiltered (All Users)", survey_responses), 
                        ("Filtered (CohortBuilder Users, >= 2025-09-01)", 
                         survey_responses[~survey_responses["app_user_id"].isin(drop_users) & (survey_responses["timestamp"] >= pd.Timestamp("2025-09-01"))])]:
        
        print(f"\n=========================================")
        print(f" Analysis: {name}")
        print(f"=========================================")
        
        # 1. Check duplicate survey responses per user per day
        sr_grouped = df_sr.groupby(["app_user_id", "date"]).size().reset_index(name="survey_count")
        counts_dist = sr_grouped["survey_count"].value_counts().sort_index()
        
        total_user_days = len(sr_grouped)
        dup_user_days = len(sr_grouped[sr_grouped["survey_count"] > 1])
        pct_dup_days = (dup_user_days / total_user_days) * 100 if total_user_days > 0 else 0.0
        
        print(f"Total user-days with at least one survey: {total_user_days}")
        print(f"Distribution of survey submissions per user per day:")
        for count, freq in counts_dist.items():
            print(f"  {count} survey(s)/day: {freq} times ({freq/total_user_days*100:.2f}%)")
            
        print(f"\nUser-days with MULTIPLE surveys: {dup_user_days} ({pct_dup_days:.2f}%)")
        
        # 2. Check duplicate question answers per user per day
        df_answers_merged = pd.merge(
            answers,
            df_sr[["id", "app_user_id", "date", "timestamp"]],
            left_on="survey_response_id",
            right_on="id"
        )
        
        ans_grouped = df_answers_merged.groupby(["app_user_id", "date", "question_id"]).size().reset_index(name="answer_count")
        ans_counts_dist = ans_grouped["answer_count"].value_counts().sort_index()
        
        total_question_days = len(ans_grouped)
        dup_question_days = len(ans_grouped[ans_grouped["answer_count"] > 1])
        pct_dup_q_days = (dup_question_days / total_question_days) * 100 if total_question_days > 0 else 0.0
        
        print(f"\nDistribution of answers per question per user per day:")
        for count, freq in ans_counts_dist.items():
            print(f"  {count} answer(s)/question/day: {freq} times ({freq/total_question_days*100:.2f}%)")
            
        print(f"Instances of a question answered MULTIPLE times on the same day: {dup_question_days} ({pct_dup_q_days:.2f}%)")
        
        # 3. Identify which question IDs are most frequently duplicated
        if dup_question_days > 0:
            dup_questions = ans_grouped[ans_grouped["answer_count"] > 1]
            dup_q_counts = dup_questions["question_id"].value_counts()
            print("\nMost frequently duplicated question IDs (same day):")
            for q_id, count in dup_q_counts.items():
                print(f"  Question ID {q_id}: duplicated on {count} user-days")
                
            # 4. Check if duplicate answers on the same day are identical or different
            different_answers_count = 0
            for _, row in dup_questions.iterrows():
                sub_df = df_answers_merged[
                    (df_answers_merged["app_user_id"] == row["app_user_id"]) &
                    (df_answers_merged["date"] == row["date"]) &
                    (df_answers_merged["question_id"] == row["question_id"])
                ]
                # Compare unique answers
                unique_ans = sub_df["answer"].unique()
                if len(unique_ans) > 1:
                    different_answers_count += 1
            
            pct_diff_answers = (different_answers_count / dup_question_days) * 100
            print(f"\nOf the {dup_question_days} duplicate question-days:")
            print(f"  Same answers given: {dup_question_days - different_answers_count} ({(dup_question_days - different_answers_count)/dup_question_days*100:.2f}%)")
            print(f"  Different answers given: {different_answers_count} ({pct_diff_answers:.2f}%)")
            
            # Print a few examples of differing answers
            if different_answers_count > 0:
                print("\nExamples of differing answers on the same day:")
                printed = 0
                for _, row in dup_questions.iterrows():
                    sub_df = df_answers_merged[
                        (df_answers_merged["app_user_id"] == row["app_user_id"]) &
                        (df_answers_merged["date"] == row["date"]) &
                        (df_answers_merged["question_id"] == row["question_id"])
                    ].sort_values("timestamp")
                    
                    unique_ans = sub_df["answer"].unique()
                    if len(unique_ans) > 1:
                        print(f"  User {row['app_user_id']}, Date {row['date']}, Question {row['question_id']}:")
                        for _, r in sub_df.iterrows():
                            print(f"    Timestamp: {r['timestamp']} -> Answer: '{r['answer']}' (Survey ID: {r['survey_response_id']})")
                        printed += 1
                        if printed >= 3:
                            break

if __name__ == "__main__":
    main()

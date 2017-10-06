import numpy as np
import pandas as pd
from audit_parser import getGPA, audit_info
from GenEdLookup import lookupGenEd
from lsa_recommender import filter_available_classes
import os
import json
from Course_class import Course
import time

a = time.time()

#schedule_path = 'data\\schedules\\joe_schedule.ics'
#audit_path = '0_you_michael_academic_audit.txt'
#audit = audit_info(audit_path)
#available_classes = filter_available_classes(schedule_path)
with open('data\\fce_json.json') as file:
    fce_dict = json.load(file)

def better_fce():
    data = {} #dict with class number key and tuple of avg values
    path = 'data\\FCE\\'
    columns = 'Semester,Year,Instructor,Dept,Course ID,Course Name,Section,Type,Responses,Enrollment,Resp. Rate %,1: Hrs Per Week 9,2: Interest in student learning,3: Explain course requirements,4: Clear learning goals,5: Instructor provides Feedback to students,6: Importance of subject,7: Explains subject matter,8: Show respect for students,9: Overall teaching,10: Overall course'
    columns = columns.split(',')
    keep_columns = ['Course ID','1: Hrs Per Week 9', '10: Overall course']
    files = [f for f in os.listdir(path)]
    for file in files:
        #print (file)
        df = pd.read_csv(path + file, names=columns)
        #print (df.shape)
        df = df[keep_columns]
        for course in df['Course ID'].unique():
            temp = df.copy()
            temp = temp[temp['Course ID'] == course]
            avg = temp.mean().values
            if len(str(course)) == 5:
                dict_key = str(course)[:2] + '-' + str(course)[2:]
            elif len(str(course)) == 4:
                dict_key = '0'+str(course)[0] + '-' + str(course)[1:]
            else:
                dict_key = course
            data[dict_key] = list(avg[1:])

    with open('data\\fce_json.json', 'w') as file:
        json.dump(data,file)
    return data

def preference_score(available_courses, audit, pref_difficulty, pref_rating, must_gened):
    # Scores range from very negative (classes to definitely avoid) to 190
    gpa = getGPA(audit)

    course_scores_df = pd.DataFrame(available_courses,columns=['Course'])
    course_scores_df['Number'] = course_scores_df['Course'].apply(lambda x: x.number)
    course_scores_df['Data'] = course_scores_df['Number'].apply \
                        (lambda x: fce_dict[x] if x in fce_dict else np.nan)
    course_scores_df['Hours'] = course_scores_df['Data'].apply(get_hours)
    course_scores_df['Ratings'] = course_scores_df['Data'].apply(get_ratings)
    course_scores_df = course_scores_df.drop(['Course','Data'], axis=1)
    course_scores_df['Predicted'] = course_scores_df['Hours'] - gpa + 3

    scores = []
    for index,row in course_scores_df.iterrows():
        score = get_score(row['Hours'],row['Ratings'], row['Predicted'],
                        pref_difficulty, pref_rating, must_gened, row['Number'])
        scores.append(score)

    course_scores_df['Score'] = scores
    sorted_df = course_scores_df.sort_values(by=['Score'], ascending=False)
    sorted_df = sorted_df.drop_duplicates()
    return list(sorted_df['Number'].values)

def top_preferred_courses(course_rankings):
    course_short = course_rankings[:100]
    return course_short

def get_hours(l):
    try:
        return l[0]
    except:
        return 8.5

def get_ratings(l):
    try:
        return l[1]
    except:
        return 3.5

def get_score(diff,rating,predicted_hours,pref_difficulty,pref_rating,must_gened,number):
    score = 0
    # If user wants a gen ed and the class is not a gen ed, automatically has lowest score
    if must_gened == True and lookupGenEd(number, "dietrich") == 0:
        score = -9999

    else:
        # The more the user cares about rating, the more rating will affect score
        # Rating can add up to 90 and subtract up to 150
        try:
            score = (rating - 3.5) * (
                pref_rating - 1) * 15
        except:
            score = 0

        # User picks how difficult/time-consuming they want the class to be
        # For 1 & 5, constant on one side and quadratic on the other side
        # For 2,3,4, quadratic function centered around a number
        # Score can be up to 100. If a class is out of range of difficulty,
        # it will decrease the score by a lot
        try:
            test = int(predicted_hours)
            ph = predicted_hours
        except:
            ph = 9

        if pref_difficulty == 1:
            if ph < 4:
                score += 100
            else:
                score += int(100 - ((4 - ph) ** 2 * 15))
        elif pref_difficulty == 2:
            score += int(100 - (7 - ph) ** 2 * 15)
        elif pref_difficulty == 3:
            score += int(100 - (9 - ph) ** 2 * 15)
        elif pref_difficulty == 4:
            score += int(100 - (11 - ph) ** 2 * 15)
        elif pref_difficulty == 5:
            if ph > 14:
                score += 100
            else:
                score += int(100 - ((14 - ph) ** 2 * 15))

    return score

#course_rankings = preference_score(available_classes,audit,4.5,5,False)
#top = top_preferred_courses(course_rankings)

#print (top)
#print ('\n', time.time()-a)

def decision_tree_master(available_courses, audit, pref_difficulty, pref_rating, must_gened):
    course_rankings = preference_score(available_courses, audit, pref_difficulty, pref_rating, must_gened)

    decision_tree_rankings = top_preferred_courses(course_rankings)

    return decision_tree_rankings
# This program parses the academic audit text to find
# which courses a student has taken in the past
# and what grades they got in those classes

import re
import numpy as np
import pandas as pd

def audit_info(fpath, fullPath=False, string=False):
    courses_dict = {'Course Number': [], 'Grade': []}
    if (not fullPath) and (not string):
        text_lines = open("data\\audits\\"+fpath, "r").read().splitlines()
    elif string:
        text_lines = fpath.splitlines()
    else:
        text_lines = open(fpath, "r").read().splitlines()
    past_courses = np.zeros((64, 2), dtype='int')
    grades_dict = {"A": 4, "B": 3, "C": 2, "D": 1, "R": 0}
    i = 0

    for line in text_lines:

        # Parse for all the course numbers xx-xxx
        course_numbers = re.findall("\d\d\-\d\d\d", line)

        if course_numbers != []:
            # Course Number
            past_courses[i, 0] = int(course_numbers[0][0:2]) * 1000 + int(
                course_numbers[0][3:6])

            # Course Grade (A=4, B=3, C=2, D=1, R=0, else=-1)
            grade = line[52:53]
            try:
                grade_num = grades_dict[grade]
            except:
                grade_num = -1
            past_courses[i, 1] = grade_num

            # Print all past course numbers and their respective grades
            #print("Course:", past_courses[i, 0], "Grade:", past_courses[i, 1])
            courses_dict['Course Number'].append(past_courses[i, 0])
            courses_dict['Grade'].append(past_courses[i, 1])
            i = i + 1

    audit = pd.DataFrame(courses_dict)
    return audit

def getGPA(audit):
    grades = audit["Grade"]
    total = 0
    num = 0
    for grade in grades:
        if grade > -1:
            total += grade
            num += 1
    if num == 0:
        gpa = 3
    else:
        gpa = total / num
    return gpa
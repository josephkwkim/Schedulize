from ics import Calendar
import numpy as np
import pandas as pd

def obtainSchedule(fileName):

    file = open(fileName, "r").read()
    c = Calendar(file)
    courses = np.array(c.events, dtype="str")
    numCourses = len(courses)

    course_data = dict()

    course_data["NAME"] = [None] * numCourses
    course_data["SECTION"] = [None] * numCourses
    course_data["NUMBER"] = [None] * numCourses
    course_data["START"] = [None] * numCourses
    course_data["END"] = [None] * numCourses
    course_data["DAYS"] = [None] * numCourses

    count = 0
    for course in courses:
        print (course)
        cDeets = course.splitlines()
        for deet in cDeets:
            if "SUMMARY" in deet:
                index = deet.find(" ::")
                name = deet[len("SUMMARY:"):index]
                specs = deet[index + len(" :: "):].split(" ")
                num = specs[0]
                section = specs[1]
                course_data["NAME"][count] = name
                course_data["SECTION"][count] = section
                course_data["NUMBER"][count] = num
            elif "DTSTART" in deet:
                shortDeet = deet[len("DTSTART:"):]
                index = shortDeet.find("T")
                start = shortDeet
                course_data["START"][count] = start
            elif "DTEND" in deet:
                shortDeet = deet[len("DTEND:"):]
                index = shortDeet.find("T")
                end = shortDeet
                course_data["END"][count] = end
            elif "RRULE" in deet:
                index = deet.find("BYDAY=")
                days = deet[index + len("BYDAYS"):]
                course_data["DAYS"][count] = days.split(',')
        count += 1

    #print(course_data["NAME"])
    #print(course_data["SECTION"])
    #print(course_data["NUMBER"])
    #print(course_data["START"])
    #print(course_data["END"])
    #print(course_data["DAYS"])
    return pd.DataFrame(course_data)

#fileName = "F17_schedule.ics"
#obtainSchedule(fileName)

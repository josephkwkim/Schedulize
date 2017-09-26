import numpy as np
import pandas as pd
import os
from audit_parser import audit_info

train_data = dict()
train_data["Audit"] = []
train_data["Schedule"] = []
X_audits = dict()
X_schedules = dict()
X_audits_grades = dict()

auditPath = "data/audits/"
schedulePath = "data/schedules/"

def loadAudits(path):
    print("Loading Audits...")
    def pullAudit(path):
        if (os.path.isdir(path) == False):
            if path[-3:] == "txt":
                print(path)
                audit = audit_info(path, fullPath=True)
                train_data["Audit"].append(audit)
        else:
            # RECURSION
            for fileName in os.listdir(path):
                pullAudit(path + "/" + fileName)
    pullAudit(path)
    print("Done!")
    print()

def inputData(auditList, scheList):
    individual = 0

    def inputAudits(auditList):
        nonlocal individual
        for i in range(len(auditList)):
            df = auditList[i]
            courseNumbers = df["Course Number"]
            courseGrades = df["Grade"]
            X_audits[individual] = []
            X_audits_grades[individual] = []
            for j in range(len(courseNumbers)):
                if courseGrades[j] >= 0:
                    str_course_num = str(courseNumbers[j])
                    if len(str_course_num) == 4:
                        str_course_num = "0" + str_course_num
                    X_audits[individual].append(str_course_num)
                    X_audits_grades[individual].append((str_course_num,
                                                       str(courseGrades[j])))
            individual += 1

    inputAudits(auditList)

# Code Referenced from ex.8 of Machine Learning by Stanford University on Coursera
def buildRecommender():
    print("Building Recommender System...")
    print()

    loadAudits(auditPath)

    inputData(train_data["Audit"], train_data["Schedule"])

    X_data = []
    X_courses = []
    X_grades = []

    for key, values in X_audits_grades.items():
        X_data.append(values)

    for course_grade in X_data:
        course, grade = zip(*course_grade)
        X_courses.append(course)
        X_grades.append(grade)

    X_courses = np.array(X_courses)
    X_grades = np.array(X_grades)

    allClasses = []
    for student in X_courses:
        for courses in student:
            allClasses.append(courses)

    allClasses = list(set(allClasses))

    n, m = X_courses.shape[0], len(allClasses)  # Num Users, Num Classes

    dfR = pd.DataFrame(0, index=np.arange(len(X_courses)), columns=allClasses)
    for row in range(n):
        takenClasses = X_courses[row]
        for col in dfR:
            courseNum = str(col)
            if courseNum in takenClasses:
                dfR.loc[dfR.index[row], col] = 1

    dfY = pd.DataFrame(0, index=np.arange(len(X_courses)), columns=allClasses)
    for row in range(n):
        takenClasses, earnedGrades = X_courses[row], X_grades[row]
        for col in dfY:
            courseNum = str(col)
            if courseNum in takenClasses:
                index = list(takenClasses).index(courseNum)
                dfY.loc[dfY.index[row], col] = int(earnedGrades[index])

    features = 20
    Y = np.array(dfY).T  # Matrix with Grades
    X = np.random.rand(m, features)
    Theta = np.random.rand(n, features)
    R = np.array(dfR).T  # Binary Matrix denoting Classes Taken

    """
    print("n Students:", n)
    print("m Classes:", m)
    print("Y:", Y.shape)
    print("R:", R.shape)
    print("X:", X.shape)
    print("Theta:", Theta.shape)
    print()
    """

    # SKIPPED REGULARIZATION - ADD LATER
    def costFunction(X, Y, Theta, R):
        M = np.power((np.dot(X, Theta.T)) - Y, 2)
        J = (1/2) * np.sum(np.multiply(R, M))
        return J

    def gradientFunction(X, Y, Theta, R):
        grad_all = ((np.dot(X, Theta.T)) - Y)
        grad_R = np.multiply(R, grad_all)

        X_grad = np.zeros(X.shape)
        Theta_grad = np.zeros(Theta.shape)

        for k in range(X.shape[1]):
            X_grad[:, k] = np.dot(grad_R, Theta[:, k])

        for l in range(Theta.shape[1]):
            Theta_grad[:, l] = np.dot(grad_R.T, X[:, l])

        return X_grad, Theta_grad

    print("Optimizing via Gradient Descent...")
    iterations = 250
    learning_rate = 0.01
    for i in range(iterations):
        cost = costFunction(X, Y, Theta, R)
        X_grad, Theta_grad = gradientFunction(X, Y, Theta, R)
        X -= learning_rate * X_grad
        Theta -= learning_rate * Theta_grad
        if (i + 1) == iterations:
            print("Iteration", i + 1)
            print("Cost:", cost)
    print("Done!")
    print()

    return X, Theta, allClasses

def makePrediction(model, user):
    X, Theta = model
    p = np.dot(X, Theta.T)
    predictions = p[:, user]
    return sorted(predictions, reverse=True)

# List Predictions & Compile Departmental Scores
def compileDepartScores(courses, pList):
    departDict = dict()
    departCounter = dict()
    for h in range(len(courses)):
        course = courses[h]
        depart = str(course)[0:2]
        if depart not in departDict:
            departDict[depart] = []
            departCounter[depart] = []

    for i in range(len(pList)):
        course, prediction = courses[i], pList[i]
        # print("Predicted Rating for", course, "is", prediction)
        depart = str(course)[0:2]
        departDict[depart].append(prediction)
        departCounter[depart].append(1)

    for key, values in departDict.items():
        departDict[key] = np.sum(values) / len(values)

    # return [(k, departDict[k]) for k in sorted(departDict, key=departDict.get, reverse=True)]
    return departDict

"""
recSystem = buildRecommender()
model, courses = (recSystem[0], recSystem[1]), recSystem[2]
user = 0  # Index for Current User - Denote with 0_last_first_academic_audit.txt
pList = makePrediction(model, user)
dScores = compileDepartScores(courses, pList)

for item in dScores:
    print("Department " + str(item[0]) + ": " + str(item[1]))
"""

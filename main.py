# main schedulizer app

from imports import *

if __name__ == '__main__':
    print("Welcome to Schedulize!")

    startTime = time()

    # Load the Data
    with open('data\\undergrad_course_info.json') as file:
        data = json.load(file)
        data = json.loads(data)

    # Recommender System
    train_data = dict()
    train_data["Audit"] = []
    train_data["Schedule"] = []
    X_audits = dict()
    X_schedules = dict()
    X_audits_grades = dict()

    auditPathCalvin = "data/audits/"
    schedulePathCalvin = "data/schedules/"
    auditPath = 'sample_academic_audit.txt'
    schedulePath = 'data\\schedules\\sample_schedule.ics'
    audit = audit_info(auditPath)
    available_classes = filter_available_classes(schedulePath)

    recSystem = buildRecommender()
    model, courses = (recSystem[0], recSystem[1]), recSystem[2]
    user = 0  # Index for Current User - Denote with 0_last_first_academic_audit.txt
    pList = makePrediction(model, user)
    dScores = compileDepartScores(courses, pList)
    print("Recommender Scores:")
    print(dScores)
    print()

    # Latent Sentiment Analysis
    lsa_predictions = export_to_master(100, schedulePath, auditPath)
    print("LSA Predictions:")
    print(lsa_predictions)
    print()

    # Decision Tree
    pref_difficulty = 4
    pref_rating = 5
    gen_ed = False

    course_rankings = preference_score(available_classes, audit, pref_difficulty, pref_rating, gen_ed)
    decision_tree_rankings = top_preferred_courses(course_rankings)
    print("Decision Tree Predictions:")
    print(decision_tree_rankings)
    print()

    course_score_d = dict()

    lsa_predictions_r = list(reversed(lsa_predictions))
    for i in range(len(lsa_predictions_r)):
        base_score = i / 100  # reverse ranking score, last item gets zero weight
        department = lsa_predictions_r[i][0:2]
        depart_score = dScores.get(department, 0.01)  # changeable coefficient for unmatched departments
        cum_score = base_score * depart_score
        course_score_d[lsa_predictions_r[i]] = cum_score

    decision_tree_rankings_r = list(reversed(decision_tree_rankings))
    for i in range(len(decision_tree_rankings_r)):
        base_score = i / 100  # reverse ranking score, last item gets zero weight
        department = decision_tree_rankings_r[i][0:2]
        depart_score = dScores.get(department, 0.01)  # changeable coefficient for unmatched departments
        cum_score = base_score * depart_score
        # account for intersection items
        existing_score = course_score_d.get(decision_tree_rankings_r[i], 0)
        course_score_d[decision_tree_rankings_r[i]] = (existing_score + cum_score) / 2

    joe = set(lsa_predictions)
    wilson = set(decision_tree_rankings)
    grand_list = joe.intersection(wilson)

    masturbater = []
    for course in grand_list:
        course_score = course_score_d[course]
        masturbater.append((course, course_score))

    import operator
    sorted_recommendations = sorted(masturbater, key=operator.itemgetter(1), reverse=True)  # sorted list of tuples

    assert(len(grand_list) == len(sorted_recommendations))

    """
    this list seems to be pretty precise at recommending courses, but it may not be the most accurate in terms of
    diversity. recommendations are concentrated, thus do not provide interesting courses that alter from a student's
    course history. good or bad?

    CALVIN IS A GOD

    Also im gonna penetrate the recommendations into a dataframe with information
    """
    print("Sorted Recommendations:")
    #loading data into lists to send to dataframe
    course_numbers = []
    scores = []
    for j in sorted_recommendations:
        #print(str(j[0]) + ": " + str(j[1]))
        course_numbers.append(j[0])
        scores.append(j[1])
    print()

    #Construct Final Dataframe
    final_printable = pd.DataFrame()
    final_printable['Number'] = course_numbers
    final_printable['Name'] = final_printable['Number'].apply(lambda x: data[x]['name'])
    final_printable['Units'] = final_printable['Number'].apply(lambda x: data[x]['units'])
    final_printable['Score'] = scores
    print(final_printable)
    print()

    print("Time Elapsed:", time() - startTime, "sec")
    print()

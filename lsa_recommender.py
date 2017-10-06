import json
from ics import Calendar
import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pickle
import cmu_course_api
import sys

#External Files
from Course_class import Course
from audit_parser import audit_info
a = time.time()
#Load Data
with open('data\\undergrad_course_info.json') as file:
    data = json.load(file)
    data = json.loads(data)
with open('data\\lecture_recitation.json') as file:
    recs = json.load(file)
with open('data\\lsa_vectors.json') as file:
    lsa_map = json.load(file)
with open('data\\processed_classes.p','rb') as file:
    semester_classes = pickle.load(file)

fpath = 'data\\schedules\\joe_schedule.ics'
audit_path = 'kim_joe_academic_audit.txt'

class Schedule(object):
    def __init__(self,num,name,beg,end,days):
        self.name = name
        self.number = num
        self.start = beg.time()
        self.end = end.time()
        self.days = days
    def __str__(self):
        return (self.num)

class Recitation(object):
    def __init__(self,num,lec_name,start,end,days):
        self.num = num
        self.lec_name = lec_name
        self.start = start
        self.end = end
        self.days = days

def filter_undergrad_courses(data):
    exceptions = ['10-601']
    undergrad = {}

    #Penetrate Grad Courses
    for course in data.keys():
        assert (isinstance(course,str))
        assert (len(course) == 6)
        course_num = course[3]
        if int(course_num) < 5:
            undergrad[course] = data[course]
        elif course in exceptions:
            undergrad[course] = data[course]
    return json.dumps(undergrad)

def parse_current_classes(lsa_map):
    classes = []
    for course in data.keys():
        for lecture in data[course]['lectures']:
            lec_name = lecture['name']
            for time in lecture['times']:
                days = time['days'] if isinstance(time['days'],list) else []
                if days == []: break
                if time['location'] == 'Doha, Qatar':
                    break
                beg = pd.to_datetime(time['begin'])
                end = pd.to_datetime(time['end'])
                lsa = lsa_map[course]
                classes.append(Course(course,beg,end,data[course]['name'],
                days,lsa,lec_name,data[course]['desc'],data[course]['prereqs']))
    return classes

def parse_calendar(fpath):
    with open(fpath) as file:
        calendar = Calendar(file)
    courses = []
    for event in calendar.events:
        days = create_days_from_event(event)
        name = create_name_from_name(event.name)
        num = create_num_from_name(event.name)
        courses.append(Schedule(num,name,event.begin.datetime,
                                event.end.datetime,days))
    return courses

def parse_audit_to_text(fpath):
    audit = pd.DataFrame(audit_info(fpath)['Course Number'])
    audit['Course Number'] = audit['Course Number'].apply(str)
    audit['Course Number'] = audit['Course Number'].apply \
                                        (lambda x: x[:2] + '-' + x[2:])
    audit['Course Number'] = audit['Course Number'].apply \
                                    (lambda x: x if x in data else np.nan)

    return audit['Course Number'].dropna().values

def create_num_from_name(string):
    num = string.split()[-2]
    num = num[:2] + '-' + num[2:]
    return num

def create_name_from_name(string):
    splitted = string.split()[:-3]
    result = ''
    for word in splitted:
        result+= word + ' '
    return result.strip()

def create_days_from_event(event):
    days_dict = {'SU':0,'MO':1,'TU':2,'WE':3,'TH':4,'FR':5,'SA':6}
    string = str(event)
    good_part = string.splitlines()[1]
    better_part = good_part.split(';')[-1]
    best_part = better_part[6:].split(',')
    days = []
    for day in best_part:
        days.append(days_dict[day])
    return days

def filter_available_classes(fpath):
    your_classes = parse_calendar(fpath)
    dont_take = set()
    for potential in semester_classes: #0(n)
        for day in potential.days: #O(3)
            for course in your_classes: #O(8)
                if day in course.days: #O(3)
                    if potential.start <= course.end and potential.end >= course.start:
                        dont_take.add(potential)

    available_classes = list(set(semester_classes).difference(dont_take))
    available_classes = filter_available_by_recitation(available_classes)
    return available_classes

def filter_available_by_recitation(available_classes):
    times = get_recitation_times(available_classes)
    dont_take = []

    for recitation in times:
        for day in recitation.days:
            for obligation in your_classes:
                if day in obligation.days:
                    if (recitation.start <= obligation.end) and \
                       (recitation.end >= obligation.start):
                        dont_take.append(recitation)

    for recitation in dont_take:
        for course in available_classes:
            if course.recitation_belongs(recitation):
                available_classes.remove(course)

    return available_classes

def get_recitation_times(available_classes):
    times = []
    for course in available_classes:
        recitations = course.recitations()  # list of recitations corresponding to that lecture
        for recitation in recitations:
            # parse to a 3-tuple of (start,end,[days])
            recitation_list = data[course.number]['sections']
            for possibility in recitation_list:
                if possibility['name'] == recitation:
                    for meeting in possibility['times']:
                        start = pd.to_datetime(meeting['begin']).time()
                        end = pd.to_datetime(meeting['end']).time()
                        days = meeting['days']
                        times.append(Recitation(course.number, course.lec_name,
                                                start, end, days))
    return times

def search(courseNum):
    assert (len(courseNum) == 6)
    for course in semester_classes:
        if course.number == courseNum:
            return course
    return None

def print_all():
    for course in available_classes:
        print('Name: {}'.format(course.name))
        print('Number: {}'.format(course.number))
        print('Starts: {}'.format(course.start))
        print('Ends: {}'.format(course.end))
        print('Days: {}'.format(course.days))
        print()

def latent_sentiment(data):
    '''
    :return: dictionary mapping class names to a 100-d vector of sentiments
    '''
    course_list = data.keys()
    description_list = []
    #parse descriptions to list
    for course in course_list:
        desc = data[course]['desc']
        if isinstance(desc,str):
            description_list.append(desc)
        else:
            description_list.append(' ')
    #create term-frequency matrix and apply tf-idf
    count_vec = CountVectorizer(stop_words='english')
    trans = count_vec.fit_transform(description_list)
    tfidf = TfidfTransformer()
    better = tfidf.fit_transform(trans)

    #reduce dimensions to 100
    svd = TruncatedSVD(n_components=100)
    reduced = svd.fit_transform(better)
    assert (len(reduced) == len(data.keys()))

    #return a dictionary mapping course number to lsa
    lsa_dict = {}
    for num,vec in zip(data.keys(),reduced):
        lsa_dict[num] = list(vec)
    return lsa_dict

def cosine_similarity(v1,v2):
    '''
    :param v1:
    :param v2:
    :return:
    '''
    a_dot_b = np.dot(v1,v2)
    norm_a_norm_b = (np.sqrt(np.dot(v1,v1))*np.sqrt(np.dot(v2,v2)))
    similarity = a_dot_b/norm_a_norm_b
    #assert(-1 <= similarity <= 1)
    return similarity

def search_similarity(c1,c2):
    return cosine_similarity(lsa_map[c1],lsa_map[c2])

def recommend_classes_avg(num,schedulepath,auditpath,available=True):
    if available:
        class_list = filter_available_classes(schedulepath)
    else:
        class_list = semester_classes
    youve_taken = parse_audit_to_text(auditpath)
    tot = np.zeros((len(youve_taken),100))
    for i,c in enumerate(youve_taken):
        vec = lsa_map[c]
        tot[i] = vec
    avg = tot.mean(axis=0)
    avg = list(avg)
    df = pd.DataFrame(class_list,columns=['b'])
    df['Number'] = df['b'].apply(lambda x: x.number)
    df['Name'] = df['b'].apply(lambda x: x.name)
    df['Similarity'] = df['b'].apply(lambda x: x.similarity(avg))
    df['Remove'] = df['Number'].apply(lambda x: x in youve_taken)
    df = df[df['Remove'] == False]
    df = df.drop(['Remove','b'],axis=1)
    df = df.drop_duplicates()
    most_similar = df.sort_values(by=['Similarity'],ascending=False)
    most_similar = most_similar.reset_index().drop('index',axis=1)
    return most_similar.head(num)

def repickle(lsa_map):
    fpath = 'data\\processed_classes.p'
    semester_classes = parse_current_classes(lsa_map)
    with open(fpath,'wb') as file:
        pickle.dump(semester_classes,file)
    print ('done')

def export_to_master(num,schedulepath,auditpath):
    exported = recommend_classes_avg(num,schedulepath,auditpath)
    exported = list(exported['Number'].values)
    return exported

def get_new_semester_data(semester):
    assert(len(semester) == 1)
    assert(isinstance(semester,str))
    keep_going = input('Keep Going?')
    if keep_going == 'No' or keep_going == 'no':
        sys.exit()
    #get semester data
    data = cmu_course_api.get_course_data(semester)['courses']
    data = filter_undergrad_courses(data)
    with open('data\\undergrad_course_info.json','w') as file:
        json.dump(data)
    #get lsa for current semester
    lsa_map2 = latent_sentiment(data)
    for key in lsa_map.keys(): #gets previous semester's classes
        if key not in lsa_map2:
            lsa_map2[key] = lsa_map[key]
    print ('lsa data for {} classes'.format(len(lsa_map2)))
    with open('data\\lsa_vectors{}.json'.format(semester),'w') as file:
        json.dump(lsa_map2,file)
    #pickle reformatted classes
    repickle(lsa_map2)
    print('done!')


#available_classes = filter_available_classes(fpath)
#youve_taken = parse_audit_to_text(audit_path)

#print(recommend_classes_avg(30,fpath,audit_path,available=False))
#print (time.time() - a)






import numpy as np

class Course(object):
    def __init__(self, num, beg, end, name, days, lsa, lec_name,
                 desc=None, prereq=None):
        self.number = num
        self.start = beg.time()
        self.end = end.time()
        self.name = name
        self.days = days
        if lec_name == 'Lec':
            self.lec_name = 'Lec 1'
        else:
            self.lec_name = lec_name
        self.description = desc
        self.prereqs = prereq
        self.department = int(self.number[:2])
        self.lsa = lsa

    def __str__(self):
        return self.number
    def __eq__(self, other):
        return (self.number == other.number) and (self.lec_name == other.lec_name)
    def __hash__(self):
        return hash(self.number+str(self.start)+str(self.lec_name))
    def similarity(self,course2):
        '''
        :param course2: {class number (str), course instance,lsa_list}
        :return: similarity between the two classes
        '''
        if isinstance(course2,str):
            course2 = search(course2).lsa
        elif isinstance(course2,list):
            course2 = course2
        else:
            course2 = course2.lsa
        return cosine_similarity(self.lsa,course2)
    def recitations(self):
        '''
        returns a list of recitations corresponding to each lecture
        :return: list
        '''
        try:
            if self.number in recs:
                recitations = recs[self.number][self.lec_name]
            else:
                recitations = []
        except:
            recitations = []
        return recitations
    def recitation_belongs(self,recitation):
        '''
        returns true if recitation belongs to that class, false otherwise
        :param recitation: Recitation object
        :return: Boolean
        '''
        if recitation.num == self.number and recitation.lec_name == self.lec_name:
            return True
        return False

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
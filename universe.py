#Denis Paperno, 2018-2020
#Code for generating interpreted languages with 'personal relations' interpretation

from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
import torch

'''List of reserved characters

at least at the level of logical form the basic elements of the language are encoded 
as characters. Some characters are reserved for relation names or grammatical words'''
reserved_chars=['p','f','e','c','t','o','s']

def newUniverse(n):
	"""generate a list of n names - characters outside of the reserved list"""
	z=[]
	i=ord('a')
	#add entity names to the list starting from 'a'
	while len(z)<n:
		ch=chr(i)
		if ch not in reserved_chars:
			z.append(ch)
		i+=1
	return z

class InterpretedLanguage:
    """InterpretedLanguage class of sets of interpreted strings

    Initialized with rel_num relations (up to 4) and 2*num_pairs entities.
    For convenience of forming symmetric relations, there is always an even number
    of entities."""
    def __init__(self, rel_num,num_pairs):
        if rel_num+3>len(reserved_chars): raise(ValueError)
        self.rel=reserved_chars[:rel_num]
        names=newUniverse(num_pairs*2)
        #friend relation is initialized randomly as a symmetric relation
        random.shuffle(names)
        self.friend={}
        for i in range(num_pairs):
            self.friend[names[2*i]]=names[2*i+1]
            self.friend[names[2*i+1]]=names[2*i]
        random.shuffle(names)
        #enemy relation is initialized randomly as a symmetric relation
        self.enemy={}
        for i in range(num_pairs):
            self.enemy[names[2*i]]=names[2*i+1]
            self.enemy[names[2*i+1]]=names[2*i]
        #parent relation is initialized randomly as a cycle involving all entities
        random.shuffle(names)
        self.parent={}
        for i in range(len(names)):
             self.parent[names[i]]=names[i-1]
        #child relation is initialized as the inverse of the parent relation
        self.child={}
        for i in range(len(names)):
             self.child[names[i-1]]=names[i]
        self.names=names
        self.reserved_chars = reserved_chars

    def examples(self,i):
        '''returns all logical forms of complexity (length) i 
        
        A logical form is a string of relation chars followed by an entity name char'''
        if i<=1: ex=self.names
        else: ex=[x+y for x in self.rel for y in self.examples(i-1)]
        return ex

    def interpret(self,s):
        """interpretation function returns the entity a logical form s describes"""
        assert type(s) is str
        assert len(s)>0
        if len(s)==1 and s in self.names: return s
        elif len(s)>1:
            if s[0]=='f': return self.friend[self.interpret(s[1:])]
            elif s[0]=='e': return self.enemy[self.interpret(s[1:])]
            elif s[0]=='p': return self.parent[self.interpret(s[1:])]
            elif s[0]=='c': return self.child[self.interpret(s[1:])]
            else: print(s); raise KeyError

    def express(self,s,b):
        """spellout function translating logical form s into a string of word identifiers
        
        s: logical form
        b: branching parameter"""
        if len(s)==1 and s in self.names: return s
        elif len(s)>1:
            r=s[0]
            if r in self.rel:
                o=random.choice(b)
                if o=='r': return 't'+r+'o'+self.express(s[1:],b) 
                elif o=='l': return self.express(s[1:],b)+'s'+r
                else: raise(KeyError)
            else: raise(KeyError)
        else: raise(KeyError)

    def lines(self,data,b):
        """Interprets and expresses logical forms from a list
        
        data: list of logical forms
        b: branching parameter
        Returns a mapping from entities to lists of strings expressing them, based on a list
        of logical forms data"""
        l=defaultdict(list)
        for d in data:
            l[self.interpret(d)].append(self.express(d,b))
        return l

    def memorization_data(self):
        """returns all logical forms of complexity 1 and 2"""
        return self.examples(1)+self.examples(2)

    def allexamples(self, b,complexity=2,min_complexity=1):
        """returns a list of (string,referent) pairs for all logical forms 
        
        Parameters define breanching and complexity range:
        b: branching parameter
        min_complexity: minimal logical form complexity (defaults to a single name)
        complexity: max number of relation elements in a logical form"""
        z=[]
        sample = []
        for c in range(min_complexity,complexity+1): sample+=self.examples(c)
        for e in sample:
            line=self.express(e,b)
            category=self.interpret(e)
            z.append((line,category))
        return z

    def randomexamples(self, k, b,complexity=3,min_complexity=1):
        "returns k random examples of complexity up to a given value"
        z=[]
        sample = []
        for c in range(complexity): sample+=self.examples(c)
        for i in range(min_complexity,k):
            e=random.choice(sample)
            line=self.express(e,b)
            category=self.interpret(e)
            z.append((line,category))
        return z

    def dataset(self, b):
        rel_example = self.allexamples(b, complexity =2)
        rec_example = self.allexamples(b, complexity =3)
        rec_example = rec_example[len(rel_example):]

        # create equal amount of recursive and non recursive relationships
        # create test and train set
        sample = random.sample(rec_example, len(rel_example)) #random sample van de 3 mensen rela

        rel_ex, ind_ex = zip(*rel_example)
        rel_rec, ind_rec = zip(*sample)

        X_train1, X_test1, y_train1, y_test1 = train_test_split(rel_ex, ind_ex, test_size=0.2)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(rel_rec, ind_rec, test_size=0.2)
        X_train_val = X_train1 + X_train2
        X_test = X_test1 + X_test2
        y_train_val = y_train1 + y_train2
        y_test = y_test1 +y_test2
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def model_input(self, vec, max_size):
        """ Changes a list 'vec' to a torch.LongTensor with paddings
            until length 'max_size' 
        """
        
        # Creates a list of all the characters possible in the strings
        # including padding value 0
        chars = list(set(self.reserved_chars + self.names + ['0'])) 

        # Lookup dicts for char to index and index to char
        char2idx = {o:i for i,o in enumerate(sorted(chars))}
        idx2char = {i:o for i,o in enumerate(sorted(chars))}

        # Padd the input vector to the specified maximum size
        vec = [vec[rel].zfill(max_size) for rel in range(len(vec))]
        
        # Create a list with all the right indices and return as longtensor
        results = [ [] for _ in range(len(vec))]
        for rel in range(len(vec)):
            results[rel] = [char2idx[ch] for ch in vec[rel]]

        return torch.LongTensor(results)
import sys
import math
import numpy as np

#==========================================================================================================

def naive_bayes(file):
    representatives = read_file(file)
    print(nfoldn_cross_validation(representatives,naive_bayes_classifier))

def k_nearest_neighbor(file):
    representatives = read_file(file)
    print(nfoldn_cross_validation(representatives,k_nearest_neighbor_classifier))

def neural_network(file):
    representatives = read_file(file)
    print(nfoldn_cross_validation(representatives,neural_network_classifier))
    return

def decision_tree(file):
    representatives = read_file(file)

    #make training and tuning sets
    tuning_set = make_tuning_set(representatives)
    training_set = make_training_set(representatives)

    #make, prune, then print decision tree
    decision_tree = make_decision_tree(training_set,set())
    prune_tree(decision_tree, tuning_set)
    print_decision_tree(decision_tree,0)

    #check validity of Tree
    print(nfoldn_cross_validation(representatives,decision_tree_classifier))

#use to classify an item with a decision tree made from data from file. 
def classify_with_decision_tree(file, item_file):
    representatives = read_file(file)
    item = read_file(item_file)[0]

    tuning_set = make_tuning_set(representatives)
    training_set = make_training_set(representatives)

    decision_tree = make_decision_tree(training_set,set())
    prune_tree(decision_tree, tuning_set)
    return decision_tree.classify(item)

#use to classify an item with Naive Bayes with information from data from file. 
def classify_with_naive_bayes(file, item_file):
    representatives = read_file(file)
    item = read_file(item_file)[0]

    nb = Naive_Bayes(representatives)
    return nb.classify(item)

#use to classify an item with k-nearest neighbor with information from data from file. 
def classify_with_knn(file, item_file):
    representatives = read_file(file)
    item = read_file(item_file)[0]

    knn = K_Nearest_Neighbor(representatives)
    return knn.classify(item)

def classify_with_neural_network(file, item_file):
    representatives = read_file(file)
    item = read_file(item_file)[0]

    nn = Neural_Network_Classifier(representatives)
    return nn.classify(nn.rep_to_input(item))

#==========================================================================================================
#                                          helper methods
#==========================================================================================================
def read_file(file):
    #returns a list of representatives
    data = open(file, "r")
    current_line = data.readline()
    representatives = []
    #while there are more representatives to read
    while len(current_line) > 2:
        values = current_line.split("\t")
        representatives.append(Representative(values[0],values[1],values[2][:-1]))
        current_line = data.readline()
    return representatives

#makes tuning set
def make_tuning_set(representatives):
    tuning_set = []
    for i in range(len(representatives)):
        if i % 4 == 0:
            tuning_set.append(representatives[i])
    return tuning_set

#makes training set
def make_training_set(representatives):
    training_set = []
    for i in range(len(representatives)):
        if not (i % 4 == 0):
            training_set.append(representatives[i])
    return training_set

#makes a decision tree
def make_decision_tree(representatives,splits_done, parents_majority = 'D'):

    #Check if there are any representatives left to check
    if len(representatives) == 0: return Decision_Tree(representatives, majority = parents_majority)

    #if decision tree has split on all possible splits
    if len(splits_done) >= len(representatives[0].votes): return Decision_Tree(representatives,parents_majority)

    #first check if set is uniformly labeled
    uniformly_labeled = True
    for i in representatives:
        if i.party != representatives[0].party:
            uniformly_labeled = False
            break
    if uniformly_labeled:
        return Decision_Tree(representatives,parents_majority)

    #Calculate entropy of entire training set
    entire_set_entropy = calculate_entropy(representatives)

    #find the best split out of the possible splits
    best_split = -1
    best_information_gain = -99999999
    possible_splits = list(filter(lambda split: split not in splits_done, list(range(len(representatives[0].votes)))))
    for i in possible_splits:
        #calculate the weighted sum of the possible entropy values
        weighted_sum_split_entropy = 0
        for j in "+-.":
            split_set = list(filter(lambda rep: rep.votes[i] == j, representatives))
            weighted_sum_split_entropy += (len(split_set) / len(representatives)) * calculate_entropy(split_set)

        #Calculate the information gained from the split
        information_gain = entire_set_entropy - weighted_sum_split_entropy

        #if information gained from this split is better than all other options, set it as the best split
        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_split = i

    #Choose single best feature (if tie, chose earlier one)
    branches = []

    #make list of splits that have been made
    new_past_splits = list(filter(lambda split: split not in possible_splits, list(range(len(representatives[0].votes)))))
    new_past_splits.append(best_split)

    #get current majority of representatives
    current_majority = get_majority_vote(representatives)
    if current_majority == None: current_majority = parents_majority

    #make branches of tree
    for i in "+-.":
        split_set = list(filter(lambda rep: rep.votes[best_split] == i, representatives))
        branch = make_decision_tree(split_set,new_past_splits,current_majority)
        branches.append((i,branch))

    #makes a tree with the branches created above
    issue_letter = "ABCDEFGHIJKLMNOPQRSTUVWQYZ"[best_split] # gets the letter of the issue being classified. 
    tree = Decision_Tree(representatives,current_majority,is_leaf = False, issue = issue_letter, branches = branches)
    return tree

#prints decision tree
def print_decision_tree(decision_tree, depth):
    # if leaf, print the party
    if decision_tree.is_leaf :
        print(decision_tree.majority) #, end="")
        """
        print(" ", end="")
        for i in decision_tree.representatives:
            print(i.party,end="")
        print()
        """
    #otherwise print out branches
    else:
        print("Issue " + decision_tree.issue + ":")
        for branch in decision_tree.branches:
            print("   "*depth + branch[0] + " ", end = "")
            print_decision_tree(branch[1], depth+1)

#Calculates entropy for a given group of representatives
def calculate_entropy(representatives):
    entropy = 0

    #lim x -> 0 of log x * x = 0
    if len(representatives) == 0:
        return 0
    
    #for both parties, find prob_party * log ( prob_party)
    for i in "DR":
        party = list(filter(lambda rep: rep.party == i, representatives))
        prob_party = len(party) / len(representatives)
        if(prob_party == 0):
            continue
        entropy += prob_party * math.log2(prob_party)
    entropy *= -1
    return entropy

#Gets the majority vote of a group of representatives
def get_majority_vote(representatives):
        D = len(list(filter(lambda rep: rep.party == 'D',representatives)))
        R = len(list(filter(lambda rep: rep.party == 'R',representatives)))
        if D == R:
            return None
        return 'D' if D > R else 'R'
    
#Uses reduced error pruning to prune branches that don't improve performance on tuning set
def prune_tree(decision_tree,tuning_set):
    current_performance = calculate_decision_tree_performance_on_set(decision_tree,tuning_set)
    while True:
        nodes = get_all_non_leaf_nodes(decision_tree)

        best_performance = -99999999
        best_cut = None
        for node in nodes:
            performance = get_performance_if_turn_to_leaf(decision_tree,node,tuning_set)
            if performance > best_performance:
                best_cut = node
                best_performance = performance
        if best_performance >= current_performance:
            turn_node_to_leaf(best_cut)
            current_performance = best_performance
        else:
            break

#returns a list of all non-leaf nodes in a decision tree
def get_all_non_leaf_nodes(decision_tree):
    if decision_tree.is_leaf:
        return []
    nodes = [decision_tree]
    for branch in decision_tree.branches:
        new_nodes = get_all_non_leaf_nodes(branch[1])
        nodes.extend(new_nodes)
    return nodes

#returns the performance of a decision tree on the tuning set if the node is turned to a leaf
def get_performance_if_turn_to_leaf(decision_tree, node, tuning_set):
    node.is_leaf = True
    performance = calculate_decision_tree_performance_on_set(decision_tree,tuning_set)
    
    node.is_leaf = False
    return performance

#turns a node to a leaf
def turn_node_to_leaf(node):
    node.is_leaf = True

#returns how well a given decision tree performes on the tuning set
def calculate_decision_tree_performance_on_set(decision_tree,tuning_set):
    sum = 0
    for i in tuning_set:
        sum += calculate_decision_tree_performance_on_individual(decision_tree, i)
    return sum / len(tuning_set)

#checks if the decision returns the correct value for a given item, returns 1 if it does, 0 if not
def calculate_decision_tree_performance_on_individual(decision_tree,item):
    while decision_tree.is_leaf != True:
        issue_index = issue_letter_to_index(decision_tree.issue)
        for i in decision_tree.branches:
            if i[0] == item.votes[issue_index]:
                return calculate_decision_tree_performance_on_individual(i[1],item)
    if item.party == decision_tree.majority:
        return 1
    else:
        return 0

#turns an issue letter to the index of that vote
def issue_letter_to_index(issue_letter):
    return "ABCDEFGHIJKLMNOPQRSTUVWQYZ".index(issue_letter)

#makes a decision tree from a list of representatives, 
# and checks if the decision returns the correct value for a given item, returns 1 if it does, 0 if not
def decision_tree_classifier(representatives, item):
    tuning_set = make_tuning_set(representatives)
    training_set = make_training_set(representatives)

    decision_tree = make_decision_tree(training_set,set(),'D')
    prune_tree(decision_tree, tuning_set)

    return calculate_decision_tree_performance_on_individual(decision_tree,item)

#makes a Naive Bayes classifier from a list of representatives, 
# and checks if the decision returns the correct value for a given item, returns 1 if it does, 0 if not
def naive_bayes_classifier(representatives,item):
    classifier = Naive_Bayes(representatives)
    if classifier.classify(item) == item.party:
        return 1
    else:
        return 0

#makes a k-nearest neighbor from a list of representatives, 
# and checks if the decision returns the correct value for a given item, returns 1 if it does, 0 if not
def k_nearest_neighbor_classifier(representatives,item):
    classifier = K_Nearest_Neighbor(representatives)
    if classifier.classify(item) == item.party:
        return 1
    else:
        return 0

def neural_network_classifier(representatives,item):
    classifier = Neural_Network_Classifier(representatives)
    if classifier.classify(item) == item.party:
        return 1
    else:
        return 0
#takes a list of representatives, and a classifier that returns 1 if it classifies an item right, and 0 if not
#returns the classier performance using nfoldn cross validation
def nfoldn_cross_validation(representatives, classifier):
    num_correct = 0
    for i in range(len(representatives)):
        item = representatives.pop(0)
        num_correct += classifier(representatives,item)
        representatives.append(item)
    return num_correct / len(representatives)

#==========================================================================================================
#                                              Classes
#==========================================================================================================
class Representative:
    def __init__(self,name, party, votes):
        self.name = name
        self.party = party
        self.votes = votes
    def get_name(self): return self.name
    def get_party(self): return self.party
    def get_votes(self): return self.votes
    def get_vote(self, issue): return self.votes[issue]
    def __str__(self): return self.name + "\t" + self.party + "\t" + self.votes

class Decision_Tree:
    def __init__(self,representatives, majority, is_leaf = True, issue = None, branches = None):
        self.is_leaf = is_leaf
        self.issue = issue
        self.representatives = representatives
        self.branches = branches
        self.majority = majority
        self.majority = self.get_majority_vote()
    #gets the majority party of the democrats that this leaf is tasked with classifying
    def get_majority_vote(self):
        D = len(list(filter(lambda rep: rep.party == 'D',self.representatives)))
        R = len(list(filter(lambda rep: rep.party == 'R',self.representatives)))
        if D == R:
            return self.majority
        return 'D' if D > R else 'R'
    
    def classify(self, item):
        if self.is_leaf == True:
            return self.majority
        issue_index = issue_letter_to_index(self.issue)
        for i in self.branches:
            if i[0] == item.votes[issue_index]:
                return i[1].classify(item)
class Naive_Bayes:
    def __init__(self,representatives):
        self.representatives = representatives
        self.probs = self.generate_probs(representatives)

    #Makes a dict where key is formatted (vote index, vote, party)
    # value returned is the probability of that vote for that vote index given the party
    def generate_probs(self,representatives):
        num_in_category = []
        pseudo_count = 0
        for party in "RD":
            reps_in_party = list(filter(lambda reps: reps.party == party,representatives))
            for vote_id in range(len(reps_in_party[0].votes)):
                plus = 0
                minus = 0
                dot = 0

                #gets how each member of the party voted on this issue
                for rep in reps_in_party:
                    vote = rep.votes[vote_id]

                    if vote == '+': plus += 1
                    if vote == '-': minus += 1
                    if vote == '.': dot += 1
                
                num_in_category.append((vote_id,'+',party,plus))
                num_in_category.append((vote_id,'-',party,minus))
                num_in_category.append((vote_id,'.',party,dot))

                #if there were any votes where no member of the party voted for that option, use pseudocounts
                if(plus == 0 or minus == 0 or dot == 0):
                    pseudo_count = 0.1
        probs = dict()
        party_totals = {
            'D' : len(list(filter(lambda reps: reps.party == 'D',representatives))),
            'R' : len(list(filter(lambda reps: reps.party == 'R',representatives)))
        }

        #makes dictionary
        for category in num_in_category:
            probs[(category[0],category[1],category[2])] = (category[3] + pseudo_count) / (party_totals[category[2]] + pseudo_count)
        probs['D'] = (party_totals['D'] + pseudo_count) / (len(representatives) + pseudo_count)
        probs['R'] = (party_totals['R'] + pseudo_count) / (len(representatives) + pseudo_count)

        return probs

    #classifies an item based on a naive bayes algorithm
    def classify(self,item):
        republican_score = 0
        democrat_score = 0

        #uses naive bayes algorithm to get how likely item is based on party
        for i in range(len(item.votes)):
            republican_score += math.log2(self.probs[(i,item.votes[i],'R')])
            democrat_score += math.log2(self.probs[(i,item.votes[i],'D')])

        democrat_score += math.log2(self.probs['D'])
        republican_score += math.log2(self.probs['R'])

        if democrat_score >= republican_score:
            return 'D'
        else:
            return 'R'

class K_Nearest_Neighbor:
    def __init__(self,representatives):
        self.representatives = representatives
    
    #checks how far appart the voting history of 2 representatives was
    def distance(self,rep1,rep2):
        sum = 0
        for vote in range(len(rep1.votes)):
            if rep1.get_vote(vote) == rep2.get_vote(vote):
                sum += 0
                continue
            if rep1.get_vote(vote) == '.' or rep2.get_vote(vote) == '.':
                sum += 1
                continue
            sum += 2
        return sum
    
    #classifies an item based on it's closest neighbor
    def classify(self,item):
        best_rep = None
        best_sim = -9999999
        for rep in self.representatives:
            sim = 1/(1+self.distance(rep,item))
            if sim > best_sim:
                best_sim = sim
                best_rep = rep
        return best_rep.party

class Neural_Network_Classifier:
    def __init__(self, representatives):
        self.net = Neural_Network(len(representatives[0].votes),2,8,2)
        self.train(representatives)

    def classify(self,item):
        output = self.net.classify(self.rep_to_input(item))
        if output[0] > output[1]:
            return 'D'
        else:
            return 'R'
    #Turns a representative into an input for the Neural Net
    def rep_to_input(self,rep):
        input = []
        for i in rep.votes:
            if i == '+':
                input.append(1.0)
            if i == '-':
                input.append(-1.0)
            if i == '.':
                input.append(0.0)
        return input

    #Turns a representative into a data point that the neural network can understand
    def rep_to_datum(self,rep):
        if rep.party == 'D':
            return (self.rep_to_input(rep),0)
        else:
            return (self.rep_to_input(rep),1)
    def train(self,reps):
        data = list(map(self.rep_to_datum,reps))
        self.net.train(data)

class Neural_Network:
    def __init__(self, num_inputs, num_hidden_layers, nodes_per_layer, num_outputs, weights = None, biases = None):
        self.num_outputs = num_outputs
        #initialize weights
        if weights == None:
            self.weights = self.generate_weights(nodes_per_layer,num_inputs,num_outputs,num_hidden_layers)
        else:
            self.weights = weights
        #initialize biases/thresh-holds
        if biases == None:
            self.biases = self.generate_biases(nodes_per_layer,num_hidden_layers,num_outputs)
        else:
            self.biases = biases

    def generate_biases(self,nodes_per_layer,num_hidden_layers,num_outputs):
        biases = [[]] # first layer is input layer, so left blank
        for i in range(num_hidden_layers):
            biases.append(np.random.uniform(low=-1.0,high=1.0,size=(nodes_per_layer)))
        biases.append(np.random.uniform(low=-1.0,high=1.0,size=(num_outputs)))
        return biases

    def generate_weights(self,nodes_per_layer,num_inputs,num_outputs,num_hidden_layers):
        weights = [[]] # first layer is input layer, so left blank
        #generate weights from first to second nodes
        weights.append(np.random.uniform(low=-1.0,high=1.0,size=(nodes_per_layer,num_inputs)))
        #generate weights for hidden layer
        for k in range(num_hidden_layers - 1):
            weights.append(np.random.uniform(low = -1.0,high = 1.0, size=(nodes_per_layer,nodes_per_layer)))

        #generate weights ending in output layer
        weights.append(np.random.uniform(low=-1.0,high=1.0,size=(num_outputs,nodes_per_layer)))
        return weights

    def print_network(self):
        for i in range(len(self.weights)):
            print("weights: " + str(self.weights[i]))
            print("biases: " + str(self.biases[i]))

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    #classifies an item, returns the output vector of the neural network
    def classify(self,input):
        last_layer = input
        for i in range(1,len(self.weights)):
            activations = np.dot(self.weights[i],last_layer)
            activations = np.add(activations,self.biases[i])
            last_layer = list(map(self.sigmoid, activations))
        return last_layer

    #returns the activations of each layer on a given input
    def get_activations_on_input(self,input):
        last_layer = input
        all_layers_activation = [input]
        for i in range(1,len(self.weights)):
            activations = np.dot(self.weights[i],last_layer)
            activations = np.add(activations,self.biases[i])
            last_layer = list(map(self.sigmoid, activations))
            all_layers_activation.append(last_layer)
        return all_layers_activation

    def get_deltas_last_layer(self,output,correct_output):
        deltas = []
        for i in range(len(output)):
            deltas.append(output[i]*(1-output[i])*(correct_output[i] - output[i]))
        return deltas

    def get_deltas_hidden_layer(self,layer,next_layer_deltas,activations):
        weights = self.weights[layer + 1]
        translated_weights = np.rot90(np.fliplr(weights))
        weighted_error = np.dot(translated_weights,next_layer_deltas)
        one_minus_activations = list(map(lambda x : 1 - x, activations))
        temp = np.dot(weighted_error,one_minus_activations)
        temp = np.dot(temp, activations)
        return temp

    def update_weights(self,layer, learning_rate, deltas, activations):
        update = np.outer(deltas,activations)
        update = np.multiply(update,learning_rate)
        self.weights[layer] = np.add(self.weights[layer],update)

    def update_biases(self,layer, learning_rate, deltas):
        update = np.multiply(deltas,learning_rate)
        self.biases[layer] = np.add(self.biases[layer],update)

    def train_on_example(self,input,correct_output,learning_rate):
        activations = self.get_activations_on_input(input)
        #update deltas for last layer
        deltas = self.get_deltas_last_layer(activations[-1],correct_output)
        self.update_weights(-1,learning_rate, deltas, activations[-2])
        self.update_biases(-1,learning_rate,deltas)
    
        current_layer = -2
        #for all weights
        for i in range(len(self.weights) - 2):
            deltas = self.get_deltas_hidden_layer(current_layer,deltas,activations[current_layer])
            self.update_weights(current_layer,learning_rate,deltas, activations[current_layer-1])
            self.update_biases(current_layer,learning_rate,deltas)
            current_layer -= 1

    # data should come in form: (input, correct_node_output)
    def train(self,data):
        try:
            for datum in data:
                output_vector = [0] * self.num_outputs
                output_vector[datum[1]] = 1.0
                self.train_on_example(datum[0],output_vector,(1)) # realistically 1 is probably a really bad training factor, but it works better for voting.tsv for some reason 
        except TypeError:
            raise TypeError("Your data should come in the form of a list of (input vector, correct_node_output)")
        
#==========================================================================================================
#interprets command line
def parse_command_line(command_line):
    if  len(command_line) == 0:
        print("Please provide a file!")
        print_help()
    #if user wants help
    if "-h" in command_line:
        print_help()
        return

    #Default to Decision Tree
    if len(command_line) == 1:
        decision_tree(command_line[0])

    if len(command_line) == 2:
        #decision tree
        if "-dt" in command_line:
            command_line.remove("-dt")
            decision_tree(command_line[0])

        #Naive bayes
        if "-nb" in command_line:
            command_line.remove("-nb")
            naive_bayes(command_line[0])

        #K-Nearest Neighbor
        if "-knn" in command_line:
            command_line.remove("-knn")
            k_nearest_neighbor(command_line[0])

        if "-nn" in command_line:
            command_line.remove("-nn")
            neural_network(command_line[0])
        else:
            print("invalid input, please use one of the following commands instead")
            print_help()
    if len(command_line) == 3:
        if "-dt" in command_line:
            command_line.remove("-dt")
            print(classify_with_decision_tree(command_line[0],command_line[1]))

        if "-nb" in command_line:
            command_line.remove("-nb")
            print(classify_with_naive_bayes(command_line[0],command_line[1]))

        if "-knn" in command_line:
            command_line.remove("-knn")
            print(classify_with_knn(command_line[0],command_line[1]))

        if "-nn" in command_line:
            command_line.remove("-nn")
            print(classify_with_neural_network(command_line[0],command_line[1]))
        else:
            print("invalid input, please use one of the following commands instead")
            print_help()

def print_help(output = sys.stderr):

    print("\npossible commands you could try:", file=output)
    print("\t -h \t prints this help menu", file=output)
    print("\t -dt \t makes a decision tree. \t\t\t\t\tUsage: python3 tree-inducer.py FILE -dt", file=output)
    print("\t -nb \t uses a naive-bayes classifier to classify an element. \t\tUsage: python3 tree-inducer.py FILE -nb ", file=output)
    print("\t -knn \t uses a k-nearest neighbor classifier to classify an element. \tUsage: python3 tree-inducer.py FILE -knn", file=output)
    print("\t -nn \t uses a neural network classifier to classify an element. \tUsage: python3 tree-inducer.py FILE -nn", file=output)
    print("\nIf you would like to classify an item with one of the given functions, use the above commands followed by a file\ncontaining the item you'd like to classify in the first position.\n(please use normal formatting, the classifier will ignore the party listed in the file)\n")

#==========================================================================================================#=====================================================
if __name__ == "__main__":
        parse_command_line(sys.argv[1:])


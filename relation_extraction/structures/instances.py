import sys
import os

def dijkstra(adj_matrix, source):
    distance = [+sys.maxint] * len(adj_matrix)  # Unknown distance function from source to v
    previous = [-1] * len(adj_matrix)  # Previous node in optimal path from source
    distance[source] = 0  # Distance from source to source
    unreached = range(len(adj_matrix))  # All nodes in the graph are unoptimized -
    # print distance
    # print adj_matrix
    # print source

    while len(unreached) > 0:  # The main loop
        u = distance.index(min(distance))  # Get the node closest to the source
        # print unreached
        # print u
        if distance[u] == +sys.maxint:
            break  # all remaining vertices are inaccessible
        else:
            unreached.remove(u)
            for v in unreached:  # where v has not yet been removed from Q.
                if adj_matrix[u][v] != '':
                    alt = distance[u] + 1
                    if alt < distance[v]:  # Relax (u,v,a)
                        distance[v] = alt
                        previous[v] = u
            distance[u] = +sys.maxint  # Set the distance to u to inf so that it get's ignored in the next iteration
    return previous




class Instance(object):
    def __init__(self,sentence, start, end, label):
        self.sentence = sentence
        self.start = start
        self.end = end
        self.label = label
        self.dependency_path = self.build_dependency_path()
        self.type_dependency_path = self.build_type_dependency_path()

    def build_dependency_path(self):
        source_token_no = self.start
        target_token_no = self.end
        previous = dijkstra(self.sentence.dependency_matrix, source_token_no)
        if previous[target_token_no] != -1:
            prev = previous[target_token_no]
            path = [prev, target_token_no]
            while prev != source_token_no:
                prev = previous[prev]
                path.insert(0,prev)
            return path

    def build_type_dependency_path(self):
        first_token = self.sentence.get_token(self.start)
        final_token = self.sentence.get_token(self.end)
        type_path = [first_token.get_word(), first_token.get_lemma(), first_token.get_ner(), first_token.get_pos()]
        for i in range(len(self.dependency_path)-1):
            dep_start = self.dependency_path[i]
            dep_end = self.dependency_path[i+1]
            dep_type = self.sentence.get_dependency_type(dep_start,dep_end)
            next_token = self.sentence.get_token(dep_end)
            current_token = self.sentence.get_token(dep_start)
            features = [dep_type, next_token.get_word()]
            if i == 0 and next_token != final_token:
                features = [dep_type, next_token.get_word(), next_token.get_lemma(), next_token.get_pos()]
            elif i ==0 and next_token == final_token:
                features = [dep_type, next_token.get_word(), next_token.get_lemma(), next_token.get_ner(), next_token.get_pos()]

            if i == len(self.dependency_path)-2 and current_token != first_token:
                features = [current_token.get_lemma(), current_token.get_pos(), dep_type, next_token.get_word(), next_token.get_lemma(), next_token.get_ner(), next_token.get_pos()]
            elif i == len(self.dependency_path)-2 and current_token == first_token:
                features = [dep_type, next_token.get_word(), next_token.get_lemma(), next_token.get_ner(), next_token.get_pos()]
            type_path = type_path + features

        return type_path

    def set_label(self,label):
        '''Sets the label of the candidate sentence (positive/negative)'''
        self.label = label


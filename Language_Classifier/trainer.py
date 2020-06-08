import csv
import sys
import pandas as pd
from csv import reader
import numpy as np
import math
import pickle

from Dtree import DTree
from Leaf import Leaf

eps = np.finfo(float).eps

_author = 'RJ'

'''
This is a code that classifies given 15 word sentences as English or Dutch.
The code uses Decision Tree and Ada Boost approach to classify the sentences
@author: Ravikiran Jois Yedur Prabhakar
'''


# Args for train: train train.dat hype_out dt/ada
# Args for predict: predict hype_out test.dat
# Decision Tree: 7/10
# Ada: 8/10


def get_language_features():
    """
    The method returns the list of attributes that is used to classify the sentences into English or Dutch
    :return: list of attributes
    """

    definite_articles = ['def_art_de', 'def_art_het']
    vowels_together = ['vowels_aa', 'vowels_ee', 'vowels_oo']
    avg_len_of_word = ['is_length_9']
    common_words = ['common_dutch_van', 'common_dutch_dat', 'common_dutch_en', 'common_dutch_niet']
    you_and_i_in_dutch = ['you_in_dutch_je', 'i_in_dutch_ik']
    ij_count = ['contains_ij']
    common_in_en = ['common_english_words']
    word_z = ['starts_z_dutch']
    word_oe_cht = ['contains_oe_cht']

    list_of_attributes = definite_articles + vowels_together + avg_len_of_word + common_words + \
                         you_and_i_in_dutch + ij_count + common_in_en + word_z + word_oe_cht
    return list_of_attributes


def has_de(words):
    """
    Check if a word in sentence is 'de'
    :param words:
    :return: Boolean value
    """
    for word in words:
        if 'de' == word.lower():
            return True
    return False


def has_het(words):
    """
    Check if a word in sentence is 'het'
    :param words:
    :return: Boolean value
    """
    for word in words:
        if 'het' == word.lower():
            return True
    return False


def has_aa(words):
    """
    Check if the sentence has 'aa' in it
    :param words:
    :return: Boolean value
    """
    for word in words:
        if 'aa' in word.lower():
            return True
    return False


def has_ee(words):
    """
    Check if the sentence has 'ee' in it
    :param words:
    :return: Boolean value
    """
    for word in words:
        if 'ee' in word.lower():
            return True
    return False


def has_oo(words):
    """
    Check if the sentence has 'oo' in it
    :param words:
    :return: Boolean value
    """
    for word in words:
        if 'oo' in word.lower():
            return True
    return False


def is_length_7(words):
    """
    Check if the average length of the sentence has word length greater than 7
    :param words:
    :return: Boolean value
    """
    sum_word_length = 0
    total_length = len(words)
    for word in words:
        sum_word_length += len(word)
    if (sum_word_length / total_length) >= 7:
        return True
    else:
        return False


def has_van(words):
    """
    Check if the sentence has the word 'van' in it
    :param words:
    :return: Boolean value
    """
    for word in words:
        if 'van' == word.lower():
            return True
    return False


def has_dat(words):
    """
    Check if the sentence has the word 'dat' in it
    :param words:
    :return: Boolean value
    """
    for word in words:
        if 'dat' == word.lower():
            return True
    return False


def has_en(words):
    """
    Check if the sentence has the word 'en' in it
    :param words:
    :return: Boolean value
    """
    for word in words:
        if 'en' == word.lower():
            return True
    return False


def has_niet(words):
    """
    Check if the sentence has the word 'niet' in it
    :param words:
    :return: Boolean value
    """
    for word in words:
        if 'niet' == word.lower():
            return True
    return False


def has_je(words):
    """
    Check if the sentence has the word 'je' in it
    :param words:
    :return: Boolean value
    """
    for word in words:
        if 'je' == word.lower():
            return True
    return False


def has_ik(words):
    """
    Check if the sentence has the word 'ik' in it
    :param words:
    :return: Boolean value
    """
    for word in words:
        if 'ik' == word.lower():
            return True
    return False


def has_ij(words):
    """
    Check if the word in the sentence has 'ij' in it
    :param words:
    :return: Boolean value
    """
    for word in words:
        if 'ij' in word.lower():
            return True
    return False


def has_common_english(words):
    """
    Check if the sentence has any of the common English words in it
    :param words:
    :return: Boolean value
    """
    for word in words:
        if 'the' == word or 'a' == word or 'i' == word or 'be' == word or 'to' == word or 'and' == word or 'of' == word:
            return True
    return False


def begins_with_z(words):
    """
    Check if the sentence has a word that starts with the letter 'z' in it
    :param words:
    :return: Boolean value
    """
    for word in words:
        if word[0] == 'z':
            return True
    return False


def has_oe_cht(words):
    """
    Check if a word in the sentence has 'oe' or 'cht' in it
    :param words:
    :return: Boolean value
    """
    for word in words:
        if 'oe' in word.lower() or 'cht' in word.lower():
            return True
    return False


def combine_all_attributes(sentence, attributes):
    """
    Method to combine all the attributes to create a list of lists of attributes
    :param sentence: Sentence to be converted to attribute list
    :param attributes: list of attributes
    :return: return list of attributes
    """
    # ['target', 'de', 'het', 'aa', 'ee', 'oo', 'is_length_9', 'van', 'dat', 'en', 'niet', 'je', 'ik', 'ij', 'the', 'a', 'i', 'be', 'to', 'and', 'of', 'z', 'oe', 'cht']
    words = sentence.split()
    words = [word.lower() for word in words]

    features = []
    features.append(has_de(words))
    features.append(has_het(words))
    features.append(has_aa(words))
    features.append(has_ee(words))
    features.append(has_oo(words))
    features.append(is_length_7(words))
    features.append(has_van(words))
    features.append(has_dat(words))
    features.append(has_en(words))
    features.append(has_niet(words))
    features.append(has_je(words))
    features.append(has_ik(words))
    features.append(has_ij(words))
    features.append(has_common_english(words))
    features.append(begins_with_z(words))
    features.append(has_oe_cht(words))

    return features


def get_features(data_file, attributes, perform):
    """
    Get the features and add the target data to it
    :param data_file:
    :param attributes:
    :param perform:
    :return: None
    """
    data_set = pd.read_csv(data_file, sep='|', names=["target", "sentence"])
    if perform == 'train':
        file_name = 'data_set_train.csv'
    else:
        file_name = 'data_set_predict.csv'
    write_file = open(file_name, "w", encoding="utf-8")
    writer = csv.writer(write_file)

    if perform == 'train':
        attributes.insert(0, 'target')
    my_attributes = [attributes]
    writer.writerows(my_attributes)

    for i in range(data_set.shape[0]):
        each_line = data_set.iloc[i]
        if perform == 'train':
            target = each_line.target
            sentence = each_line.sentence
        if perform == 'predict':
            sentence = each_line.target
        features = combine_all_attributes(sentence, attributes)
        my_attributes = features
        if perform == 'train':
            if target == 'en':
                my_attributes.insert(0, 'en')
            else:
                my_attributes.insert(0, 'nl')

        my_attributes = [my_attributes]
        writer.writerows(my_attributes)


def read_csv_file(perform):
    """
    Read the csv file and convert it to a list of lists
    :param perform:
    :return: list of lists
    """
    if perform == 'train':
        file_name = 'data_set_train.csv'
    else:
        file_name = 'data_set_predict.csv'
    with open(file_name, 'r') as read_obj:
        csv_reader = reader(read_obj)
        data = list(csv_reader)
    return data


def plurality_value(parent_examples, attributes):
    """
    To get the majority value among the target components
    :param parent_examples:
    :param attributes:
    :return: Majority language label - nl or en
    """
    count_nl = 0
    count_en = 0
    popular = ''
    for i in range(1, len(parent_examples)):
        if parent_examples[i][0] == 'nl':
            count_nl += 1
        elif parent_examples[i][0] == 'en':
            count_en += 1
    if len(attributes) == 1:
        if count_en > count_nl:
            popular = 'en'
        else:
            popular = 'nl'
    else:
        if count_en > count_nl:
            popular = 'en'
        else:
            popular = 'nl'
    return popular


def all_same_classification(examples):
    """
    Check if all the sentences belong to the same language
    :param examples:
    :return: Boolean value
    """
    class_name = examples[1][0]
    count = 0
    for item in examples[1:]:
        if item[0] == class_name:
            count += 1
    if count == len(examples) - 1 or count == 0:
        return True
    else:
        return False


def find_target_entropy(examples):
    """
    To get the target entropy of the dataset
    :param examples:
    :return: Entropy of the target
    """
    target_unique_dict = dict()
    for i in range(1, len(examples)):
        if examples[i][0] not in target_unique_dict:
            target_unique_dict[examples[i][0]] = 0
        target_unique_dict[examples[i][0]] += 1
    target_unique = list(target_unique_dict.keys())
    count_target = list(target_unique_dict.values())

    e_of_s_target = 0
    for pos in range(len(target_unique)):
        p_i = count_target[pos] / sum(count_target)
        e_of_s_target += - p_i * np.log2(p_i)
    return e_of_s_target


def find_attribute_entropy(examples, attribute, target):
    """
    To get the entropy of the attribute by calculating the weighted entropy
    :param examples:
    :param attribute:
    :param target:
    :return: Weighted entropy
    """
    target_unique = set()
    for i in range(1, len(examples)):
        target_unique.add(examples[i][0])

    # target_length = len(target_unique)
    attribute_index = examples[0].index(attribute)
    attribute_unique = set()
    for i in range(1, len(examples)):
        attribute_unique.add(examples[i][attribute_index])
    # attribute_unique = list(attribute_unique)
    true_en = 0
    false_en = 0
    true_nl = 0
    false_nl = 0
    for i in range(1, len(examples)):
        if str(examples[i][attribute_index]) == str(True):
            if str(examples[i][0]) == 'en':
                true_en += 1
            else:
                true_nl += 1
        else:
            if examples[i][0] == 'en':
                false_en += 1
            else:
                false_nl += 1

    total_true_denominator = true_en + true_nl
    total_false_denominator = false_en + false_nl

    p_i_true_en = (true_en / (total_true_denominator + eps))
    p_i_true_nl = (true_nl / (total_true_denominator + eps))
    p_i_false_en = (false_en / (total_false_denominator + eps))
    p_i_false_nl = (false_nl / (total_false_denominator + eps))

    if p_i_true_en == 0 and p_i_true_nl != 0:
        e_true = (((-1) * p_i_true_nl) * math.log2(p_i_true_nl))
    elif p_i_true_en != 0 and p_i_true_nl == 0:
        e_true = (((-1) * p_i_true_en) * math.log2(p_i_true_en))
    elif p_i_true_en == 0 and p_i_true_nl == 0:
        e_true = 0
    else:
        e_true = (((-1) * p_i_true_en) * math.log2(p_i_true_en)) + (((-1) * p_i_true_nl) * math.log2(p_i_true_nl))

    if p_i_false_en == 0 and p_i_false_nl != 0:
        e_false = (((-1) * p_i_false_nl) * math.log2(p_i_false_nl))
    elif p_i_false_en != 0 and p_i_false_nl == 0:
        e_false = (((-1) * p_i_false_en) * math.log2(p_i_false_en))
    elif p_i_false_en == 0 and p_i_false_nl == 0:
        e_false = 0
    else:
        e_false = (((-1) * p_i_false_en) * math.log2(p_i_false_en)) + (((-1) * p_i_false_nl) * math.log2(p_i_false_nl))

    weightedAvg = ((e_true * total_true_denominator) / (len(examples) + eps)) + \
                  ((e_false * total_false_denominator) / (len(examples) + eps))

    return weightedAvg


def choose_best_attribute(examples, attributes, target_entropy_val):
    """
    To calculate the information gain and get the attribute with the highest information gain
    :param examples:
    :param attributes:
    :param target_entropy_val:
    :return: Attribute with the maximum information gain
    """
    information_gain = []
    target = examples[0][0]
    for attribute in attributes[1:]:
        attribute_entropy_val = find_attribute_entropy(examples, attribute, target)
        information_gain.append((target_entropy_val - attribute_entropy_val))

    max_ig = max(information_gain)
    max_index = information_gain.index(max_ig)
    max_iga_attribute = attributes[max_index]
    return max_iga_attribute


def partition(examples, A, A_index, attributes):
    """
    Partition the data into true rows and false rows
    :param examples:
    :param A:
    :param A_index:
    :param attributes:
    :return: true and false rows
    """
    true_rows, false_rows = [], []
    true_rows.append(attributes)
    false_rows.append(attributes)
    for i in range(1, len(examples)):
        if examples[i][A_index] == 'True':
            true_rows.append(examples[i])
        else:
            false_rows.append(examples[i])
    return true_rows, false_rows


def decision_tree(examples, attributes, parent_examples, weights=None):
    """
    The method to construct the decision tree
    :param examples:
    :param attributes:
    :param parent_examples:
    :param weights:
    :return: Decision Tree
    """
    if len(examples) == 1:
        # print(plurality_value(parent_examples))
        return Leaf(plurality_value(parent_examples, attributes))
    elif all_same_classification(examples):
        # print(examples[0][0])
        return Leaf(examples[1][0])
    elif len(attributes) == 1:
        # print(plurality_value(examples))
        return Leaf(plurality_value(examples, attributes))
    else:
        target_entropy_val = find_target_entropy(examples)
        A = choose_best_attribute(examples, attributes, target_entropy_val)
        node = DTree(A, None, None, None, None)
        A_index = attributes.index(A)
        true_data, false_data = partition(examples, A, A_index, examples[0])

        updated_columns = []
        updated_columns.extend(attributes[:A_index])
        updated_columns.extend(attributes[A_index + 1:])
        updated_columns_copy_left = updated_columns[:]
        updated_columns_copy_right = updated_columns[:]

        node.true_branch = decision_tree(true_data, updated_columns_copy_left, examples)
        node.false_branch = decision_tree(false_data, updated_columns_copy_right, examples)
        return node


def to_print(node, spacing=''):
    """
    Method to print the decision tree
    :param node:
    :param spacing:
    :return: None
    """
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.result)
        return
    print(spacing + str(node.node_name))
    print(spacing + '--> True')
    to_print(node.true_branch, spacing + '    ')
    print(spacing + '--> False')
    to_print(node.false_branch, spacing + '    ')


def classify(row, node):
    """
    To classify the row to true and false branches for prediction
    :param row:
    :param node:
    :return: the leaf node value
    """
    if isinstance(node, Leaf):
        return node.result

    if row[0] == 'True':
        return classify(row[1:], node.true_branch)
    else:
        return classify(row[1:], node.false_branch)


def predict_dt(examples, node):
    """
    Print the language that is predicted
    :param examples:
    :param node:
    :return: None
    """
    for row in examples[1:]:
        print(classify(row, node))


# def adaboost(examples, K):
#     for item in examples:
#         print(item)
#
#     n = len(examples)-1
#     print(n)
#     eps_2 = 1/(2*n)
#     w = [1/n]*n
#     print(eps_2, w)
#     h, z = [], []
#     attributes = examples[0]
#     # print(attributes)
#     for k in range(K):
#         h_k = decision_tree(examples, attributes, w)
#         h.append(h_k)
#         error = 0

# class Node:
#     __slots__ = 'node_name', 'true_branch', 'false_branch', 'total_targets', 'target_set', 'attributes', 'visited', 'depth'
#
#     def __init__(self, total_targets, target_set, attributes, visited, depth):
#         self.node_name = None
#         self.true_branch = None
#         self.false_branch = None
#         self.total_targets = total_targets
#         self.target_set = target_set
#         self.attributes = attributes
#         self.visited = visited
#         self.depth = depth


def find_ada_attribute_entropy(examples, attribute, target, weights):
    """
    To find the entropy of the attribute based on the weights of the sentences
    :param examples:
    :param attribute:
    :param target:
    :param weights:
    :return: weighted entropy value
    """
    attribute_index = examples[0].index(attribute)
    attribute_unique = set()
    for i in range(1, len(examples)):
        attribute_unique.add(examples[i][attribute_index])

    true_en = 0
    false_en = 0
    true_nl = 0
    false_nl = 0
    for i in range(1, len(examples) - 1):
        if examples[i][attribute_index] == 'True':
            if examples[i][0] == 'en':
                true_en += weights[i]
            elif examples[i][0] == 'nl':
                true_nl += weights[i]
        elif examples[i][attribute_index] == 'False':
            if examples[i][0] == 'en':
                false_en += weights[i]
            elif examples[i][0] == 'nl':
                false_nl += weights[i]

    total_true_denominator = true_en + true_nl
    total_false_denominator = false_en + false_nl

    p_i_true_en = (true_en / (total_true_denominator + eps))
    p_i_true_nl = (true_nl / (total_true_denominator + eps))
    p_i_false_en = (false_en / (total_false_denominator + eps))
    p_i_false_nl = (false_nl / (total_false_denominator + eps))

    if p_i_true_en == 0 and p_i_true_nl != 0:
        e_true = (((-1) * p_i_true_nl) * math.log2(p_i_true_nl)) + 0
    elif p_i_true_en != 0 and p_i_true_nl == 0:
        e_true = (((-1) * p_i_true_en) * math.log2(p_i_true_en)) + 0
    elif p_i_true_en == 0 and p_i_true_nl == 0:
        e_true = 0
    else:
        e_true = (((-1) * p_i_true_en) * math.log2(p_i_true_en)) + (((-1) * p_i_true_nl) * math.log2(p_i_true_nl))

    if p_i_false_en == 0 and p_i_false_nl != 0:
        e_false = (((-1) * p_i_false_nl) * math.log2(p_i_false_nl)) + 0
    elif p_i_false_en != 0 and p_i_false_nl == 0:
        e_false = (((-1) * p_i_false_en) * math.log2(p_i_false_en)) + 0
    elif p_i_false_en == 0 and p_i_false_nl == 0:
        e_false = 0
    else:
        e_false = (((-1) * p_i_false_en) * math.log2(p_i_false_en)) + (
                ((-1) * p_i_false_nl) * math.log2(p_i_false_nl))

    weightedAvg = ((e_true * total_true_denominator) / (len(examples) + eps)) \
                  + ((e_false * total_false_denominator) / (len(examples) + eps))
    return weightedAvg


def get_node_object(max_iga_attribute, attributes, weights, depth, examples, target_set, node):
    """
    Method to assign the true and false branch for the stump
    :param max_iga_attribute:
    :param attributes:
    :param weights:
    :param depth:
    :param examples:
    :param target_set:
    :param node:
    :return: Node
    """
    attribute_index = examples[0].index(max_iga_attribute)
    count_true_en = 0
    count_false_en = 0
    count_true_nl = 0
    count_false_nl = 0

    for i in range(1, len(examples) - 1):
        if examples[i][attribute_index] == 'True':
            if examples[i][0] == 'en':
                count_true_en += 1 * weights[i]
            elif examples[i][0] == 'nl':
                count_true_nl += 1 * weights[i]
        else:
            if examples[i][0] == 'en':
                count_false_en += 1 * weights[i]
            elif examples[i][0] == 'nl':
                count_false_nl += 1 * weights[i]

    true_nf = DTree(None, target_set, attributes, None, depth + 1)
    false_nf = DTree(None, target_set, attributes, None, depth + 1)

    if count_true_en > count_true_nl:
        true_nf.node_name = 'en'
    elif count_true_nl > count_true_en:
        true_nf.node_name = 'nl'
    if count_false_en > count_false_nl:
        false_nf.node_name = 'en'
    elif count_false_nl > count_false_en:
        false_nf.node_name = 'nl'

    node.true_branch = true_nf
    node.false_branch = false_nf

    return node


def get_stump(node, weights, target_set, attributes, depth, examples, list_of_stump_names):
    """
    Method to get the attribute with the maximum information gain and get the node object
    :param node:
    :param weights:
    :param target_set:
    :param attributes:
    :param depth:
    :param examples:
    :param list_of_stump_names:
    :return: Node
    """
    en_count = 0
    nl_count = 0
    for i in range(1, len(examples) - 1):
        if examples[i][0] == 'en':
            en_count += 1 * weights[i]
        else:
            nl_count += 1 * weights[i]

    p_target_en = (en_count / (en_count + nl_count))
    p_target_nl = (nl_count / (en_count + nl_count))

    target_entropy = ((-1) * p_target_en * math.log2(p_target_en)) + ((-1) * p_target_nl * math.log2(p_target_nl))

    information_gain = []
    target = examples[0][0]
    for attribute in attributes:
        attribute_entropy_val = find_ada_attribute_entropy(examples, attribute, target, weights)
        information_gain.append((target_entropy - attribute_entropy_val))

    max_ig = max(information_gain)
    max_index = information_gain.index(max_ig)
    max_iga_attribute = attributes[max_index]
    node.node_name = max_iga_attribute
    node_object = get_node_object(max_iga_attribute, attributes, weights, depth, examples, target_set, node)
    return node_object


def adaboost(examples, no_of_stumps):
    """
    The Ada Boost method that creates the stumps and calculate the normalized weights
    :param examples:
    :param no_of_stumps:
    :return: hypothesis weights and list of stumps
    """
    n = len(examples) - 1
    weights = [1 / n] * n
    attributes = examples[0][1:]

    number_list = []
    stumps = []
    sentence_weights = [1] * no_of_stumps
    target_set = []
    list_of_stump_names = []
    for i in range(1, len(examples)):
        target_set.append(examples[i][0])

    for i in range(1, len(examples)):
        number_list.append(i)

    for i in range(no_of_stumps):
        node = DTree(None, number_list, target_set, attributes, 0)
        stump = get_stump(node, weights, target_set, attributes, 0, examples, list_of_stump_names)
        error_val = 0
        for index in range(1, len(examples) - 1):
            attribute_index = attributes.index(stump.node_name) + 1
            if examples[index][attribute_index] == 'True':
                if stump.true_branch.node_name != examples[index][0]:
                    error_val += weights[index]

            elif examples[index][attribute_index] == 'False':
                if stump.false_branch.node_name != examples[index][0]:
                    error_val += weights[index]

        total_performance_of_stump = 0.5 * np.log((1 - error_val) / error_val)

        for index in range(1, len(examples) - 1):
            attribute_index = attributes.index(stump.node_name)
            if examples[index][attribute_index] == 'True':
                if stump.true_branch.node_name == examples[index][0]:
                    weights[index] *= (np.exp((-1) * total_performance_of_stump))
                else:
                    weights[index] *= (np.exp(total_performance_of_stump))

            elif examples[index][attribute_index] == 'False':
                if stump.false_branch.node_name == examples[index][0]:
                    weights[index] *= (np.exp((-1) * total_performance_of_stump))
                else:
                    weights[index] *= (np.exp(total_performance_of_stump))

        stumps.append(stump)

        weights = [(x / sum(weights)) for x in weights]
        sentence_weights[i] = math.log2((1 - error_val) / error_val)
        attributes.remove(stump.node_name)

    return sentence_weights, stumps


def predict_ada(examples, node):
    """
    The method to predict the language based on the number of en and nl sentences
    :param examples:
    :param node:
    :return: None
    """
    list_of_stumps = node[0]
    weights = node[1]

    for i in range(1, len(examples)):
        _sum = 0
        row = examples[i]
        for index in range(len(list_of_stumps)):
            if list_of_stumps[index].true_branch.node_name == 'en':
                att_index = examples[0].index(list_of_stumps[index].node_name)
                if row[att_index] == 'True':
                    _sum += ((-1) * weights[index])
                else:
                    _sum += (1 * weights[index])
            else:
                att_index = examples[0].index(list_of_stumps[index].node_name)
                if row[att_index] == 'True':
                    _sum += ((-1) * weights[index])
                else:
                    _sum += (1 * weights[index])
        if _sum > 0:
            print('en')
        else:
            print('nl')


# def most_useful_stumps(examples, node):
#     """
#     Used to test the usefulness of the data manually
#     :param examples:
#     :param node:
#     :return: None
#     """
#     list_of_stumps = node[0]
#     weights = node[1]
#
#     for i in range(1, len(examples)):
#         _sum = 0
#         row = examples[i]
#         index = 12
#         # for index in range(len(list_of_stumps)):
#         if list_of_stumps[0].true_branch.node_name == 'en':
#             att_index = examples[0].index(list_of_stumps[index].node_name)
#             if row[att_index] == 'True':
#                 _sum += ((-1) * weights[index])
#             else:
#                 _sum += (1 * weights[index])
#         else:
#             att_index = examples[0].index(list_of_stumps[index].node_name)
#             if row[att_index] == 'True':
#                 _sum += ((-1) * weights[index])
#             else:
#                 _sum += (1 * weights[index])
#         if _sum > 0:
#             print('en')
#         else:
#             print('nl')
#     print(list_of_stumps[index].node_name)


def main():
    """
    The main method
    :return: None
    """
    perform = sys.argv[1]
    attributes = get_language_features()

    if perform == 'train':
        data_file = sys.argv[2]
        hypothesisOut = sys.argv[3]
        learning_type = sys.argv[4]
        get_features(data_file, attributes, perform)
        examples = read_csv_file(perform)

        if learning_type == 'dt':
            node = decision_tree(examples, attributes, parent_examples=())
            pickle.dump(node, open(hypothesisOut, "wb"))
            to_print(node)
        elif learning_type == 'ada':
            weights, stumps = adaboost(examples, 16)
            pickle.dump((stumps, weights), open(hypothesisOut, "wb"))

    elif perform == 'predict':
        print('Which learning algorithm do you want to use to predict?')
        print('Enter \'dt\' for Decision Tree or \'ada\' for Adaboost')
        learning_type = input('Please enter the learning type: ')

        if learning_type == 'dt':
            hypothesisOut = sys.argv[2]
            data_file = sys.argv[3]
            get_features(data_file, attributes, perform)
            examples = read_csv_file(perform)
            node = pickle.load(open(hypothesisOut, "rb"))
            predict_dt(examples, node)

        elif learning_type == 'ada':
            hypothesisOut = sys.argv[2]
            data_file = sys.argv[3]
            get_features(data_file, attributes, perform)
            examples = read_csv_file(perform)
            node = pickle.load(open(hypothesisOut, "rb"))
            predict_ada(examples, node)
            # most_useful_stumps(examples, node)


if __name__ == '__main__':
    main()

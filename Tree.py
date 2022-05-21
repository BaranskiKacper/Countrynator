from Leaf import Leaf
from DecisionNode import DecisionNode
from Question import Question
from utils import class_counts


class Tree:
    def __init__(self):
        self.used_values = []
        self.dont_know_nodes = []

    def build_tree(self, rows):
        """Builds the tree."""

        # Try partitioning the dataset on each of the unique attribute,
        # calculate the information gain,
        # and return the question that produces the highest gain.
        gain, question = self.find_best_split(rows)

        # Base case: no further info gain
        # Since we can ask no further questions,
        # we'll return a leaf.
        if gain == 0:
            return Leaf(rows)

        # If we reach here, we have found a useful feature / value
        # to partition on.
        true_rows, false_rows = self.partition(rows, question)

        # Recursively build the true branch.
        true_branch = self.build_tree(true_rows)

        # Recursively build the false branch.
        false_branch = self.build_tree(false_rows)

        # Return a Question node.
        # This records the best feature / value to ask at this point,
        # as well as the branches to follow
        # depending on the answer.
        return DecisionNode(question, true_branch, false_branch)

    def find_best_split(self, rows):
        """Find the best question to ask by iterating over every feature / value
        and calculating the information gain."""
        best_gain = 0  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_value = None
        current_uncertainty = self.gini(rows)
        n_features = len(rows[0]) - 1  # number of columns

        for col in range(n_features):  # for each feature

            values = set([row[col] for row in rows])  # unique values in the column
            values = sorted(values)

            for val in values:  # for each value

                if val not in self.used_values:
                    question = Question(col, val)

                    # try splitting the dataset
                    true_rows, false_rows = self.partition(rows, question)

                    # Skip this split if it doesn't divide the
                    # dataset.
                    if len(true_rows) == 0 or len(false_rows) == 0:
                        continue

                    # Calculate the information gain from this split
                    gain = self.info_gain(true_rows, false_rows, current_uncertainty)

                    if gain >= best_gain:
                        best_gain, best_question, best_value = gain, question, val

        self.used_values.append(best_value)
        return best_gain, best_question

    def partition(self, rows, question):
        """Partitions a dataset.

        For each row in the dataset, check if it matches the question. If
        so, add it to 'true rows', otherwise, add it to 'false rows'.
        """
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows

    def gini(self, rows):
        """Calculate the Gini Impurity for a list of rows."""
        counts = class_counts(rows)
        impurity = 1
        for _ in counts:
            prob_of_lbl = 1 / float(len(rows))
            impurity -= prob_of_lbl**2
        return impurity

    def info_gain(self, left, right, current_uncertainty):
        """Information Gain.

        The uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        """
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)

    def print_tree(self, node, spacing=""):
        """World's most elegant tree printing function."""

        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            print(spacing + "Predict:", node.predictions[0])
            return

        # Print the question at this node
        print(spacing + str(node.question))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.print_tree(node.true_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.print_tree(node.false_branch, spacing + "  ")

    def get_leaf_count(self, node):
        if isinstance(node, Leaf):
            return 1
        else:
            return self.get_leaf_count(node.true_branch) + self.get_leaf_count(node.false_branch)

    def game(self, node):
        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            print("Predict:", node.predictions[0], "- Czy zgadlem?")
            is_correct = input()
            if is_correct == "tak":
                return
            else:
                if not self.dont_know_nodes:
                    print("Nie wiem o jakie panstwo chodzi :(")
                    print("Musiales odpowiedziec na jakies pytanie zle")
                    return
                else:
                    dont_know_node = self.dont_know_nodes.pop()
                    self.game(dont_know_node)

        # Print the question at this node
        if isinstance(node, DecisionNode):
            print(str(node.question))
            guess = input()

            if guess == "tak":
                # Call this function recursively on the true branch
                self.game(node.true_branch)

            if guess == "nie wiem":
                false_branch_leaf_count = self.get_leaf_count(node.false_branch)
                true_branch_leaf_count = self.get_leaf_count(node.true_branch)
                if false_branch_leaf_count < true_branch_leaf_count:
                    self.dont_know_nodes.append(node.true_branch)
                    self.game(node.false_branch)
                else:
                    self.dont_know_nodes.append(node.false_branch)
                    self.game(node.true_branch)

            if guess == "nie":
                # Call this function recursively on the false branch
                self.game(node.false_branch)

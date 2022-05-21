from Tree import *
from utils import load_data


def main():
    training_data = load_data()

    t = Tree()
    my_tree = t.build_tree(training_data)
    t.print_tree(my_tree)

    while True:
        t.game(my_tree)
        print("Czy chcesz zagrac jeszcze raz?")
        option = input()
        if option == "nie":
            break


if __name__ == "__main__":
    main()

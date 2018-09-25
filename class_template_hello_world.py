"""
Hello World as a Python class template.

Example class module.

Sig Nin
25 Sep 2018
"""

# Text to display.
MESSAGE = "Hello world!"

class HelloWorld:
    """
    Class to display a greeting.
    """

    # Method to set a new name.
    def set_name(self, name):
        """
        Set a new name to display in the greeting.
        """
        self.name = name

    # Method to get the name.
    def get_name(self):
        """
        Get the name to be displayed in the greeting.
        """
        return self.name

    # This is the constructor, which takes a name
    # to be displayed in the greeting.
    def __init__(self, name):
        """
        Initialize a new instance with a name.
        """
        self.set_name(name)

    # Method to display the greeting
    def display(self):
        """
        Display the greeting followed by the name.
        """
        print(MESSAGE + ', from ', self.name)

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':

    from datetime import datetime
    from random import randint

    names = [ "Charlie", "Zuma", "Apollo", "Avery", "Deepta",
              "Maria", "Kun", "Antonio", "Perry", "Xena" ]

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    # Run the tests 5 times
    for k in range(5):

        i = randint(0, len(names)-1)
        name = names[i]

        messenger = HelloWorld(name)
        print(messenger.get_name())
        messenger.display()

        i = randint(0, len(names)-1)
        name = names[i]

        messenger.set_name(name)
        print(messenger.get_name())
        messenger.display()

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

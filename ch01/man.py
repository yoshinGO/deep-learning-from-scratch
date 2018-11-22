class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")

    def hello(self):
        print("hell " + self.name + "!")

    def goodbye(self):
        print("good-bye " + self.name + "!")


m = Man("David")
m.hello()
m.goodbye()

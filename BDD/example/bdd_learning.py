from dd.autoref import BDD as _BDD

class BDD:
    def __init__(self):
        self.bdd = _BDD()
    
    def add_var(self, var, level):
        self.bdd.add_var(var, level)


if __name__ == '__main__':
    bdd = BDD()
    bdd.add_var('x', 0)
    bdd.add_var('y', 1)
    bdd.add_var('z', 2)
    print("=== Internal BDD Manager State ===")
    print(bdd.bdd)
    print("\n=== Registered Variables (Name -> Level) ===")
    # Accessing internal BDD manager's variable storage
    # Different backends might store this differently, but 'vars' or 'levels' are common.
    # Let's try printing the vars attribute which usually holds the variables.
    if hasattr(bdd.bdd, 'vars'):
        print(f"Variables: {bdd.bdd.vars}")
    
    # We can also check specific levels if the backend supports it
    print(f"Level of x: {bdd.bdd.level_of_var('x')}")
    print(f"Level of y: {bdd.bdd.level_of_var('y')}")
    print(f"Level of z: {bdd.bdd.level_of_var('z')}")
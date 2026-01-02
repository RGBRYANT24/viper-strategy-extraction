from dd.autoref import BDD
b = BDD()
print("Methods with 'dot' or 'dump':")
print([d for d in dir(b) if 'dump' in d or 'dot' in d])

# Try to print docstring of dump
if hasattr(b, 'dump'):
    print("\nDocstring of dump:")
    print(b.dump.__doc__)

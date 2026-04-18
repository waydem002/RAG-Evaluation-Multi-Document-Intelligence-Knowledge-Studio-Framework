EVALUATION_DATA: list[dict[str, str]] = [
    {
        "question": "What is a Python list comprehension and how would you use it to create a list of squares from 1 to 10?",
        "ground_truth": "A list comprehension is a concise way to create lists. "
                        "Example: squares = [x**2 for x in range(1, 11)]. "
                        "This creates [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]."
    },
    # -----
    # It's best to test your code with only a question or two
    # until you're confident that it's working.
    # This saves a lot of your free API calls until you need them.
    # -----
    # {
    # "question": "Explain the difference between `deepcopy()` and `copy()` from the copy module.",
    # "ground_truth": "copy() creates a shallow copy, while deepcopy() creates a deep (recursive) copy. "
    #                 "Shallow copy copies the object but not nested mutable objects, whereas deep copy "
    #                 "recursively copies all nested objects."
    # },
    {
        "question": "How do you check if a key exists in a Python dictionary without raising a KeyError?",
        "ground_truth": "You can use the 'in' keyword: `if key in my_dict:`, or use the safer method "
                        "`my_dict.get(key)` which returns None if the key doesn't exist (or a default value)."
    },
    {
        "question": "What is the difference between `__str__` and `__repr__` in Python classes?",
        "ground_truth": "`__str__` is used for informal, user-friendly string representation (called by str() and print()). "
                        "`__repr__` is used for official, unambiguous representation (called by repr() and should be valid Python code if possible)."
    },
    {
        "question": "Write a Python function that takes a list of integers and returns only the even numbers using a filter and lambda.",
        "ground_truth": "def get_even_numbers(nums):\n"
                        "    return list(filter(lambda x: x % 2 == 0, nums))"
    },
    #{
     #   "question": "What is the capital of Germany?",
      #  "ground_truth": "This question is unrelated to Python programming. "
       #                 "The capital of Germany is Berlin, but it has no connection to the Python language."
    #},
     {
     "question": "Explain how Python's Global Interpreter Lock (GIL) affects multithreading.",
     "ground_truth": "The GIL is a mutex that protects access to Python objects, preventing multiple native threads from executing Python bytecodes at once. "
                     "This makes true CPU-bound parallelism difficult with threads, which is why multiprocessing is often preferred for CPU-heavy tasks."
     }
]
def get_prompt(dataset_name):
    if "Toy" in dataset_name:
        instruction_prompt = "Given a list of toys the user has played before, please recommend a new toy that the user likes to the user."
        history_prompt = "The user has played the following toys before: "
    elif "Amazon_Books" in dataset_name:
        instruction_prompt = "Given a list of books the user has read before, please recommend a new book that the user likes to the user."
        history_prompt = "The user has read the following books before: "
    elif "Clothing" in dataset_name:
        instruction_prompt = "Given a list of clothing the user has worn before, please recommend a new clothing that the user likes to the user."
        history_prompt = "The user has worn the following clothing before: "
    elif "Office" in dataset_name:
        instruction_prompt = "Given a list of office products the user has used before, please recommend a new office product that the user likes to the user."
        history_prompt = "The user has used the following office products before: "

    return instruction_prompt, history_prompt

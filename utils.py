def get_prompt(dataset_name):
    if "amazon_game" in dataset_name:
        instruction_prompt = "Given a list of video games the user has played before, please recommend a new video game that the user likes to the user."
        history_prompt = "The user has played the following video games before: "
    elif ("Movie" in dataset_name) or ("ml-10m" in dataset_name) or ("ml-100k" in dataset_name):
        instruction_prompt = "Given a list of movies the user has watched before, please recommend a new movie that the user likes to the user."
        history_prompt = "The user has watched the following movies before: "
    elif "Toy" in dataset_name:
        instruction_prompt = "Given a list of toys the user has played before, please recommend a new toy that the user likes to the user."
        history_prompt = "The user has played the following toys before: "
    elif "Amazon_Books" in dataset_name:
        instruction_prompt = "Given a list of books the user has read before, please recommend a new book that the user likes to the user."
        history_prompt = "The user has read the following books before: "
    elif "Food" in dataset_name:
        instruction_prompt = "Given a list of foods the user has eaten before, please recommend a new food that the user likes to the user."
        history_prompt = "The user has eaten the following foods before: "
    elif "steam" in dataset_name:
        instruction_prompt = "Given a list of games the user has played before, please recommend a new game that the user likes to the user."
        history_prompt = "The user has played the following games before: "
    elif "Musical_Instruments" in dataset_name:
        instruction_prompt = "Given a list of musical instruments the user has played before, please recommend a new musical instrument that the user likes to the user."
        history_prompt = "The user has played the following musical instruments before: "
    elif "CD" in dataset_name:
        instruction_prompt = "Given a list of CDs the user has listened before, please recommend a new CD that the user likes to the user."
        history_prompt = "The user has listened the following CDs before: "
    elif "beauty" in dataset_name:
        instruction_prompt = "Given a list of beauty products the user has used before, please recommend a new beauty product that the user likes to the user."
        history_prompt = "The user has used the following beauty products before: "
    elif "Digital_Music" in dataset_name:
        instruction_prompt = "Given a list of digital music the user has listened before, please recommend a new digital music that the user likes to the user."
        history_prompt = "The user has listened the following digital music before: "
    elif "Clothing" in dataset_name:
        instruction_prompt = "Given a list of clothing the user has worn before, please recommend a new clothing that the user likes to the user."
        history_prompt = "The user has worn the following clothing before: "
    elif "Electronics" in dataset_name:
        instruction_prompt = "Given a list of electronics the user has used before, please recommend a new electronic that the user likes to the user."
        history_prompt = "The user has used the following electronics before: "
    elif "Sports" in dataset_name:
        instruction_prompt = "Given a list of outdoor sports equipment the user has played before, please recommend a new outdoor sports equipment that the user likes to the user."
        history_prompt = "The user has used the following outdoor sports equipment before: "
    elif "Home" in dataset_name:
        instruction_prompt = "Given a list of home products the user has used before, please recommend a new home product that the user likes to the user."
        history_prompt = "The user has used the following home products before: "
    elif "Kindle" in dataset_name:
        instruction_prompt = "Given a list of Kindle books the user has read before, please recommend a new Kindle book that the user likes to the user."
        history_prompt = "The user has read the following Kindle books before: "
    elif "Office" in dataset_name:
        instruction_prompt = "Given a list of office products the user has used before, please recommend a new office product that the user likes to the user."
        history_prompt = "The user has used the following office products before: "

    return instruction_prompt, history_prompt

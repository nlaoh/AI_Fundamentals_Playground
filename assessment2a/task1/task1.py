# What you need to do
# 1. Write at least five about either the history or provide definition in regards to Artifical Intelligence
# 2. At least one fact needs to integer number, at least one fact needs to string and at least one fact needs to a list
# 3. One fact needs to be defind as 'turing-test-proposed-year' i.e. in what year was the turing test proposed
#       (Check module 1)


facts = {
    ### https://labs.vocareum.com/main/main.php?m=editor&asnid=1911338&stepid=1911339&hideNavBar=1
    ### YOUR CODE HERE
    ###
    "definition": "Artificial intelligence is a system that can act independently as its own entity and formulate decisions.",
    "origin": "The term 'artificial intelligence' was coined at Dartmouth University by John McCarthy and Marvin Minsky in 1956.",
    "market-value": 136550000000,
    "attributes": [ 
        "Learning (Machine Learning)",
        "Decision-making, automated reasoning",
        "Language-processing",
        "Vision and speech recognition",
        "Robotics (movement)",
        "Knowledge representation"
    ],
    "types": [
        "Logic-based agent",
        "Expert Systems (Knowledge-based agents)",
        "Machine Learning",
        "Pattern Recognition"     
    ],
    "turing-test-proposed-year": 1950
}

running = True


print("Hi, I'm an AI History/Definition Knowledge based system\n")

# What you need to do
#1. Match the key terms entered by the user with the knowledge base created above. 
#2. Write your Python code below.
#3. Run your code to ensure it is working properly.

###
### YOUR CODE HERE
###

while running: 
    user_input = str(input("What would you like to know about artificial intelligence?\n")).lower()
    if "definition" in user_input \
        or "define" in user_input \
        or "what is artificial intelligence" in user_input \
        or "tell me about artificial intelligence" in user_input:
        print(facts["definition"])
    elif "origin" in user_input \
        or "when was artificial intelligence" in user_input \
        or "when was ai" in user_input \
        or "created" in user_input:
        print(facts["origin"])
    elif "market" in user_input \
        or "value" in user_input:
        print("The estimated market-value of AI as of 2022 is " + format(facts["market-value"], ',') + " US dollars.")
    elif "attribute" in user_input \
        or "aspects" in user_input:
        print("These are the attributes that an AI must have:")
        for attribute in facts["attributes"]:
            print("\t" + attribute)
    elif "types" in user_input \
        or "forms" in user_input:
        print("These are some types of AI:")
        for form in facts["types"]:
            print("\t" + form)
    elif "turing" in user_input \
        or "turing-test" in user_input \
        or "turing test" in user_input:
        print("The turing-test was proposed by Alan Turing in", facts["turing-test-proposed-year"])
    elif "quit" in user_input \
        or "exit" in user_input: #quit condition
        running = False
    else:
        print("I'm not sure what you mean, could you try rephrasing?\n")


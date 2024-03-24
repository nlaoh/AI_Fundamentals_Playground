from pomegranate import *

code = Node(DiscreteDistribution({
    "plain-text": 0.05,
    "binary": 0.7,
    "obfuscated": 0.25
}), name="code")

resize = Node(ConditionalProbabilityTable([
    ["plain-text", "no", 0.95],
    ["plain-text", "yes", 0.05],
    ["binary", "no", 0.75],
    ["binary", "yes", 0.25],
    ["obfuscated", "no", 0.4],
    ["obfuscated", "yes", 0.6]
], [code.distribution]), name="resize")

signature = Node(ConditionalProbabilityTable([
    ["plain-text", "no", "present", 0.8],
    ["plain-text", "no", "absent", 0.2],
    ["plain-text", "yes", "present", 0.9],
    ["plain-text", "yes", "absent", 0.1],

    ["binary", "no", "present", 0.6],
    ["binary", "no", "absent", 0.4],
    ["binary", "yes", "present", 0.8],
    ["binary", "yes", "absent", 0.2],

    ["obfuscated", "no", "present", 0.6],
    ["obfuscated", "no", "absent", 0.4],
    ["obfuscated", "yes", "present", 0.95],
    ["obfuscated", "yes", "absent", 0.05],
], [code.distribution, resize.distribution]), name="signature")

outcome = Node(ConditionalProbabilityTable([
    ["present", "malware", 0.9],
    ["present", "software", 0.1],
    ["absent", "malware", 0.4],
    ["absent", "software", 0.6],
], [signature.distribution]), name="outcome")

model = BayesianNetwork()

#TODO: Add the states
###
### YOUR CODE HERE
###
model.add_state(code)
model.add_state(resize)
model.add_state(signature)
model.add_state(outcome)

#TODO: Add the edges
###
### YOUR CODE HERE
###
model.add_edge(code, resize)
model.add_edge(code, signature)
model.add_edge(resize, signature)
model.add_edge(signature, outcome)

#TODO: Train the model
###
### YOUR CODE HERE
###
model.bake()

low_probability = 1
medium_probability = 1
high_probability = 0
low_probability_prediction = 1
high_probability_prediction = 0

#TODO: Calculate a probability that is less that 0.01 for a given observation (assign to low_probability)
###
### YOUR CODE HERE
###
# for i in range(100):
low_probability = model.probability([["plain-text", "yes", "absent", "software"]])
print("low probability is: ", low_probability)

#TODO: Calculate a probability that is less than 0.1 but greater than 0.01 for a given observation (assign to medium_probability)
###
### YOUR CODE HERE
###
medium_probability = model.probability([["obfuscated", "no", "absent", "software"]])
print("medium probability is: ", medium_probability)

#TODO: Calculate a probability that is greater than 0.1 for a given observation (assign to high_probability)
###
### YOUR CODE HERE
###
high_probability = model.probability([["binary", "no", "present", "malware"]])
print("high probability is: ", high_probability)

#TODO: Calculate predictions based on the evidence that is less than 0.5 and assigned it to low_probability_prediction.
###
### YOUR CODE HERE
###
predictions = model.predict_proba({
    "code": "obfuscated",
    "resize": "no",
    "signature": "present"
})

low_probability_prediction = predictions[3].parameters[0]['software']
print("low_probability_prediction is: ", low_probability_prediction)

#TODO: Calculate predictions based on the evidence that is greater than 0.8 and assigned it to high_probability_prediction.
###
### YOUR CODE HERE
###
predictions = model.predict_proba({
    "resize": "plain-text",
    "code": "obfuscated",
    "signature": "present"
})

high_probability_prediction = predictions[3].parameters[0]['malware']
print("high_probability_prediction is: ", high_probability_prediction)
facts = {
    'name':"Royal Melbourne Institute of Technology",
    'short-name':"RMIT",
    'about':'RMIT University, officially the Royal Melbourne Institute of Technology (RMIT), is a public research university in Melbourne, Australia. Founded in 1887 by Francis Ormond, RMIT began as a night school offering classes in art, science, and technology, in response to the industrial revolution in Australia. It was a private college for more than a hundred years before merging with the Phillip Institute of Technology to become a public university in 1992. It has an enrolment of around 95,000 higher and vocational education students, making it the largest dual-sector education institution in Australia. With an annual revenue of around A$1.5 billion, it is also one of the wealthiest universities in Australia. It is rated a five star university by Quacquarelli Symonds (QS) and is ranked 15th in the World for art and design subjects in the QS World University Rankings, making it the top art and design university in Australia and Oceania. \n\nThe main campus of RMIT is situated on the northern edge of the historic Hoddle Grid in the city centre of Melbourne. It has two satellite campuses in the city\'s northern suburbs of Brunswick and Bundoora and a training site situated on the RAAF Williams base in the western suburb of Point Cook. It also has a training site at Bendigo Airport in the Victorian city of Bendigo and a research site in Hamilton near the Grampians National Park. In Asia, it has two branch campuses in Ho Chi Minh City and Hanoi and a training centre in Da Nang in Vietnam as well as teaching partnerships in China, Hong Kong, Indonesia, Singapore and Sri Lanka. In Europe, it has a research and collaboration centre in the Spanish city of Barcelona.',
    'location':'124 La Trobe St, Melbourne VIC 3000 (City campus)',
    'year':'1887',
    'website':'rmit.edu.au',
    'student-number':94933,
    'undergraduates-number':58775,
    'postgraduates-number':19064,
    'doctoral-number':2194,
    'other-student-number':17094,
    'chancellor':'Ziggy Switkowski',
    'vice-chancellor':'Martin Bean',
    'schools':['Accounting, Information Systems and Supply Chain',
                'Architecture and Urban Design',
                'Art',
                'Business and Law',
                'Computing Technologies',
                'Design',
                'Economics, Finance and Marketing',
                'Education',
                'Engineering',
                'Fashion and textiles',
                'Health and Biomedical Sciences',
                'Global, Urban and Social Studies',
                'Management',
                'Media and Communication',
                'Property, Construction and Project Management',
                'Science'],
    'colleges':['College of Business and Law',
                 'College of Design and Social Context',
                 'College of Vocational Education',
                 'STEM College'

    ]
}

running = True

print("Hi, I'm a " + facts['short-name'] + " Knowledge based system\n")

while running:
    question_asked = str(input("What would you like to know?\n")).lower()
    
    if "name" in question_asked \
        and "uni" in question_asked \
        or "university" in question_asked:

        print("The name of this university is " + facts['name'] + \
            " also known as " + facts['short-name'])
    elif "introduction" in question_asked \
        or "about rmit" in question_asked \
        or "about" in question_asked \
        and ("rmit" in question_asked \
        or "uni" in question_asked \
        or "university" in question_asked):

        print("About " + facts['short-name'] + " university", end='\n\n')
        print(facts['about'])
    elif "location of rmit" in question_asked \
        or "location" in question_asked \
        and ("rmit" in question_asked \
        or "uni" in question_asked \
        or "university" in question_asked) \
        or "address of rmit" in question_asked \
        or "address" in question_asked \
        and ("rmit" in question_asked \
        or "uni" in question_asked \
        or "university" in question_asked):

        print(facts['short-name'] +  "university is situated at ", end='')
        print(facts['location'])
    elif "year of rmit" in question_asked \
        or "year" in question_asked \
        and ("rmit" in question_asked \
        or "uni" in question_asked \
        or "university" in question_asked):

        print(facts['short-name'] +  " was first established in ", end='')
        print(facts['year'])
    elif "website of rmit" in question_asked \
        or "website" in question_asked \
        and ("rmit" in question_asked \
        or "uni" in question_asked \
        or "university" in question_asked):

        print(facts['short-name'] +  "'s website is ", end='')
        print(facts['website'])
    elif "students of rmit" in question_asked \
        or "students" in question_asked \
        and ("rmit" in question_asked \
        or "uni" in question_asked \
        or "university" in question_asked \
        or "number" in question_asked):

        print("The number of students at " + facts['short-name'] + " is ", end='')
        print(facts['student-number'])
    elif "undergraduates of rmit" in question_asked \
        or "undergrads of rmit" in question_asked \
        or "undergraduates" in question_asked \
        and ("rmit" in question_asked \
        or "uni" in question_asked \
        or "university" in question_asked \
        or "number" in question_asked) \
        or "undergrads" in question_asked \
        and ("rmit" in question_asked \
        or "uni" in question_asked \
        or "university" in question_asked \
        or "number" in question_asked) :

        print("The number of undergraduate students at " + facts['short-name'] + " is ", end='')
        print(facts['undergraduates-number'])  
    elif "postgraduates of rmit" in question_asked \
        or "postgrads of rmit" in question_asked \
        or "postgraduates" in question_asked \
        and ("rmit" in question_asked \
        or "uni" in question_asked \
        or "university" in question_asked \
        or "number" in question_asked) \
        or "postgrads" in question_asked \
        and ("rmit" in question_asked \
        or "uni" in question_asked \
        or "university" in question_asked \
        or "number" in question_asked) :

        print("The number of postgraduate students at " + facts['short-name'] + " is ", end='')
        print(facts['postgraduates-number']) 
    elif "doctorals of rmit" in question_asked \
        or "doctorals" in question_asked \
        and ("rmit" in question_asked \
        or "uni" in question_asked \
        or "university" in question_asked \
        or "university" in question_asked):

        print("The number of doctoral students at " + facts['short-name'] + " is ", end='')
        print(facts['doctoral'])
    elif "other students of rmit" in question_asked \
        or "other students" in question_asked \
        and ("rmit" in question_asked \
        or "uni" in question_asked \
        or "university" in question_asked
        or "number" in question_asked):

        print("The number of other students at " + facts['short-name'] + " is ", end='')
        print(facts['other-student'])    
    elif "who" in question_asked \
        and ("chancellor" in question_asked):

        print(facts['short-name'] +  "'s chancellor is ", end='')
        print(facts['chancellor'])

    elif "who" in question_asked \
        and ("vice chancellor" in question_asked
        or "vc" in question_asked):

        print(facts['short-name'] +  "'s chancellor is ", end='')
        print(facts['vice-chancellor'])
    elif "what" in question_asked \
        and ("schools" in question_asked):

        print(facts['short-name'] +  "'s schools are ", end='\n\n')
        
        for school in facts['schools']:
            print(school)
    elif "what" in question_asked \
        and ("colleges" in question_asked):

        print(facts['short-name'] +  "'s colleges are", end='\n\n')

        for school in facts['colleges']:
            print(school)
    elif "end" in question_asked \
        or "exit" in question_asked \
        or "quit" in question_asked \
        or 'q' in question_asked \
        or len(question_asked) == 0:

        running = False
    else:
        print("I have no knowledge or I'm not sure what you mean?") 
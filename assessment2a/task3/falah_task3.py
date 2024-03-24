class Node:
    instance_count = 0
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.neighbours = list()
        
        Node.instance_count += 1

def start():

    # Sample image represented as 2D array of numbers
    
    # 0 are obstacles
    # 1 are free or empty spaces
    
    # A complete valid image is when the 1s have been converted to 2s once the graph.floodfill algorithm is complete.

    image = [
                [1,0,0,0,0],
                [1,1,0,1,0],
                [0,1,1,1,0],
                [0,1,1,1,1],
                [0,1,1,0,1]
            ]

    graph = Graph()

    print(graph.floodFill(image, 2, 2, 2))
    
    # for row in graph.floodFill(image, 2, 2, 2):
    #     print(row)
    
    # DO NOT REMOVE THIS RETURN STATEMENT!

    return graph

class Graph:    
    def floodFill(self, image, x, y, pixel):
        start = Node(x, y)

        queue = list()
        
        image[y][x] = pixel

        queue.append(start)

        # Remember to add unique x and y coordinates of newly discovered nodes to the explored set
        
        # Be mindful of the ordering of the search

        ###
        ### YOUR CODE HERE
        ###
        count = 0
        edges = 0
        while len(queue) > 0:
            count += 1
            currentNode = queue.pop(0)
            image[currentNode.y][currentNode.x] = pixel
            currentNode.neighbours = self.neighbours(image, currentNode.x, currentNode.y, currentNode)
            
            for adjacent_neighbour in currentNode.neighbours:
                if image[adjacent_neighbour.y][adjacent_neighbour.x] != pixel:
                    queue.append(adjacent_neighbour)
                    edges += 1
            print("This is a new iteration. {}\nEdges: {}\nCurrent node instance count is: {}".format(count, edges, Node.instance_count)) #TESTING
            for row in image: #TESTING 
                print(row)
            print()
                    
                # Tried to make code more readable by introducing new variables
                # for whether a pixel was occupied or not, but it just made things
                # more confusing...
                
#                 occupied = False
#                 if image[adjacent_neighbour.y][adjacent_neighbour.x] == pixel:
#                     occupied = True
                    
#                 if not occupied:
#                     image[adjacent_neighbour.y][adjacent_neighbour.x] = pixel
#                     queue.append(adjacent_neighbour)

                # Tests output on each iteration

#             for row in image:
#                 print(row)
            
#             print()

        # Return the modified image represented as 2D array of numbers

        return image
    def neighbours(self, image, x, y, currentNode):
            U = y - 1
            D = y + 1
            L = x - 1
            R = x + 1
            
            # Write the neighbours function to find the neighbours in four directions for a given pixel.
            
            # An edge is valid if the pixel is newly discovered, i.e. an edge is created when the neighbour's pixel value is one.

            # Append a valid new Node to the neighbours of the currentNode

            # Remember to do boundary checking

            ###
            ### YOUR CODE HERE
            ###
            
            # Variables to set up the boundaries of the image
            top_bound = 0
            right_bound = len(image[0]) - 1
            bottom_bound = len(image) - 1
            left_bound = 0
            
            # Make sure that for each adjacent spot checked, it is
            # both available and within the boundaries of the image
            if U >= top_bound and image[U][x] == 1:
                currentNode.neighbours.append(Node(x, U))

            if D <= bottom_bound and image[D][x] == 1:
                currentNode.neighbours.append(Node(x, D))

            if L >= left_bound and image[y][L] == 1:
                currentNode.neighbours.append(Node(L, y))
                
            if R <= right_bound and image[y][R] == 1:
                currentNode.neighbours.append(Node(R, y))
            
            # Return the current node's (the pixel in question) neighbours, not always a maximum of four.

            return currentNode.neighbours

graph = start()
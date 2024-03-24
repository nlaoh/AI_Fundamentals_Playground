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
    # image = [
    #             [0,1,1,1,1,1,1,1,1,1,1,0],
    #             [0,1,1,1,1,1,1,1,1,1,1,0],
    #             [0,1,1,1,1,1,1,1,1,1,1,0],
    #             [0,1,1,1,1,1,1,1,1,1,1,0],
    #             [0,1,1,1,1,1,1,1,1,1,1,0]
    #         ]

    graph = Graph()

    ##TESTING CODE : SHOWING GRID
    # print("Image before floodfill:")
    # for row in image:
    #     print(row)
    # print()
    # print("Updated image:")
    # for row in graph.floodFill(image, 2, 2, 2):
    #     print(row)
    # print()
    print(graph.floodFill(image, 2, 2, 2)) ##START IN THE MIDDLE
    
    # DO NOT REMOVE THIS RETURN STATEMENT!

    return graph


class Graph:    
    explored = set()

    def floodFill(self, image, x, y, pixel):
        start = Node(x,y)
        queue = list()
        
        image[y][x] = pixel #Sets value of starting node to pixel
        queue.append(start)
        self.explored.add((x, y))
        # for row in image:
        #     print(row)
        # print()
        # Remember to add unique x and y coordinates of newly discovered nodes to the explored set
        
        # Be mindful of the ordering of the search

        ###
        ### YOUR CODE HERE
        ###
        count = 0
        while len(queue) > 0:
            count += 1
            currentNode = queue.pop(0)
            image[currentNode.y][currentNode.x] = pixel
            print("This is a new iteration. {}\nCurrent node instance count is: {}".format(count, Node.instance_count)) #TESTING
            for row in image: #TESTING 
                print(row)
            print()
            currentNode.neighbours = self.neighbours(image, currentNode.x, currentNode.y, currentNode)
            for adj in currentNode.neighbours:
                queue.append(adj)
                                        
        # Return the modified image represented as 2D array of numbers

        return image

    def neighbours(self, image, x, y, currentNode): #Traverse?
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
        rows = len(image) - 1
        columns = len(image[0]) - 1

        if U >= 0 and image[U][x] == 1 and ((x, U) not in self.explored): #Check up
            currentNode.neighbours.append(Node(x, U))
            self.explored.add((x, U))
        if D <= rows and image[D][x] == 1 and ((x, D) not in self.explored): #Check down
            currentNode.neighbours.append(Node(x, D))
            self.explored.add((x, D))
        if L >= 0 and image[y][L] == 1 and ((L, y) not in self.explored): #Check left
            currentNode.neighbours.append(Node(L, y))
            self.explored.add((L, y))
        if R <= columns and image[y][R] == 1 and ((R, y) not in self.explored): #Check right
            currentNode.neighbours.append(Node(R, y))
            self.explored.add((R, y))
        # Return the current node's (the pixel in question) neighbours, not always a maximum of four.

        return currentNode.neighbours


graph = start()
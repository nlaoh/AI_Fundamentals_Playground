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
            start = Node(x, y)
            queue = []
            visited = [[False for _ in range(len(image[0]))] for _ in range(len(image))]

            queue.append(start)
            visited[y][x] = True

            # for row in image:
            #     print(row)
            # print()
            # Remember to add unique x and y coordinates of newly discovered nodes to the explored set

            # Be mindful of the ordering of the search

            ###
            ### YOUR CODE HERE
            ###
            while len(queue) > 0:
                currentNode = queue.pop(0)
                image[currentNode.y][currentNode.x] = pixel
                # if image[currentNode.y][currentNode.x] == pixel:
                #     break
                currentNode.neighbours = self.neighbours(image, currentNode.x, currentNode.y, currentNode)
                for adj in currentNode.neighbours:
                    if (adj.x, adj.y) not in self.explored:
                        queue.append(adj)
                        self.explored.add((adj.x, adj.y))
                print("This is a new iteration.\nCurrent node instance count is: {}", Node.instance_count) #TESTING
                for row in image: #TESTING 
                    print(row)
                print()
                    # if image[adj.y][adj.x] == 1:
                    #     image[adj.y][adj.x] = pixel
                    #     queue.append(adj)

            # Return the modified image represented as 2D array of numbers

            return image

        def neighbours(self, image, x, y, currentNode):
            U = y - 1
            D = y + 1
            L = x - 1
            R = x + 1

            rows = len(image) - 1
            columns = len(image[0]) - 1

            neighbor_nodes = []

            if U >= 0 and image[U][x] == 1 and ((x, U) not in self.explored):
                neighbor_nodes.append(Node(x, U))
            if D <= rows and image[D][x] == 1 and ((x, D) not in self.explored):
                neighbor_nodes.append(Node(x, D))
            if L >= 0 and image[y][L] == 1 and ((L, y) not in self.explored):
                neighbor_nodes.append(Node(L, y))
            if R <= columns and image[y][R] == 1 and ((R, y) not in self.explored):
                neighbor_nodes.append(Node(R, y))

            return neighbor_nodes


graph = start()
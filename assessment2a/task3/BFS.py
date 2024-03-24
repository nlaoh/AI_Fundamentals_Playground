class Node:
    def __init__(self, name):
        self.name = name
        self.visited = False
        self.neighbours = list()

def backtrace(parent, start, end): 
    path = [end]
    
    while path[-1].name != start.name: #while the last isnt the start
        path.append(parent[path[-1]]) #add the last parent
        # to the end of the path
        
    path.reverse() #reverse array
    
    return path

def traverse(root, end):
    
    parent = dict()
    # parent = vector<>
    
    # Treat like a queue
    queue = list() 

    root.visited = True
    queue.append(root)

    # Keep looping until the queue
    while len(queue) > 0:
        currentNode = queue.pop(0) #choose the first item from the queue and remove it from the queue
        
        if currentNode.name == end.name:
            return backtrace(parent, root, end)
        
        currentNode.visited = True

        for adjacent_neighbour in currentNode.neighbours: 
            if not adjacent_neighbour.visited:
                parent[adjacent_neighbour] = currentNode
                queue.append(adjacent_neighbour)

def display(path):
    print('Path: ', end='')
    for node in path:
        print(node.name + ' ', end='')


a = Node('A')
b = Node('B')
c = Node('C')
d = Node('D')
e = Node('E')
f = Node('F')
g = Node('G')
h = Node('H')

a.neighbours.append(b)
a.neighbours.append(f)
a.neighbours.append(g)

b.neighbours.append(a)
b.neighbours.append(c)
b.neighbours.append(d)

c.neighbours.append(b)

d.neighbours.append(b)
d.neighbours.append(e)

f.neighbours.append(a)

g.neighbours.append(a)
g.neighbours.append(h)

h.neighbours.append(g)

display(traverse(a, d))
